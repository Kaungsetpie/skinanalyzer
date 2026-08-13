"""
Train MobileNetV2 acne-type classifier using TensorFlow/Keras.
Classes: comedonal_acne (acne0 / Grade 0), inflammatory_acne (acne1 / Grade 1).
MobileNetV2 standard preprocessing [-1, 1].

Usage:
  python train_acne_type_model.py --data-dir data/ACNE04/train

Saves:  models/acne_type_model.keras
        models/acne_type_classes.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE          = 224
BATCH_SIZE        = 32
EPOCHS_HEAD       = 15
EPOCHS_FINE       = 25
MODEL_SAVE_PATH   = os.path.join('models', 'acne_type_model.keras')
CLASSES_SAVE_PATH = os.path.join('models', 'acne_type_classes.json')

# Class 0: Comedonal Acne (acne0_1024), Class 1: Inflammatory Acne (acne1_1024)
CLASSES           = ['comedonal_acne', 'inflammatory_acne']

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Folder Mapping အသစ် (acne0 & acne1 သီးသန့်)
_GRADE_TO_CLASS = {
    'acne0_1024': 'comedonal_acne',
    'acne1_1024': 'inflammatory_acne',
    'Grade_0':    'comedonal_acne',
    'Grade_1':    'inflammatory_acne',
    'acne0':      'comedonal_acne',
    'acne1':      'inflammatory_acne',
}


def _images_in(folder: Path) -> list:
    return [p for p in folder.rglob('*') if p.suffix.lower() in _IMG_EXTS]


def load_paths_and_labels(data_dir: str):
    data_dir = Path(data_dir)
    paths, labels = [], []

    for folder_name, class_name in _GRADE_TO_CLASS.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            matches = [m for m in data_dir.rglob(folder_name) if m.is_dir()]
            if not matches:
                continue
            folder = matches[0]
        for img_path in _images_in(folder):
            paths.append(str(img_path))
            labels.append(class_name)

    if not paths:
        raise FileNotFoundError(f"No valid acne0/acne1 images found in {data_dir}.")

    label_indices = [CLASSES.index(l) for l in labels]
    print(f"\nAcne-type dataset from {data_dir}")
    for cls in CLASSES:
        print(f"  {cls:20s}: {labels.count(cls)} images")
    print(f"  Total: {len(paths)}")
    return paths, label_indices


def build_dataset(paths, labels, num_classes, augment=False):
    labels_oh = tf.one_hot(labels, num_classes)

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        # ✅ MobileNetV2 Preprocessing standard သို့ ပြောင်းလဲခြင်း
        img = preprocess_input(img)
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), labels_oh))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int) -> Model:
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)


def train(data_dir: str) -> None:
    os.makedirs('models', exist_ok=True)

    paths, label_indices = load_paths_and_labels(data_dir)
    num_classes = len(CLASSES)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, label_indices, test_size=0.2, stratify=label_indices, random_state=42
    )

    cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    class_weight_dict = {i: float(cw[i]) for i in range(num_classes)}
    print(f"Class weights: { {CLASSES[i]: f'{cw[i]:.3f}' for i in range(num_classes)} }")

    train_ds = build_dataset(train_paths, train_labels, num_classes, augment=True)
    val_ds   = build_dataset(val_paths,   val_labels,   num_classes, augment=False)

    model = build_model(num_classes)

    print(f"\n{'='*60}\nPhase 1: Training head\n{'='*60}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
              class_weight=class_weight_dict,
              callbacks=[
                 # EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
                  #ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
                  ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True),
              ])

    print(f"\n{'='*60}\nPhase 2: Fine-tuning last 4 blocks\n{'='*60}")
    base_layer = model.layers[1]
    base_layer.trainable = True
    for layer in base_layer.layers[:-12]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(3e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
              class_weight=class_weight_dict,
              callbacks=[
                 # EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
                #ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
                  ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True),
              ])

    with open(CLASSES_SAVE_PATH, 'w') as f:
        json.dump(CLASSES, f)
    print(f"\nModel saved to:   {MODEL_SAVE_PATH}")
    print(f"Classes saved to: {CLASSES_SAVE_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileNetV2 acne-type classifier (TF/Keras)')
    parser.add_argument('--data-dir', default='data/ACNE04/train')
    train(parser.parse_args().data_dir)