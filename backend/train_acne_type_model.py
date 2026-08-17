"""
Train MobileNetV2 3-class acne classifier (no_acne, comedonal_acne, inflammatory_acne)
using clean, noise-filtered dataset.

Usage:
  python train_acne_type_model.py --data-dir data/acne_clean

Saves:
  models/acne_type_model.keras
  models/acne_type_classes.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 25
MODEL_SAVE_PATH = os.path.join('models', 'acne_type_model.keras')
CLASSES_SAVE_PATH = os.path.join('models', 'acne_type_classes.json')

CLASSES = ['no_acne', 'comedonal_acne', 'inflammatory_acne']
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_paths_and_labels(data_dir: str):
    data_dir = Path(data_dir)
    paths, labels = [], []

    for cls in CLASSES:
        cls_folder = data_dir / cls
        if not cls_folder.exists():
            continue
        for img_path in cls_folder.glob('*'):
            if img_path.suffix.lower() in _IMG_EXTS:
                paths.append(str(img_path))
                labels.append(cls)

    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}.")

    label_indices = [CLASSES.index(l) for l in labels]
    print(f"\n[1/3] Clean 3-Class Acne dataset summary:")
    for cls in CLASSES:
        print(f"  • {cls:20s}: {labels.count(cls)} images")
    print(f"  • Total images        : {len(paths)}")
    return paths, label_indices


def build_dataset(paths, labels, num_classes, augment=False):
    labels_oh = tf.one_hot(labels, num_classes)

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), labels_oh))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int = 3) -> Model:
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
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
    print(f"  • Train: {len(train_paths)}  |  Validation: {len(val_paths)}")

    cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    class_weight_dict = {i: float(cw[i]) for i in range(num_classes)}
    print(f"  • Class weights: { {CLASSES[i]: round(cw[i], 3) for i in range(num_classes)} }")

    train_ds = build_dataset(train_paths, train_labels, num_classes, augment=True)
    val_ds   = build_dataset(val_paths,   val_labels,   num_classes, augment=False)

    model = build_model(num_classes)

    # ---- Phase 1: Train Head -----------------------------------------------
    print(f"\n[2/3] {'='*55}")
    print(" Phase 1: Training 3-Class Head (Base Frozen)")
    print('='*60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight_dict,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
            ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ],
    )

    # ---- Phase 2: Fine-Tuning ---------------------------------------------
    print(f"\n[3/3] {'='*55}")
    print(" Phase 2: Fine-Tuning Top MobileNetV2 Layers")
    print('='*60)

    base_layer = model.layers[1]
    base_layer.trainable = True
    for layer in base_layer.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
            ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ],
    )

    with open(CLASSES_SAVE_PATH, 'w') as f:
        json.dump(CLASSES, f)

    # ---- Evaluation Report on Validation Set -------------------------------
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    def load_val(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        return preprocess_input(img)

    val_imgs_ds = tf.data.Dataset.from_tensor_slices(tf.constant(val_paths)).map(load_val).batch(BATCH_SIZE)
    preds = best_model.predict(val_imgs_ds, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    print("\n" + "="*60)
    print(" 🎯 FINAL 3-CLASS ACNE MODEL VALIDATION REPORT")
    print("="*60)
    print(classification_report(val_labels, pred_labels, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels, pred_labels))
    print(f"\n✅ Model saved to:  {MODEL_SAVE_PATH}")
    print(f"✅ Classes saved to: {CLASSES_SAVE_PATH}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileNetV2 3-class acne classifier (TF/Keras)')
    parser.add_argument('--data-dir', default='data/acne_clean')
    args = parser.parse_args()
    train(args.data_dir)