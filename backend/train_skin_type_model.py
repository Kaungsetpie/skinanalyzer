"""
Train MobileNetV2 skin-type classifier (oily / normal / dry) using TensorFlow/Keras.
Pixels are normalized to [0, 1] as specified in the project methodology.

--- Dataset ---
  kaggle datasets download -d shakyadivanshu/oily-and-dry-skin-dataset
  Unzip so structure is: data/skin_type/Oily/, data/skin_type/Dry/, data/skin_type/Normal/

--- Usage ---
  python train_skin_type_model.py --data-dir data/skin_type
  python train_skin_type_model.py predict path/to/face.jpg

Saves:  models/skin_type_model.keras
        models/skin_type_classes.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE         = 224
BATCH_SIZE       = 32
EPOCHS_HEAD      = 15
EPOCHS_FINE      = 25
MODEL_SAVE_PATH  = os.path.join('models', 'skin_type_model.keras')
CLASSES_SAVE_PATH = os.path.join('models', 'skin_type_classes.json')

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

_CLASS_FOLDER_NAMES = {
    'oily':   ['oily', 'Oily', 'OILY'],
    'normal': ['normal', 'Normal', 'NORMAL'],
    'dry':    ['dry', 'Dry', 'DRY'],
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _find_all_class_folders(root: Path, candidates: list) -> list:
    return [p for p in root.rglob('*') if p.is_dir() and p.name in candidates]


def _images_in(folder: Path) -> list:
    return [p for p in folder.rglob('*') if p.suffix.lower() in _IMG_EXTS]


def load_paths_and_labels(data_dir: str):
    data_dir = Path(data_dir)
    paths, labels, found_classes = [], [], []

    for class_name, candidates in _CLASS_FOLDER_NAMES.items():
        folders = _find_all_class_folders(data_dir, candidates)
        if not folders:
            print(f"  [warn] No folder found for class '{class_name}'")
            continue
        found_classes.append(class_name)
        for folder in folders:
            for img_path in _images_in(folder):
                paths.append(str(img_path))
                labels.append(class_name)

    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}.")
    if len(found_classes) < 2:
        raise ValueError("Need at least 2 classes to train.")

    label_indices = [found_classes.index(l) for l in labels]
    print(f"\nSkin-type dataset from {data_dir}")
    for cls in found_classes:
        n = labels.count(cls)
        print(f"  {cls:8s}: {n} images")
    print(f"  Total: {len(paths)} images  |  Classes: {found_classes}")
    return paths, label_indices, found_classes


# ---------------------------------------------------------------------------
# tf.data pipeline — normalize [0,255] → [0,1]
# ---------------------------------------------------------------------------

def build_dataset(paths, labels, num_classes, augment=False):
    paths_t  = tf.constant(paths)
    labels_t = tf.one_hot(labels, num_classes)

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.3)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model — MobileNetV2 + custom head
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> Model:
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_dir: str) -> None:
    os.makedirs('models', exist_ok=True)

    paths, label_indices, classes = load_paths_and_labels(data_dir)
    num_classes = len(classes)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, label_indices, test_size=0.2, stratify=label_indices, random_state=42
    )
    print(f"\nTrain: {len(train_paths)}  Val: {len(val_paths)}")

    cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    class_weight_dict = {i: float(cw[i]) for i in range(num_classes)}
    print(f"Class weights: { {classes[i]: f'{cw[i]:.3f}' for i in range(num_classes)} }")

    train_ds = build_dataset(train_paths, train_labels, num_classes, augment=True)
    val_ds   = build_dataset(val_paths,   val_labels,   num_classes, augment=False)

    model = build_model(num_classes)

    # ---- Phase 1: head only -----------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 1: Training head (MobileNetV2 features frozen)")
    print('='*60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight_dict,
        callbacks=[
            #EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
           # ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3),
            ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
        ],
    )

    # ---- Phase 2: fine-tune last 4 feature blocks -------------------------
    print(f"\n{'='*60}")
    print("Phase 2: Fine-tuning last 4 MobileNetV2 blocks")
    print('='*60)

    base_layer = model.layers[1]  # the MobileNetV2 base
    base_layer.trainable = True
    for layer in base_layer.layers[:-4 * 3]:  # freeze all except last ~4 blocks
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
           # ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3),
            #ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
        ],
    )

    with open(CLASSES_SAVE_PATH, 'w') as f:
        json.dump(classes, f)
    print(f"\nModel saved to:  {MODEL_SAVE_PATH}")
    print(f"Classes saved to: {CLASSES_SAVE_PATH}")
    print("Restart the backend server to load the new model.")


# ---------------------------------------------------------------------------
# Quick single-image prediction
# ---------------------------------------------------------------------------

def predict(image_path: str) -> None:
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH} — train first.")
        return

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    with open(CLASSES_SAVE_PATH) as f:
        classes = json.load(f)

    img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]

    print(f"\n{image_path}")
    for cls, prob in zip(classes, probs):
        bar = '#' * int(prob * 30)
        print(f"  {cls:8s}: {prob:.3f}  {bar}")
    print(f"\n  Prediction: {classes[int(np.argmax(probs))].upper()}")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileNetV2 skin-type classifier (TF/Keras)')
    sub = parser.add_subparsers(dest='cmd')

    train_p = sub.add_parser('train')
    train_p.add_argument('--data-dir', required=True)

    pred_p = sub.add_parser('predict')
    pred_p.add_argument('image')

    parser.add_argument('--data-dir')
    args = parser.parse_args()

    if args.cmd == 'predict':
        predict(args.image)
    elif args.cmd == 'train':
        train(args.data_dir)
    elif getattr(args, 'data_dir', None):
        train(args.data_dir)
    else:
        parser.print_help()