"""
Train MobileNetV2 binary hyperpigmentation classifier using TensorFlow/Keras.
Pixels normalized to [0, 1].

--- Dataset ---
  Positives: muhammadalirhojab/melasma  (all images are hyperpigmentation)
  Negatives: normal skin images from data/skin_type/

--- Usage ---
  python train_hyperpigmentation_model.py --data-dir data/melasma --normal-dir data/skin_type

Saves:  models/hyperpigmentation_model.keras
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS_HEAD     = 15
EPOCHS_FINE     = 25
MODEL_SAVE_PATH = os.path.join('models', 'hyperpigmentation_model.keras')

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _images_in(folder: Path) -> list:
    return [p for p in folder.rglob('*') if p.suffix.lower() in _IMG_EXTS]


def load_paths_and_labels(data_dir: str, normal_dir: str | None):
    positives = _images_in(Path(data_dir))
    print(f"Positives (hyperpigmentation): {len(positives)}")

    negatives = []
    if normal_dir:
        n_dir = Path(normal_dir)
        for name in ['Normal', 'normal', 'dry', 'Dry', 'oily', 'Oily']:
            folder = n_dir / name
            if folder.is_dir():
                negatives.extend(_images_in(folder))
        max_neg = min(len(negatives), 4 * len(positives))
        rng = np.random.default_rng(42)
        negatives = list(rng.choice(negatives, size=max_neg, replace=False))
    print(f"Negatives (normal skin):       {len(negatives)}")

    paths  = [str(p) for p in positives] + [str(p) for p in negatives]
    labels = [1] * len(positives) + [0] * len(negatives)
    print(f"Total: {len(paths)}")
    return paths, labels


def build_dataset(paths, labels, augment=False):
    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.float32)

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.3)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(labels)))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> Model:
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def train(data_dir: str, normal_dir: str | None) -> None:
    os.makedirs('models', exist_ok=True)

    paths, labels = load_paths_and_labels(data_dir, normal_dir)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    class_weight_dict = {0: 1.0, 1: pos_weight}
    print(f"pos_weight={pos_weight:.2f}")

    train_ds = build_dataset(train_paths, train_labels, augment=True)
    val_ds   = build_dataset(val_paths,   val_labels,   augment=False)

    model = build_model()

    print(f"\n{'='*60}\nPhase 1: Training head\n{'='*60}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
              class_weight=class_weight_dict,
              callbacks=[
                  EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
                  ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3),
                  ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
              ])

    print(f"\n{'='*60}\nPhase 2: Fine-tuning last 4 blocks\n{'='*60}")
    base_layer = model.layers[1]
    base_layer.trainable = True
    for layer in base_layer.layers[:-12]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-6),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
              class_weight=class_weight_dict,
              callbacks=[
                  EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
                  ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3),
                  ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
              ])

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hyperpigmentation classifier (TF/Keras)')
    parser.add_argument('--data-dir',   required=True)
    parser.add_argument('--normal-dir', default=None)
    args = parser.parse_args()
    train(args.data_dir, args.normal_dir)
