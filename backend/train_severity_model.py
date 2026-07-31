"""
Train MobileNetV2 binary severity classifier using TensorFlow/Keras.

--- Dataset (ACNE04) ---
  kaggle datasets download -d rutvikdeshpande/acne04-dataset
  unzip acne04-dataset.zip -d data/ACNE04

  Severity mapping:
    Grade 0 / Grade 1  →  not severe (0)
    Grade 2 / Grade 3  →  severe     (1)

--- Usage ---
  python train_severity_model.py --data-dir data/ACNE04

  OR with explicit positive/negative folders:
  python train_severity_model.py --positive-dir data/severe --negative-dir data/normal

Saves:  models/severity_model.keras
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
MODEL_SAVE_PATH = os.path.join('models', 'severity_model.keras')

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

_GRADE_FOLDERS = {
    0: ['Grade_0', 'grade0', 'grade_0', '0', 'clear'],
    1: ['Grade_1', 'grade1', 'grade_1', '1', 'mild'],
    2: ['Grade_2', 'grade2', 'grade_2', '2', 'moderate'],
    3: ['Grade_3', 'grade3', 'grade_3', '3', 'severe'],
}
_GRADE_TXT = {
    0: ['Grade_0.txt', 'grade0.txt'],
    1: ['Grade_1.txt', 'grade1.txt'],
    2: ['Grade_2.txt', 'grade2.txt'],
    3: ['Grade_3.txt', 'grade3.txt'],
}


def _images_in(folder: Path) -> list:
    return [p for p in folder.rglob('*') if p.suffix.lower() in _IMG_EXTS]


def _find_folder(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    return None


def load_from_acne04(data_dir: str):
    """Load ACNE04 dataset, mapping Grade 0/1 → not severe, Grade 2/3 → severe."""
    root = Path(data_dir)
    positives, negatives = [], []

    jpeg_dir = root / 'JPEGImages'
    if jpeg_dir.is_dir():
        # Text-file layout: Grade_N.txt lists filenames
        for grade, txts in _GRADE_TXT.items():
            for txt_name in txts:
                txt = root / txt_name
                if txt.exists():
                    imgs = [jpeg_dir / ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
                    imgs = [p for p in imgs if p.exists()]
                    (positives if grade >= 2 else negatives).extend(imgs)
                    break
    else:
        # Folder-per-grade layout
        for grade, names in _GRADE_FOLDERS.items():
            folder = _find_folder(root, names)
            if folder:
                imgs = _images_in(folder)
                (positives if grade >= 2 else negatives).extend(imgs)

    if not positives and not negatives:
        raise ValueError(f"No images found in {data_dir}. Check the ACNE04 folder structure.")

    print(f"ACNE04 — Grade 2+3 (severe): {len(positives)}  |  Grade 0+1 (normal): {len(negatives)}")
    return [str(p) for p in positives], [str(p) for p in negatives]


def load_paths_and_labels(positive_dir: str | None, negative_dir: str | None,
                          data_dir: str | None):
    if data_dir:
        pos_paths, neg_paths = load_from_acne04(data_dir)
    else:
        pos_paths = [str(p) for p in _images_in(Path(positive_dir))]
        neg_paths = [str(p) for p in _images_in(Path(negative_dir))]

    # Cap negatives at 4× positives to avoid extreme imbalance
    rng = np.random.default_rng(42)
    max_neg = min(len(neg_paths), 4 * len(pos_paths))
    if len(neg_paths) > max_neg:
        neg_paths = list(rng.choice(neg_paths, size=max_neg, replace=False))

    print(f"Training — Positives (severe): {len(pos_paths)}  |  Negatives (normal): {len(neg_paths)}")

    paths  = pos_paths + neg_paths
    labels = [1] * len(pos_paths) + [0] * len(neg_paths)
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
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
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


def train(positive_dir: str | None, negative_dir: str | None, data_dir: str | None) -> None:
    os.makedirs('models', exist_ok=True)

    paths, labels = load_paths_and_labels(positive_dir, negative_dir, data_dir)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    class_weight_dict = {0: 1.0, 1: pos_weight}

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
    parser = argparse.ArgumentParser(description='Train severity classifier (TF/Keras)')
    parser.add_argument('--data-dir', help='Path to ACNE04 dataset root (Grade_0…Grade_3 folders)')
    parser.add_argument('--positive-dir', help='Path to severe/positive images')
    parser.add_argument('--negative-dir', help='Path to normal/negative images')
    args = parser.parse_args()

    if not args.data_dir and not (args.positive_dir and args.negative_dir):
        parser.error('Provide either --data-dir (ACNE04) or both --positive-dir and --negative-dir')

    train(args.positive_dir, args.negative_dir, args.data_dir)
