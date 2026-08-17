"""
Train all 3 MobileNetV2 models from scratch on the Curated Golden Datasets.

Models:
1. Acne Type Classifier (no_acne, comedonal_acne, inflammatory_acne)
2. Skin Type Classifier (oily, normal, dry)
3. Hyperpigmentation Classifier (positive, negative)
"""

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
BATCH_SIZE = 16
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def build_model(num_classes: int, is_binary: bool = False) -> Model:
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
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    if is_binary:
        outputs = Dense(1, activation='sigmoid')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
        
    return Model(inputs, outputs)


def train_single_model(data_dir: str, classes: list[str], save_model_path: str, save_classes_path: str, is_binary: bool = False):
    print(f"\n{'='*70}")
    print(f" 🚀 Training Model: {os.path.basename(save_model_path)}")
    print(f"    Data Directory: {data_dir}")
    print(f"    Classes       : {classes}")
    print(f"{'='*70}")

    paths, labels = [], []
    for cls in classes:
        folder = Path(data_dir) / cls
        for p in folder.glob('*'):
            if p.suffix.lower() in _IMG_EXTS:
                paths.append(str(p))
                labels.append(cls)

    label_indices = [classes.index(l) for l in labels]
    num_classes = len(classes)
    
    print(f"Dataset breakdown: { {c: labels.count(c) for c in classes} } | Total: {len(paths)}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, label_indices, test_size=0.2, stratify=label_indices, random_state=42
    )

    cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    cw_dict = {i: float(cw[i]) for i in range(num_classes)}

    def build_ds(p_list, l_list, augment=False):
        if is_binary:
            y_t = tf.constant(l_list, dtype=tf.float32)
        else:
            y_t = tf.one_hot(l_list, num_classes)

        def load(p, y):
            raw = tf.io.read_file(p)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img)
            return img, y

        def aug(img, y):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.10)
            img = tf.image.random_contrast(img, 0.90, 1.10)
            return img, y

        ds = tf.data.Dataset.from_tensor_slices((tf.constant(p_list), y_t)).map(load, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.shuffle(300).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = build_ds(train_paths, train_labels, augment=True)
    val_ds = build_ds(val_paths, val_labels, augment=False)

    model = build_model(num_classes, is_binary=is_binary)
    loss_fn = 'binary_crossentropy' if is_binary else 'categorical_crossentropy'
    
    # Phase 1: Train Top Head
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=['accuracy'])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        class_weight=cw_dict,
        verbose=1,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0),
            ModelCheckpoint(save_model_path, monitor='val_accuracy', save_best_only=True, verbose=0)
        ]
    )

    # Phase 2: Fine Tune Top 25 Layers
    base_layer = model.layers[1]
    base_layer.trainable = True
    for l in base_layer.layers[:-25]:
        l.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss=loss_fn, metrics=['accuracy'])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=cw_dict,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0),
            ModelCheckpoint(save_model_path, monitor='val_accuracy', save_best_only=True, verbose=0)
        ]
    )

    if save_classes_path:
        with open(save_classes_path, 'w') as f:
            json.dump(classes, f)

    # Validation Evaluation
    best_m = tf.keras.models.load_model(save_model_path)
    def load_eval(p):
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return preprocess_input(tf.cast(img, tf.float32))

    eval_ds = tf.data.Dataset.from_tensor_slices(tf.constant(val_paths)).map(load_eval).batch(BATCH_SIZE)
    preds = best_m.predict(eval_ds, verbose=0)
    
    if is_binary:
        pred_labels = (preds.flatten() > 0.5).astype(int)
    else:
        pred_labels = np.argmax(preds, axis=1)

    print("\n🎯 VALIDATION REPORT:")
    print(classification_report(val_labels, pred_labels, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels, pred_labels))
    print(f"✅ Saved {save_model_path}\n")


def main():
    os.makedirs('models', exist_ok=True)
    
    # 1. Acne Model
    train_single_model(
        data_dir='data/golden/acne',
        classes=['no_acne', 'comedonal_acne', 'inflammatory_acne'],
        save_model_path='models/acne_type_model.keras',
        save_classes_path='models/acne_type_classes.json',
        is_binary=False
    )

    # 2. Skin Type Model
    train_single_model(
        data_dir='data/golden/skin_type',
        classes=['oily', 'normal', 'dry'],
        save_model_path='models/skin_type_model.keras',
        save_classes_path='models/skin_type_classes.json',
        is_binary=False
    )

    # 3. Hyperpigmentation Model
    train_single_model(
        data_dir='data/golden/hyperpigmentation',
        classes=['negative', 'positive'],
        save_model_path='models/hyperpigmentation_model.keras',
        save_classes_path=None,
        is_binary=True
    )

if __name__ == '__main__':
    main()
