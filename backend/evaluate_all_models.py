import os
from pathlib import Path
import tensorflow as tf
import numpy as np

MODELS_DIR = 'models'
DATA_DIR = 'data'
IMG_SIZE = 224
BATCH_SIZE = 32
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def evaluate_single_model(model_path, folder_dict, is_binary=False):
    if not os.path.exists(model_path):
        return "❌ File Not Found"

    model = tf.keras.models.load_model(model_path)
    
    paths, labels = [], []
    for label, folder_path in folder_dict.items():
        p_path = Path(folder_path)
        if p_path.exists():
            imgs = [str(p) for p in p_path.rglob('*') if p.suffix.lower() in _IMG_EXTS]
            paths.extend(imgs)
            labels.extend([label] * len(imgs))

    if not paths:
        return "⚠️ Dataset missing"

    def load_img(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(labels)))
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

    # Re-compile properly according to type
    if is_binary:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    loss, accuracy = model.evaluate(ds, verbose=0)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    total_params = model.count_params()

    return {
        "size": file_size_mb,
        "params": total_params,
        "loss": loss,
        "accuracy": accuracy * 100
    }

configs = {
    "Skin Type Model": {
        "file": "skin_type_model.keras",
        "binary": False,
        "folders": {0: f"{DATA_DIR}/skin_type_preprocessed/oily", 1: f"{DATA_DIR}/skin_type_preprocessed/normal", 2: f"{DATA_DIR}/skin_type_preprocessed/dry"}
    },
    "Acne Type Model": {
        "file": "acne_type_model.keras",
        "binary": False,
        "folders": {0: f"{DATA_DIR}/comedonal", 1: f"{DATA_DIR}/inflammatory"}
    },
    "Severity Model": {
        "file": "severity_model.keras",
        "binary": True,
        "folders": {0: f"{DATA_DIR}/inflammatory", 1: f"{DATA_DIR}/severe"}
    },
    "Hyperpigmentation Model": {
        "file": "hyperpigmentation_model.keras",
        "binary": True,
        "folders": {1: f"{DATA_DIR}/hyperpigmentation", 0: f"{DATA_DIR}/skin_type"}
    }
}

print("="*60)
print(" 📊 CORRECTED AI MODELS EVALUATION REPORT ")
print("="*60)

for model_name, cfg in configs.items():
    m_path = os.path.join(MODELS_DIR, cfg["file"])
    print(f"\n🔹 {model_name} ({cfg['file']}):")
    print("-" * 50)
    res = evaluate_single_model(m_path, cfg["folders"], cfg["binary"])
    
    if isinstance(res, dict):
        print(f"  • Status          : 🟢 Evaluated Successfully")
        print(f"  • Model File Size : {res['size']:.2f} MB")
        print(f"  • Total Parameters: {res['params']:,} weights")
        print(f"  • Loss            : {res['loss']:.4f}")
        print(f"  • Accuracy        : {res['accuracy']:.2f}%")
    else:
        print(f"  • Status          : {res}")

print("\n" + "="*60)