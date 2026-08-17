import os
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'severity_model.keras')

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not _TF_AVAILABLE or not os.path.exists(MODEL_PATH):
        return None
    try:
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded severity model from {MODEL_PATH}")
        return _model
    except Exception as e:
        print(f"Failed to load severity model: {e}")
        return None


def _preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def classify_severity(pil_image: Image.Image) -> tuple[bool, float]:
    """
    Returns (is_severe, confidence 0-1).
    Requires a trained severity_model.keras. Returns (False, 0.0) when no model is present
    so the pipeline proceeds to normal recommendations rather than falsely alarming the user.
    """
    model = _load_model()
    if model is None:
        return False, 0.0
    score = float(model.predict(_preprocess(pil_image), verbose=0)[0][0])
    return score > 0.70, score
