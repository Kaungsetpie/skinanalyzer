import os
import json
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    print("TensorFlow not available — skin type classifier disabled.")

MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'skin_type_model.keras')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'skin_type_classes.json')

_model   = None
_classes = None


def _load_model():
    global _model, _classes
    if _model is not None:
        return _model, _classes
    if not _TF_AVAILABLE or not os.path.exists(MODEL_PATH):
        return None, None
    try:
        _model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH) as f:
            _classes = json.load(f)
        print(f"Loaded skin-type model  classes={_classes}")
        return _model, _classes
    except Exception as e:
        print(f"Failed to load skin-type model: {e}")
        return None, None


def _preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert('RGB').resize((224, 224))
    arr = preprocess_input(np.array(img, dtype=np.float32))
    return np.expand_dims(arr, axis=0)


def classify_skin_type(pil_image: Image.Image) -> tuple[str | None, dict[str, float]]:
    """Returns (predicted_class, {class: confidence}). Falls back to (None, {}) if model not loaded."""
    model, classes = _load_model()
    if model is None:
        return None, {}
    probs = model.predict(_preprocess(pil_image), verbose=0)[0]
    scores = {cls: float(p) for cls, p in zip(classes, probs)}
    predicted = classes[int(np.argmax(probs))]
    return predicted, scores
