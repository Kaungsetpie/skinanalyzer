import os
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hyperpigmentation_model.pt')

_model = None

_transform = None
if _TORCH_AVAILABLE:
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _build_model() -> "nn.Module":
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )
    return model


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not _TORCH_AVAILABLE or not os.path.exists(MODEL_PATH):
        return None
    try:
        m = _build_model()
        m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        m.eval()
        _model = m
        print(f"Loaded hyperpigmentation model from {MODEL_PATH}")
        return _model
    except Exception as e:
        print(f"Failed to load hyperpigmentation model: {e}")
        return None


def classify_hyperpigmentation(pil_image: Image.Image) -> tuple[bool, float]:
    """
    Returns (has_hyperpigmentation, confidence 0-1).
    Returns (False, 0.0) if model is not loaded.
    """
    model = _load_model()
    if model is None:
        return False, 0.0

    img = pil_image.convert('RGB')
    tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        score = float(torch.sigmoid(model(tensor).squeeze()))
    return score > 0.5, score
