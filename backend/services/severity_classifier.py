import os
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("PyTorch not available — severity classifier will use heuristic fallback.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'severity_model.pt')
_model = None

_transform = None
if _TORCH_AVAILABLE:
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_mobilenet_model() -> "nn.Module":
    """MobileNetV2 binary severity classifier (output: logit, not sigmoid)."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
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
        m = build_mobilenet_model()
        m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        m.eval()
        _model = m
        print(f"Loaded MobileNetV2 severity model from {MODEL_PATH}")
        return _model
    except Exception as e:
        print(f"Failed to load severity model: {e}")
        return None


def _mobilenet_classify(pil_image: Image.Image, model) -> tuple[bool, float]:
    img = pil_image.convert('RGB')
    tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        score = float(torch.sigmoid(logit))
    return score > 0.5, score


def _heuristic_classify(pil_image: Image.Image) -> tuple[bool, float]:
    """Fallback when no trained model is present — OpenCV color analysis."""
    img_array = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Highly inflamed / red areas (severe inflammatory acne, rosacea)
    red_mask = cv2.bitwise_or(
        cv2.inRange(img_hsv, np.array([0, 120, 80]),   np.array([8, 255, 255])),
        cv2.inRange(img_hsv, np.array([172, 120, 80]), np.array([180, 255, 255])),
    )
    # Very dark spots (suspicious lesions)
    dark_mask = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

    total = img_array.shape[0] * img_array.shape[1]
    score = min(1.0,
                np.count_nonzero(red_mask) / total * 4.0 +
                np.count_nonzero(dark_mask) / total * 1.5)
    return score > 0.25, score


def classify_severity(pil_image: Image.Image) -> tuple[bool, float]:
    """
    Returns (is_severe, confidence 0-1).
    Uses models/severity_model.pt (MobileNetV2) if available, else OpenCV heuristic.
    """
    model = _load_model()
    if model is not None:
        return _mobilenet_classify(pil_image, model)
    return _heuristic_classify(pil_image)
