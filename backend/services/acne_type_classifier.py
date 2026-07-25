
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'acne_type_model.pt')

_model = None
_classes = None

_transform = None
if _TORCH_AVAILABLE:
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _build_model(num_classes: int) -> "nn.Module":
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def _load_model():
    global _model, _classes
    if _model is not None:
        return _model, _classes
    if not _TORCH_AVAILABLE or not os.path.exists(MODEL_PATH):
        return None, None
    try:
        ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        _classes = ckpt['classes']
        m = _build_model(len(_classes))
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        _model = m
        print(f"Loaded acne-type model from {MODEL_PATH}  classes={_classes}")
        return _model, _classes
    except Exception as e:
        print(f"Failed to load acne-type model: {e}")
        return None, None


def classify_acne_type(pil_image: Image.Image) -> tuple[str | None, dict[str, float]]:
    """
    Returns (predicted_class, {class: confidence}).
    predicted_class is one of: 'no_acne', 'comedonal_acne', 'inflammatory_acne'.
    Returns (None, {}) if model is not loaded.
    """
    model, classes = _load_model()
    if model is None:
        return None, {}

    img = pil_image.convert('RGB')
    tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()

    if isinstance(probs, float):
        probs = [probs]

    scores = {cls: float(p) for cls, p in zip(classes, probs)}
    predicted = classes[int(np.argmax(probs))]
    return predicted, scores
