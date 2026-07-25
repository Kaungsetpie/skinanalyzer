import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

# -----------------------------
# Face Landmarker (468/478 landmarks, same topology as the legacy
# mp.solutions.face_mesh, which isn't available in this mediapipe build)
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_landmarker.task')

_base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
_options = vision.FaceLandmarkerOptions(
    base_options=_base_options,
    num_faces=1,
)
face_landmarker = vision.FaceLandmarker.create_from_options(_options)

# -----------------------------
# ROI Landmark Index
# -----------------------------
# Whole face (forehead, cheeks, nose, jawline, chin) minus eyes and lips.
# Ears and hair are already outside the face-oval boundary, so no separate
# exclusion is needed for them. Cutting the outer-lip polygon removes the
# mouth cavity (and any visible teeth) along with the lips themselves.

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
    0, 37, 39, 40, 185
]

EXCLUDED_REGIONS = [LEFT_EYE, RIGHT_EYE, LIPS]

# T-zone (forehead) vs cheek sub-regions, used only for the combination-skin
# oiliness heuristic below — not part of the main face crop.
FOREHEAD = [10, 67, 103, 109, 151, 337, 299, 296, 284]

LEFT_CHEEK = [50, 101, 118, 117, 111, 35, 31, 228, 229, 230, 231, 232, 233, 244, 189]

RIGHT_CHEEK = [280, 330, 347, 346, 340, 265, 261, 448, 449, 450, 451, 452, 453, 464, 413]


# -----------------------------
# Polygon helpers
# -----------------------------

def _landmarks_to_points(landmarks, indexes, w, h):
    return np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indexes],
        np.int32
    )


def _crop_polygon(image, landmarks, indexes):
    h, w, _ = image.shape

    pts = _landmarks_to_points(landmarks, indexes, w, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    roi = cv2.bitwise_and(image, image, mask=mask)

    x, y, w_box, h_box = cv2.boundingRect(pts)
    roi = roi[y:y + h_box, x:x + w_box]

    return roi


def crop_face_region(image, landmarks):
    h, w, _ = image.shape

    face_pts = _landmarks_to_points(landmarks, FACE_OVAL, w, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [face_pts], 255)

    for region in EXCLUDED_REGIONS:
        pts = _landmarks_to_points(landmarks, region, w, h)
        cv2.fillPoly(mask, [pts], 0)

    roi = cv2.bitwise_and(image, image, mask=mask)

    x, y, w_box, h_box = cv2.boundingRect(face_pts)
    roi = roi[y:y + h_box, x:x + w_box]

    return roi


def _detect_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


def _shine_ratio(region_bgr):
    """Fraction of pixels that look like specular oil highlights (bright, desaturated)."""
    total = region_bgr.shape[0] * region_bgr.shape[1]
    if total == 0:
        return 0.0

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    shine_mask = (v > 200) & (s < 60)

    return float(np.count_nonzero(shine_mask)) / total


# -----------------------------
# Combination-skin heuristic (T-zone vs cheeks)
# -----------------------------

def detect_combination_skin(img, threshold=0.05):
    """
    Compares oiliness (specular shine) between the T-zone (forehead) and the
    cheeks. Returns (is_combination, {"t_zone_shine": .., "cheek_shine": ..}).
    """
    landmarks = _detect_landmarks(img)
    if landmarks is None:
        return False, {}

    forehead = _crop_polygon(img, landmarks, FOREHEAD)
    left_cheek = _crop_polygon(img, landmarks, LEFT_CHEEK)
    right_cheek = _crop_polygon(img, landmarks, RIGHT_CHEEK)

    t_zone_shine = _shine_ratio(forehead)
    cheek_shine = (_shine_ratio(left_cheek) + _shine_ratio(right_cheek)) / 2

    is_combination = (t_zone_shine - cheek_shine) > threshold

    return is_combination, {"t_zone_shine": t_zone_shine, "cheek_shine": cheek_shine}


# -----------------------------
# Main Preprocess
# -----------------------------

def preprocess_image(img):
    landmarks = _detect_landmarks(img)

    if landmarks is None:
        return None

    roi = crop_face_region(img, landmarks)

    roi = cv2.resize(roi, (224, 224))

    return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
