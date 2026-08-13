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


# -----------------------------
# Polygon helpers
# -----------------------------

def _landmarks_to_points(landmarks, indexes, w, h):
    return np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indexes],
        np.int32
    )

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
