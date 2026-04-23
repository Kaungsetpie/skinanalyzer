import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


def preprocess_image(img):
    print(f"Preprocessing image...{img}")
    # 1. Create a FaceDetector object
    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # 2. Load input image
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detectionResult = detector.detect(image)
    print(f"Detection result: {detectionResult}")
    if detectionResult.detections:
        best_detection = max(detectionResult.detections, key=lambda d: d.categories[0].score)
        score = best_detection.categories[0].score
        bbox = best_detection.bounding_box
        print(f"Best detection score: {score}, bounding box: {bbox}")
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        face_cropped = img[y:y+h, x:x+w]
        return Image.fromarray(face_cropped)
    else:
        print("No face detected.")
        return None

