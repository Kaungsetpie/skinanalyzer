from PIL import Image

from services.severity_classifier import classify_severity
from services.acne_type_classifier import classify_acne_type

IMAGE_PATH = "45.jpg"

img = Image.open(IMAGE_PATH)

print("\n" + "=" * 60)
print("MODEL TEST")
print("=" * 60)

# -------------------------
# Severity
# -------------------------
is_severe, severity_score = classify_severity(img)

print("\nSEVERITY")
print(f"Prediction : {'SEVERE' if is_severe else 'NOT SEVERE'}")
print(f"Score      : {severity_score:.3f}")

# -------------------------
# Acne Type
# -------------------------
acne_type, acne_scores = classify_acne_type(img)

print("\nACNE TYPE")
print(f"Prediction : {acne_type}")
print(f"Scores     : {acne_scores}")

print("\n" + "=" * 60)