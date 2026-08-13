import os
import cv2
import numpy as np
import mediapipe as mp

# ရှိပြီးသား preprocessor ထဲက detect function နဲ့ landmark အသီးသီးကို import လုပ်ပါ
# (preprocessor.py နာမည်အတိုင်း import လုပ်ထားပါတယ်)
from services.preprocessor import (
    _detect_landmarks, 
    _landmarks_to_points, 
    crop_face_region, 
    FACE_OVAL, 
    EXCLUDED_REGIONS
)

def export_4_presentation_steps(image_path, output_dir="pre_steps"):
    """
    မင်းရဲ့ တကယ့် preprocessor.py logic ကိုသုံးပြီး
    Slide အတွက် လိုအပ်တဲ့ ပုံ ၄ ပုံကို ထုတ်ယူပေးမယ့် function ဖြစ်ပါတယ်။
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Original Image ကို ဖတ်မယ်
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image မတွေ့ပါဘူး - {image_path}")
        return

    h, w, _ = img.shape
    
    # Step 1: Original Image သိမ်းမယ်
    cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), img)

    # Landmarks တွေကို ရှာမယ်
    landmarks = _detect_landmarks(img)
    if landmarks is None:
        print("Error: မျက်နှာ ရှာမတွေ့ပါဘူး!")
        return

    # Step 2: Mediapipe Landmarks Overlay Image (Dots/Polygons ပြမည့်ပုံ)
    mp_draw_img = img.copy()
    
    # Face Oval Polygon ဆွဲမယ် (အပြာရောင်)
    face_pts = _landmarks_to_points(landmarks, FACE_OVAL, w, h)
    cv2.polylines(mp_draw_img, [face_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # Eyes & Lips Excluded Regions ဆွဲမယ် (အနီရောင်)
    for region in EXCLUDED_REGIONS:
        pts = _landmarks_to_points(landmarks, region, w, h)
        cv2.polylines(mp_draw_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Landmark Points လေးတွေ အစက်ချပေးမယ် (အဝါရောင်)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(mp_draw_img, (cx, cy), 1, (0, 255, 255), -1)

    cv2.imwrite(os.path.join(output_dir, "2_mediapipe_landmarks.jpg"), mp_draw_img)

    # Step 3: Cropped ROI (မျက်လုံး၊ နှုတ်ခမ်း မပါဘဲ မျက်နှာသီးသန့်)
    roi = crop_face_region(img, landmarks)
    cv2.imwrite(os.path.join(output_dir, "3_cropped_roi.jpg"), roi)

    # Step 4: Resized ROI (224x224)
    roi_224 = cv2.resize(roi, (224, 224))
    cv2.imwrite(os.path.join(output_dir, "4_resized_224x224.jpg"), roi_224)

    print(f"✨ အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ! '{output_dir}' folder ထဲမှာ သွားကြည့်လို့ရပါပြီ။")

if __name__ == "__main__":
    # ⚠️ ဒီနေရာမှာ ပိုးအိဆီမှာရှိတဲ့ စမ်းချင်တဲ့ မျက်နှာပုံလမ်းကြောင်း လဲပေးပါ
    TEST_IMAGE_PATH = "data/skin_type/train/oily/sample.jpg"  # မိမိပုံလမ်းကြောင်း ပြင်ပါ
    
    export_4_presentation_steps("data/skin_type/train/normal/test.jpg")