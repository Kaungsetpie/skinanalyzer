import os
import random
import shutil
from pathlib import Path

# မမရဲ့ Hyperpigmentation Folder လမ်းကြောင်း
TARGET_DIR = Path("data/hyperpigmentation")
BACKUP_DIR = Path("data/hyperpigmentation_extra") # ပိုတဲ့ပုံတွေ သွားသိမ်းမယ့် Folder

KEEP_COUNT = 1274
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Folder ထဲရှိ ပုံများ ရှာခြင်း
all_images = [p for p in TARGET_DIR.rglob('*') if p.suffix.lower() in _IMG_EXTS]
print(f"လက်ရှိ Folder ထဲမှာရှိသော ပုံစုစုပေါင်း: {len(all_images)} ပုံ")

if len(all_images) > KEEP_COUNT:
    # Random ၁,၂၇၄ ပုံ ရွေးထုတ်ခြင်း
    random.seed(42)
    images_to_keep = set(random.sample(all_images, KEEP_COUNT))
    images_to_move = [p for p in all_images if p not in images_to_keep]

    # ပိုတဲ့ပုံတွေ သွားသိမ်းမယ့် Folder ဆောက်ခြင်း
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # ပိုနေတဲ့ ပုံများကို extra folder ထဲ သွားရွှေ့ခြင်း
    for img_path in images_to_move:
        shutil.move(str(img_path), str(BACKUP_DIR / img_path.name))

    print(f"✅ အောင်မြင်ပါသည်။ {KEEP_COUNT} ပုံကို '{TARGET_DIR}' ထဲမှာ ချန်ခဲ့ပြီး၊")
    print(f"📦 ပိုနေသော {len(images_to_move)} ပုံကို '{BACKUP_DIR}' ထဲသို့ ရွှေ့လိုက်ပါပြီမမ!")
else:
    print("⚠️ Folder ထဲမှာ ပုံ ၁,၂၇၄ ခုထက် နည်းနေပါသည် သို့မဟုတ် ကွက်တိ ဖြစ်နေပါပြီမမ!")