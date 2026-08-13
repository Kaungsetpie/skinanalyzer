import os
import random

folder = r"data/ACNE04/train/no_acne"
target_count = 500

valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

files = [
    f for f in os.listdir(folder)
    if f.lower().endswith(valid_exts)
]

print("Current images:", len(files))

if len(files) <= target_count:
    print("No need to remove images.")
else:
    random.seed(42)   # အမြဲတူတဲ့ random result ရဖို့
    files_to_delete = random.sample(files, len(files) - target_count)

    for f in files_to_delete:
        os.remove(os.path.join(folder, f))

    print(f"Deleted {len(files_to_delete)} images.")

remaining = len([
    f for f in os.listdir(folder)
    if f.lower().endswith(valid_exts)
])

print("Remaining images:", remaining)