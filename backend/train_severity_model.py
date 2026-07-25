"""
Train a MobileNetV2 binary severity classifier on the ACNE04 dataset.

--- Dataset download ---
Request the dataset from the authors (ICCV 2019 paper):
  "Joint Acne Image Grading and Counting via Label Distribution Learning"
  https://github.com/xpwu95/LDL

Or download from Kaggle:
  kaggle datasets download -d rutvikdeshpande/acne04-dataset

Extract so the structure looks like one of these:
  data/ACNE04/
    Grade_0/   Grade_1/   Grade_2/   Grade_3/    ← folder-per-grade layout

  OR

  data/ACNE04/
    JPEGImages/   ← all images here
    Grade_0.txt   Grade_1.txt   Grade_2.txt   Grade_3.txt  ← filenames per grade

--- Severity mapping ---
  Grade 0 (clear)    → not severe (0)
  Grade 1 (mild)     → not severe (0)
  Grade 2 (moderate) → severe    (1)  ← recommend dermatologist
  Grade 3 (severe)   → severe    (1)  ← recommend dermatologist

--- Usage ---
  python train_severity_model.py --data-dir data/ACNE04
  python train_severity_model.py predict path/to/face.jpg

Model saves to models/severity_model.pt and is auto-loaded by
services/severity_classifier.py on the next server restart.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEVERE_GRADES = {2, 3}   # grades that need a dermatologist
IMG_SIZE = 224
BATCH_SIZE = 16          # small — ACNE04 is ~1,457 images
EPOCHS_HEAD = 15
EPOCHS_FINE = 25
MODEL_SAVE_PATH = os.path.join('models', 'severity_model.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Folder name variants used by different ACNE04 releases
_GRADE_FOLDER_NAMES = {
    0: ['Grade_0', 'grade0', 'grade_0', '0', 'clear', 'level0'],
    1: ['Grade_1', 'grade1', 'grade_1', '1', 'mild',  'level1'],
    2: ['Grade_2', 'grade2', 'grade_2', '2', 'moderate', 'level2'],
    3: ['Grade_3', 'grade3', 'grade_3', '3', 'severe', 'level3'],
}
_GRADE_TXT_NAMES = {
    0: ['Grade_0.txt', 'grade0.txt'],
    1: ['Grade_1.txt', 'grade1.txt'],
    2: ['Grade_2.txt', 'grade2.txt'],
    3: ['Grade_3.txt', 'grade3.txt'],
}
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


# ---------------------------------------------------------------------------
# Dataset loading — handles both layouts automatically
# ---------------------------------------------------------------------------

def _find_folder(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    return None


def _images_in(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS]


def _load_folder_layout(data_dir: Path) -> pd.DataFrame | None:
    """Grade_0/ Grade_1/ Grade_2/ Grade_3/ folders inside data_dir."""
    rows = []
    for grade, candidates in _GRADE_FOLDER_NAMES.items():
        folder = _find_folder(data_dir, candidates)
        if folder is None:
            continue
        for img_path in _images_in(folder):
            rows.append({'path': str(img_path), 'grade': grade})
    return pd.DataFrame(rows) if rows else None


def _load_txt_layout(data_dir: Path) -> pd.DataFrame | None:
    """Grade_N.txt files listing image names + JPEGImages/ folder."""
    img_dirs = [data_dir / 'JPEGImages', data_dir / 'images', data_dir]
    rows = []
    for grade, candidates in _GRADE_TXT_NAMES.items():
        txt = next((data_dir / n for n in candidates if (data_dir / n).exists()), None)
        if txt is None:
            continue
        for name in txt.read_text().splitlines():
            name = name.strip()
            if not name:
                continue
            for img_dir in img_dirs:
                p = img_dir / name
                if not p.exists():
                    p = img_dir / (name + '.jpg')
                if p.exists():
                    rows.append({'path': str(p), 'grade': grade})
                    break
    return pd.DataFrame(rows) if rows else None


def load_dataframe(data_dir: str) -> pd.DataFrame:
    data_dir = Path(data_dir)

    df = _load_folder_layout(data_dir)
    if df is None or df.empty:
        df = _load_txt_layout(data_dir)
    if df is None or df.empty:
        raise FileNotFoundError(
            f"Could not find ACNE04 images in {data_dir}.\n"
            "Expected one of:\n"
            "  • Grade_0/ Grade_1/ Grade_2/ Grade_3/ subfolders\n"
            "  • Grade_0.txt … Grade_3.txt files + JPEGImages/ folder"
        )

    df['label'] = df['grade'].apply(lambda g: 1 if g in SEVERE_GRADES else 0)
    df = df.reset_index(drop=True)

    print(f"\nACNE04 dataset loaded from {data_dir}  (device: {DEVICE})")
    for g in sorted(df['grade'].unique()):
        n = (df['grade'] == g).sum()
        tag = 'SEVERE' if g in SEVERE_GRADES else 'not severe'
        print(f"  Grade {g} ({tag}): {n} images")
    print(f"\n  Total severe    : {df['label'].sum()}")
    print(f"  Total not severe: {(df['label'] == 0).sum()}")
    return df[['path', 'label']]


# ---------------------------------------------------------------------------
# Dataset & transforms
# ---------------------------------------------------------------------------

class AcneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths  = df['path'].values
        self.labels = df['label'].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), torch.tensor(self.labels[idx])


# Stronger augmentation — ACNE04 is small (~1.4k images)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace head: 1280 → 128 → 1
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    acc = float(np.mean((np.array(all_probs) > 0.5) == np.array(all_labels)))
    return avg_loss, auc, acc


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(data_dir: str) -> None:
    os.makedirs('models', exist_ok=True)

    df = load_dataframe(data_dir)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    # Weighted loss for class imbalance
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['label'].values)
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight: {pos_weight.item():.3f}")

    train_ds = AcneDataset(train_df, train_transform)
    val_ds   = AcneDataset(val_df,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model()
    best_auc = 0.0

    def save_if_best(auc):
        nonlocal best_auc
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  *** Saved (val AUC {auc:.4f}) → {MODEL_SAVE_PATH}")

    # ---- Phase 1: head only ------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 1: Training head (MobileNetV2 features frozen)")
    print('='*60)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)
    patience_counter, patience_limit = 0, 6

    for epoch in range(1, EPOCHS_HEAD + 1):
        tr_loss, tr_auc, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_auc, vl_acc = run_epoch(model, val_loader,   criterion)
        scheduler.step(vl_auc)
        improved = vl_auc > best_auc
        save_if_best(vl_auc)
        patience_counter = 0 if improved else patience_counter + 1
        print(f"  Ep {epoch:2d}/{EPOCHS_HEAD} | "
              f"train loss {tr_loss:.4f} auc {tr_auc:.4f} acc {tr_acc:.3f} | "
              f"val loss {vl_loss:.4f} auc {vl_auc:.4f} acc {vl_acc:.3f}")
        if patience_counter >= patience_limit:
            print("  Early stopping.")
            break

    # ---- Phase 2: fine-tune last 4 feature blocks --------------------------
    print(f"\n{'='*60}")
    print("Phase 2: Fine-tuning last 4 MobileNetV2 feature blocks")
    print('='*60)
    for block in model.features[-4:]:
        for param in block.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-6, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)
    patience_counter = 0

    for epoch in range(1, EPOCHS_FINE + 1):
        tr_loss, tr_auc, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_auc, vl_acc = run_epoch(model, val_loader,   criterion)
        scheduler.step(vl_auc)
        improved = vl_auc > best_auc
        save_if_best(vl_auc)
        patience_counter = 0 if improved else patience_counter + 1
        print(f"  Ep {epoch:2d}/{EPOCHS_FINE} | "
              f"train loss {tr_loss:.4f} auc {tr_auc:.4f} acc {tr_acc:.3f} | "
              f"val loss {vl_loss:.4f} auc {vl_auc:.4f} acc {vl_acc:.3f}")
        if patience_counter >= patience_limit:
            print("  Early stopping.")
            break

    # ---- Final report ------------------------------------------------------
    print(f"\n{'='*60}")
    print("Final evaluation on validation set")
    print('='*60)
    # Reload best checkpoint
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    _, vl_auc, _ = run_epoch(model, val_loader, criterion)

    # Collect predictions for classification report
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            logits = model(imgs.to(DEVICE)).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.tolist())
    preds = (np.array(all_probs) > 0.5).astype(int)
    print(f"\nBest val AUC: {best_auc:.4f}")
    print(classification_report(all_labels, preds, target_names=['not severe', 'severe']))
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("Restart the backend server to load the new model.")


# ---------------------------------------------------------------------------
# Quick single-image prediction
# ---------------------------------------------------------------------------

def predict(image_path: str) -> None:
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH} — train first.")
        return
    from services.severity_classifier import build_mobilenet_model
    model = build_mobilenet_model()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=True))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        score = float(torch.sigmoid(model(tensor).squeeze()))
    label = 'SEVERE — see a dermatologist' if score > 0.5 else 'Not severe'
    print(f"\n{image_path}")
    print(f"  Result : {label}")
    print(f"  Score  : {score:.3f}")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 skin severity classifier on ACNE04',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='cmd')

    train_p = sub.add_parser('train', help='Train the model')
    train_p.add_argument('--data-dir', required=True, help='Path to ACNE04 directory')

    pred_p = sub.add_parser('predict', help='Predict severity for a single image')
    pred_p.add_argument('image', help='Path to a face image')

    parser.add_argument('--data-dir', help='Shorthand for: train --data-dir')

    args = parser.parse_args()
    if args.cmd == 'predict':
        predict(args.image)
    elif args.cmd == 'train':
        train(args.data_dir)
    elif getattr(args, 'data_dir', None):
        train(args.data_dir)
    else:
        parser.print_help()
