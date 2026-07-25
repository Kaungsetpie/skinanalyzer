"""
Train a MobileNetV2 3-class skin-type classifier (oily / normal / dry).

--- Dataset download ---
  kaggle datasets download -d shakyadivanshu/oily-and-dry-skin-dataset
  # or search Kaggle for "oily-and-dry-skin-dataset" and note the owner slug

  Unzip so the structure looks like:
    data/skin_type/
      oily/      (or Oily/)
      dry/       (or Dry/)
      normal/    (or Normal/)  ← present in most versions of this dataset

--- Usage ---
  python train_skin_type_model.py --data-dir data/skin_type
  python train_skin_type_model.py predict path/to/face.jpg

Model saves to models/skin_type_model.pt and is auto-loaded by
services/skin_type_classifier.py on the next server restart.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ['oily', 'normal', 'dry']          # label indices 0, 1, 2
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 25
MODEL_SAVE_PATH = os.path.join('models', 'skin_type_model.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Possible folder names per class (case-insensitive checked via candidates list)
_CLASS_FOLDER_NAMES = {
    'oily':   ['oily', 'Oily', 'OILY', 'oily_skin'],
    'normal': ['normal', 'Normal', 'NORMAL', 'normal_skin', 'combination', 'Combination'],
    'dry':    ['dry', 'Dry', 'DRY', 'dry_skin'],
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _find_folder(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    return None


def _images_in(folder: Path) -> list[Path]:
    imgs = [p for p in folder.rglob('*') if p.suffix.lower() in _IMG_EXTS]
    return imgs


def _find_all_class_folders(root: Path, candidates: list[str]) -> list[Path]:
    """Return every folder under root (at any depth) whose name matches a candidate."""
    matches = []
    for p in root.rglob('*'):
        if p.is_dir() and p.name in candidates:
            matches.append(p)
    return matches


def load_dataframe(data_dir: str):
    import pandas as pd
    global CLASSES
    data_dir = Path(data_dir)
    rows = []
    found_classes = []

    for class_name, candidates in _CLASS_FOLDER_NAMES.items():
        folders = _find_all_class_folders(data_dir, candidates)
        if not folders:
            print(f"  [warn] No folder found for class '{class_name}' in {data_dir}")
            continue
        found_classes.append(class_name)
        for folder in folders:
            for img_path in _images_in(folder):
                rows.append({'path': str(img_path), 'class': class_name})

    if not rows:
        raise FileNotFoundError(
            f"No images found in {data_dir}.\n"
            "Expected subfolders named oily/, normal/, dry/ (case-insensitive) at any depth.\n"
            "Download with: kaggle datasets download -d shakyadivanshu/oily-and-dry-skin-dataset"
        )

    if len(found_classes) < 2:
        raise ValueError("Need at least 2 classes (oily, dry, normal) to train.")

    CLASSES = found_classes

    df = pd.DataFrame(rows).reset_index(drop=True)
    df['label'] = df['class'].apply(lambda c: CLASSES.index(c))

    print(f"\nSkin-type dataset loaded from {data_dir}  (device: {DEVICE})")
    for cls in found_classes:
        n = (df['class'] == cls).sum()
        print(f"  {cls:8s}: {n} images")
    print(f"  Total  : {len(df)} images  |  Classes found: {found_classes}")

    return df[['path', 'label']]


# ---------------------------------------------------------------------------
# Dataset & transforms
# ---------------------------------------------------------------------------

class SkinTypeDataset(Dataset):
    def __init__(self, df, transform):
        self.paths  = df['path'].values
        self.labels = df['label'].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), torch.tensor(self.labels[idx])


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return avg_loss, acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(data_dir: str) -> None:
    import pandas as pd
    from sklearn.utils.class_weight import compute_class_weight

    os.makedirs('models', exist_ok=True)

    df = load_dataframe(data_dir)
    num_classes = len(CLASSES)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_df['label'].values)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: { {CLASSES[i]: f'{cw[i]:.3f}' for i in range(num_classes)} }")

    train_ds = SkinTypeDataset(train_df, train_transform)
    val_ds   = SkinTypeDataset(val_df,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(num_classes)
    best_acc = 0.0
    patience_limit = 6

    def save_if_best(acc):
        nonlocal best_acc
        if acc > best_acc:
            best_acc = acc
            checkpoint = {'classes': CLASSES, 'state_dict': model.state_dict()}
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f"  *** Saved (val acc {acc:.4f}) → {MODEL_SAVE_PATH}")

    # ---- Phase 1: head only ------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 1: Training head (MobileNetV2 features frozen)")
    print('='*60)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)
    patience_counter = 0

    for epoch in range(1, EPOCHS_HEAD + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_preds, vl_labels = run_epoch(model, val_loader, criterion)
        scheduler.step(vl_acc)
        improved = vl_acc > best_acc
        save_if_best(vl_acc)
        patience_counter = 0 if improved else patience_counter + 1
        print(f"  Ep {epoch:2d}/{EPOCHS_HEAD} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.3f}")
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
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_preds, vl_labels = run_epoch(model, val_loader, criterion)
        scheduler.step(vl_acc)
        improved = vl_acc > best_acc
        save_if_best(vl_acc)
        patience_counter = 0 if improved else patience_counter + 1
        print(f"  Ep {epoch:2d}/{EPOCHS_FINE} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.3f}")
        if patience_counter >= patience_limit:
            print("  Early stopping.")
            break

    # ---- Final report ------------------------------------------------------
    print(f"\n{'='*60}")
    print("Final evaluation on validation set")
    print('='*60)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    _, vl_acc, vl_preds, vl_labels = run_epoch(model, val_loader, criterion)
    print(f"\nBest val acc: {best_acc:.4f}")
    print(classification_report(vl_labels, vl_preds, target_names=CLASSES))
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("Restart the backend server to load the new model.")


# ---------------------------------------------------------------------------
# Quick single-image prediction
# ---------------------------------------------------------------------------

def predict(image_path: str) -> None:
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH} — train first.")
        return

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=False)
    classes = checkpoint['classes']
    model = build_model(len(classes))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    img = Image.open(image_path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    print(f"\n{image_path}")
    for cls, prob in zip(classes, probs):
        bar = '#' * int(prob * 30)
        print(f"  {cls:8s}: {prob:.3f}  {bar}")
    pred_class = classes[int(np.argmax(probs))]
    print(f"\n  Prediction: {pred_class.upper()}")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 skin-type classifier (oily/normal/dry)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='cmd')

    train_p = sub.add_parser('train', help='Train the model')
    train_p.add_argument('--data-dir', required=True, help='Path to skin_type directory')

    pred_p = sub.add_parser('predict', help='Predict skin type for a single image')
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
