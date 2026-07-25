"""
Train a MobileNetV2 binary classifier for hyperpigmentation / melasma detection.

--- Dataset download ---
  kaggle datasets download -d muhammadalirhojab/melasma
  unzip melasma.zip -d data/melasma

  The muhammadalirhojab/melasma dataset is a flat folder of melasma images only
  (all positive examples). You must supply a separate folder of normal skin images
  as the negative class via --normal-dir.

  Use the Normal/ folder from the skin-type dataset as negatives:
    python train_hyperpigmentation_model.py \\
        --data-dir data/melasma \\
        --normal-dir data/skin_type

--- Usage ---
  python train_hyperpigmentation_model.py --data-dir data/melasma --normal-dir data/skin_type
  python train_hyperpigmentation_model.py predict path/to/face.jpg

Model saves to models/hyperpigmentation_model.pt and is auto-loaded by
services/hyperpigmentation_classifier.py on the next server restart.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 25
MODEL_SAVE_PATH = os.path.join('models', 'hyperpigmentation_model.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _images_in(folder: Path) -> list:
    return [p for p in folder.rglob('*') if p.suffix.lower() in _IMG_EXTS]


_MAX_NEG_RATIO = 4   # cap negatives at this multiple of positives


def load_dataframe(data_dir: str, normal_dir: str) -> "pd.DataFrame":
    """
    data_dir   — flat folder of melasma images (all positive, label=1).
    normal_dir — folder tree of normal/non-melasma skin images (label=0).
                 Recursively collects all images inside.
    Negatives are capped at _MAX_NEG_RATIO * n_positives to avoid severe imbalance.
    """
    import pandas as pd
    pos_paths = _images_in(Path(data_dir))
    neg_paths = _images_in(Path(normal_dir))

    if not pos_paths:
        raise FileNotFoundError(f"No images found in --data-dir {data_dir}")
    if not neg_paths:
        raise FileNotFoundError(f"No images found in --normal-dir {normal_dir}")

    # Subsample negatives if needed
    cap = len(pos_paths) * _MAX_NEG_RATIO
    if len(neg_paths) > cap:
        rng = np.random.default_rng(42)
        neg_paths = list(rng.choice(neg_paths, size=cap, replace=False))
        print(f"  [info] Negatives capped at {cap} ({_MAX_NEG_RATIO}× positives)")

    rows = (
        [{'path': str(p), 'label': 1} for p in pos_paths] +
        [{'path': str(p), 'label': 0} for p in neg_paths]
    )
    df = pd.DataFrame(rows).reset_index(drop=True)

    print(f"\nHyperpigmentation dataset  (device: {DEVICE})")
    print(f"  Melasma / positive: {(df['label'] == 1).sum()}  ({data_dir})")
    print(f"  Normal  / negative: {(df['label'] == 0).sum()}  ({normal_dir})")
    print(f"  Total             : {len(df)}")
    return df


class MelasmaDataset(Dataset):
    def __init__(self, df, transform):
        self.paths  = df['path'].values
        self.labels = df['label'].values.astype(np.float32)
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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model() -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )
    return model.to(DEVICE)


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


def train(data_dir: str, normal_dir: str) -> None:
    os.makedirs('models', exist_ok=True)

    df = load_dataframe(data_dir, normal_dir)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['label'].values)
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight: {pos_weight.item():.3f}")

    train_ds = MelasmaDataset(train_df, train_transform)
    val_ds   = MelasmaDataset(val_df,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model()
    best_auc = 0.0
    patience_limit = 6

    def save_if_best(auc):
        nonlocal best_auc
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  *** Saved (val AUC {auc:.4f}) → {MODEL_SAVE_PATH}")

    print(f"\n{'='*60}")
    print("Phase 1: Training head (features frozen)")
    print('='*60)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)
    patience_counter = 0

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

    print(f"\n{'='*60}")
    print("Phase 2: Fine-tuning last 4 feature blocks")
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

    print(f"\n{'='*60}")
    print("Final evaluation")
    print('='*60)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            logits = model(imgs.to(DEVICE)).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.tolist())
    preds = (np.array(all_probs) > 0.5).astype(int)
    print(f"\nBest val AUC: {best_auc:.4f}")
    print(classification_report(all_labels, preds, target_names=['normal', 'melasma']))
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("Restart the backend server to load the new model.")


def predict(image_path: str) -> None:
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH} — train first.")
        return
    model = build_model()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=True))
    model.eval()
    img = Image.open(image_path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        score = float(torch.sigmoid(model(tensor).squeeze()))
    label = 'MELASMA / Hyperpigmentation detected' if score > 0.5 else 'No hyperpigmentation'
    print(f"\n{image_path}")
    print(f"  Result : {label}")
    print(f"  Score  : {score:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 hyperpigmentation/melasma classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='cmd')

    train_p = sub.add_parser('train')
    train_p.add_argument('--data-dir',    required=True,
                         help='Flat folder of melasma images (all positive)')
    train_p.add_argument('--normal-dir',  required=True,
                         help='Folder of normal skin images (negatives). '
                              'E.g. data/skin_type  or  data/skin_type/train/Normal')

    pred_p = sub.add_parser('predict')
    pred_p.add_argument('image')

    args = parser.parse_args()
    if args.cmd == 'predict':
        predict(args.image)
    elif args.cmd == 'train':
        train(args.data_dir, args.normal_dir)
    else:
        parser.print_help()
