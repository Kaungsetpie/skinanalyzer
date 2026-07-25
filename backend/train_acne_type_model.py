"""
Train a MobileNetV2 3-class acne type classifier on the ACNE04 dataset.

Grade mapping:
  Grade 0 (clear)        → no_acne          (label 0)
  Grade 1 (mild/comedonal) → comedonal_acne  (label 1)  blackheads / whiteheads
  Grade 2 (moderate)     → inflammatory_acne (label 2)  papules / pustules
  Grade 3 (severe)       → inflammatory_acne (label 2)  nodules / cysts

--- Dataset ---
Same ACNE04 dataset used by train_severity_model.py:
  data/ACNE04/  with Grade_0/ Grade_1/ Grade_2/ Grade_3/ subfolders

--- Usage ---
  python train_acne_type_model.py --data-dir data/ACNE04
  python train_acne_type_model.py predict path/to/face.jpg

Model saves to models/acne_type_model.pt and is auto-loaded by
services/acne_type_classifier.py on the next server restart.
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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ['no_acne', 'comedonal_acne', 'inflammatory_acne']
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS_HEAD = 15
EPOCHS_FINE = 25
MODEL_SAVE_PATH = os.path.join('models', 'acne_type_model.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_GRADE_FOLDER_NAMES = {
    0: ['Grade_0', 'grade0', 'grade_0', '0', 'clear',    'level0'],
    1: ['Grade_1', 'grade1', 'grade_1', '1', 'mild',     'level1'],
    2: ['Grade_2', 'grade2', 'grade_2', '2', 'moderate', 'level2'],
    3: ['Grade_3', 'grade3', 'grade_3', '3', 'severe',   'level3'],
}
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Grade → CLASSES index
_GRADE_TO_LABEL = {0: 0, 1: 1, 2: 2, 3: 2}


def _find_folder(root: Path, candidates: list) -> Path | None:
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    return None


def _images_in(folder: Path) -> list:
    return [p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS]


def load_dataframe(data_dir: str):
    import pandas as pd
    data_dir = Path(data_dir)
    rows = []
    for grade, candidates in _GRADE_FOLDER_NAMES.items():
        folder = _find_folder(data_dir, candidates)
        if folder is None:
            continue
        label = _GRADE_TO_LABEL[grade]
        for img_path in _images_in(folder):
            rows.append({'path': str(img_path), 'label': label})

    if not rows:
        raise FileNotFoundError(
            f"No ACNE04 images found in {data_dir}.\n"
            "Expected Grade_0/ Grade_1/ Grade_2/ Grade_3/ subfolders.\n"
            "Download: kaggle datasets download -d rutvikdeshpande/acne04-dataset"
        )

    import pandas as pd
    df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"\nACNE04 acne-type dataset from {data_dir}  (device: {DEVICE})")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:22s}: {(df['label'] == i).sum()}")
    print(f"  Total: {len(df)}")
    return df


class AcneTypeDataset(Dataset):
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
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
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
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(CLASSES)),
    )
    return model.to(DEVICE)


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
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return avg_loss, acc, all_preds, all_labels


def train(data_dir: str) -> None:
    os.makedirs('models', exist_ok=True)

    df = load_dataframe(data_dir)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    cw = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=train_df['label'].values)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: { {CLASSES[i]: f'{cw[i]:.3f}' for i in range(len(CLASSES))} }")

    train_ds = AcneTypeDataset(train_df, train_transform)
    val_ds   = AcneTypeDataset(val_df,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model()
    best_acc = 0.0
    patience_limit = 6

    def save_if_best(acc):
        nonlocal best_acc
        if acc > best_acc:
            best_acc = acc
            torch.save({'classes': CLASSES, 'state_dict': model.state_dict()}, MODEL_SAVE_PATH)
            print(f"  *** Saved (val acc {acc:.4f}) → {MODEL_SAVE_PATH}")

    print(f"\n{'='*60}")
    print("Phase 1: Training head (features frozen)")
    print('='*60)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)
    patience_counter = 0

    for epoch in range(1, EPOCHS_HEAD + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = run_epoch(model, val_loader,   criterion)
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

    print(f"\n{'='*60}")
    print("Final evaluation")
    print('='*60)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    _, _, vl_preds, vl_labels = run_epoch(model, val_loader, criterion)
    print(f"\nBest val acc: {best_acc:.4f}")
    print(classification_report(vl_labels, vl_preds, target_names=CLASSES))
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("Restart the backend server to load the new model.")


def predict(image_path: str) -> None:
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH} — train first.")
        return
    ckpt = torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=False)
    classes = ckpt['classes']
    model = build_model()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    img = Image.open(image_path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()
    print(f"\n{image_path}")
    for cls, prob in zip(classes, probs):
        bar = '#' * int(prob * 30)
        print(f"  {cls:22s}: {prob:.3f}  {bar}")
    print(f"\n  Prediction: {classes[int(np.argmax(probs))].upper()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 acne-type classifier on ACNE04',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='cmd')
    train_p = sub.add_parser('train')
    train_p.add_argument('--data-dir', required=True)
    pred_p  = sub.add_parser('predict')
    pred_p.add_argument('image')
    parser.add_argument('--data-dir')

    args = parser.parse_args()
    if args.cmd == 'predict':
        predict(args.image)
    elif args.cmd == 'train':
        train(args.data_dir)
    elif getattr(args, 'data_dir', None):
        train(args.data_dir)
    else:
        parser.print_help()
