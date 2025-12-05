"""Minimal ViT training script for the 1:4 dementia dataset copy.

This intentionally keeps the pipeline simple:
- No k-fold training, SMOTE, SWA, or complex schedulers
- Basic Resize → ToTensor → Normalize transforms (no heavy augmentation)
- Single train/validation split followed by a one-pass evaluation on the test set

Run with: `python vit_minimal.py`
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from timm import create_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent


class Config:
    # Data
    train_csv = SCRIPT_DIR / "train_ratio_1in4.csv"
    test_csv = SCRIPT_DIR / "test_ratio_1in4.csv"
    train_dir = SCRIPT_DIR / "dataset_ratio_1in4" / "train"
    test_dir = SCRIPT_DIR / "dataset_ratio_1in4" / "test"
    val_split = 0.2

    # Model
    model_name = "vit_tiny_patch16_224"
    pretrained = True
    num_classes = 1

    # Training
    batch_size = 32
    num_epochs = 20
    learning_rate = 3e-4
    weight_decay = 1e-2
    num_workers = 2
    img_size = 224
    seed = 42

    # Output
    model_dir = SCRIPT_DIR / "models" / "minimal_vit"
    best_model_path = model_dir / "vit_tiny_minimal_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class SimpleDataset(Dataset):
    def __init__(self, csv_path: Path, img_dir: Path, indices: List[int] | None = None) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {img_dir}")
        self.data = pd.read_csv(csv_path)
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = build_transforms()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img_path = self.img_dir / row['filename']
        image = Image.open(img_path).convert('RGB')
        tensor = self.transform(image)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return tensor, label.unsqueeze(0)


def create_dataloaders() -> Tuple[DataLoader, DataLoader]:
    df = pd.read_csv(Config.train_csv)
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=Config.val_split,
        random_state=Config.seed,
        stratify=df['label'],
    )
    train_dataset = SimpleDataset(Config.train_csv, Config.train_dir, train_idx.tolist())
    val_dataset = SimpleDataset(Config.train_csv, Config.train_dir, val_idx.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def create_test_loader() -> DataLoader:
    test_dataset = SimpleDataset(Config.test_csv, Config.test_dir)
    return DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )


def train_one_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(Config.device)
        labels = labels.to(Config.device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(Config.device)
            labels = labels.to(Config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().astype(int).flatten().tolist())
            all_preds.extend(preds.cpu().numpy().astype(int).flatten().tolist())

    avg_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    set_seed(Config.seed)
    os.makedirs(Config.model_dir, exist_ok=True)

    train_loader, val_loader = create_dataloaders()
    test_loader = create_test_loader()

    model = create_model(Config.model_name, pretrained=Config.pretrained, num_classes=Config.num_classes)
    model = model.to(Config.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, Config.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d}/{Config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = model.state_dict()
            torch.save(best_state, Config.best_model_path)
            print(f"  Saved new best model to {Config.best_model_path}")

    if best_state is None:
        print("Training finished without a valid checkpoint; skipping test evaluation.")
        return

    print("\nEvaluating best checkpoint on the test set...")
    model.load_state_dict(best_state)
    model.eval()
    test_metrics = evaluate(model, test_loader, criterion)
    for key, value in test_metrics.items():
        print(f"Test {key.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()
