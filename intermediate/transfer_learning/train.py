"""
Training script for Transfer Learning (ResNet18 -> CIFAR-10 subset).
Run from repo root: python -m intermediate.transfer_learning.train

Uses a small CIFAR-10 subset and only fine-tunes layer4 + fc — CPU-feasible.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms

from intermediate.transfer_learning.model import TransferLearningModel
from intermediate.common.metrics import (
    compute_batch_accuracy, evaluate_model, print_evaluation_summary
)
from intermediate.common.plotting import plot_training_curves, plot_confusion_matrix

# ---- Config ----
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 6
LR = 1e-4  # lower LR — fine-tuning pretrained weights, not training from scratch
TRAIN_SUBSET_SIZE = 4000   # CIFAR-10 full train is 50k — subset for CPU speed
TEST_SUBSET_SIZE = 1000
RESULTS_DIR = Path(__file__).parent / "results_artifacts"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_datasets"

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

torch.manual_seed(42)


def load_data():
    """Downloads/loads CIFAR-10, resizes to 224x224 for ResNet18 input, subsets for CPU speed."""
    # ImageNet normalization stats — required since we're using ImageNet pretrained weights
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    full_test = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    train_subset = Subset(full_train, range(TRAIN_SUBSET_SIZE))
    test_subset = Subset(full_test, range(TEST_SUBSET_SIZE))

    val_size = int(0.15 * len(train_subset))
    train_size = len(train_subset) - val_size
    train_ds, val_ds = random_split(train_subset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    return train_ds, val_ds, test_subset


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_acc += compute_batch_accuracy(outputs, labels) * inputs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_acc / n


def validate(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_acc += compute_batch_accuracy(outputs, labels) * inputs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_acc / n


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-10 (this will download ~170MB on first run)...")
    train_ds, val_ds, test_ds = load_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = TransferLearningModel(num_classes=10, freeze_backbone=True).to(DEVICE)

    trainable, total = model.get_trainable_params()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    criterion = nn.CrossEntropyLoss()
    # Only optimize params that require grad — freezing is enforced here too
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)}...")
    print("Note: CPU + ResNet18 at 224x224 is slow — expect several minutes per epoch.")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = validate(model, val_loader, criterion)

        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        train_accs.append(tr_acc)
        val_accs.append(v_acc)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

    torch.save(model.state_dict(), RESULTS_DIR / "transfer_learning_model.pt")

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        title="Transfer Learning (ResNet18) — CIFAR-10 Classification",
        save_path=RESULTS_DIR / "training_curves.png"
    )

    results = evaluate_model(model, test_loader, DEVICE, class_names=CLASS_NAMES)
    print_evaluation_summary(results, model_name="Transfer Learning (ResNet18)")

    plot_confusion_matrix(
        results["confusion_matrix"], CLASS_NAMES,
        title="Transfer Learning — Confusion Matrix",
        save_path=RESULTS_DIR / "confusion_matrix.png"
    )

    return results


if __name__ == "__main__":
    main()