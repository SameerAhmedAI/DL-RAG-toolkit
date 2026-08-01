"""
Training script for the ANN model on the Breast Cancer Wisconsin dataset.
Run from repo root: python -m intermediate.ann.train
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root on path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from intermediate.ann.model import ANNModel
from intermediate.common.metrics import (
    compute_batch_accuracy, evaluate_model, print_evaluation_summary
)
from intermediate.common.plotting import plot_training_curves, plot_confusion_matrix

# ---- Config ----
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
RESULTS_DIR = Path(__file__).parent / "results_artifacts"

torch.manual_seed(42)


def load_data():
    """Loads and splits the Breast Cancer dataset into train/val/test tensors."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    class_names = data.target_names.tolist()

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def to_tensor_dataset(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

    train_ds = to_tensor_dataset(X_train, y_train)
    val_ds = to_tensor_dataset(X_val, y_val)
    test_ds = to_tensor_dataset(X_test, y_test)

    return train_ds, val_ds, test_ds, X.shape[1], class_names


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

    train_ds, val_ds, test_ds, input_dim, class_names = load_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = ANNModel(input_dim=input_dim, hidden_dims=[64, 32],
                      num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Training ANN on {len(train_ds)} samples, validating on {len(val_ds)}...")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = validate(model, val_loader, criterion)

        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        train_accs.append(tr_acc)
        val_accs.append(v_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), RESULTS_DIR / "ann_model.pt")

    # Plot curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        title="ANN — Breast Cancer Classification",
        save_path=RESULTS_DIR / "training_curves.png"
    )

    # Final test evaluation
    results = evaluate_model(model, test_loader, DEVICE, class_names=class_names)
    print_evaluation_summary(results, model_name="ANN")

    plot_confusion_matrix(
        results["confusion_matrix"], class_names,
        title="ANN — Confusion Matrix",
        save_path=RESULTS_DIR / "confusion_matrix.png"
    )

    return results


if __name__ == "__main__":
    main()