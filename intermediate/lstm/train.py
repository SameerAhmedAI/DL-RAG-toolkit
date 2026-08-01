"""
Training script for the LSTM on the same 20 Newsgroups subset as the RNN.
Run from repo root: python -m intermediate.lstm.train
Identical setup to RNN train.py except the model class — intentional, for direct comparison.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from intermediate.lstm.model import LSTMModel
from intermediate.common.text_utils import (
    Vocabulary, TextClassificationDataset, collate_batch
)
from intermediate.common.metrics import (
    compute_batch_accuracy, evaluate_model, print_evaluation_summary
)
from intermediate.common.plotting import plot_training_curves, plot_confusion_matrix

# ---- Config (identical to RNN for fair comparison) ----
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
MAX_LEN = 100
MAX_VOCAB = 10000
CATEGORIES = ['rec.sport.baseball', 'sci.space', 'comp.graphics', 'talk.politics.guns']
RESULTS_DIR = Path(__file__).parent / "results_artifacts"

torch.manual_seed(42)


def load_data():
    data = fetch_20newsgroups(
        subset='all', categories=CATEGORIES, remove=('headers', 'footers', 'quotes')
    )
    texts, labels = data.data, data.target
    class_names = data.target_names

    filtered = [(t, l) for t, l in zip(texts, labels) if len(t.strip()) > 20]
    texts, labels = zip(*filtered)

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    vocab = Vocabulary(max_vocab_size=MAX_VOCAB, min_freq=2)
    vocab.build(X_train)

    train_ds = TextClassificationDataset(X_train, y_train, vocab, MAX_LEN)
    val_ds = TextClassificationDataset(X_val, y_val, vocab, MAX_LEN)
    test_ds = TextClassificationDataset(X_test, y_test, vocab, MAX_LEN)

    return train_ds, val_ds, test_ds, vocab, class_names


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

    train_ds, val_ds, test_ds, vocab, class_names = load_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    print(f"Vocab size: {len(vocab)}")

    model = LSTMModel(vocab_size=len(vocab), num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Training LSTM on {len(train_ds)} samples, validating on {len(val_ds)}...")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = validate(model, val_loader, criterion)

        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        train_accs.append(tr_acc)
        val_accs.append(v_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

    torch.save(model.state_dict(), RESULTS_DIR / "lstm_model.pt")

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        title="LSTM — 20 Newsgroups Classification",
        save_path=RESULTS_DIR / "training_curves.png"
    )

    results = evaluate_model(model, test_loader, DEVICE, class_names=class_names)
    print_evaluation_summary(results, model_name="LSTM")

    plot_confusion_matrix(
        results["confusion_matrix"], class_names,
        title="LSTM — Confusion Matrix",
        save_path=RESULTS_DIR / "confusion_matrix.png"
    )

    return results


if __name__ == "__main__":
    main()