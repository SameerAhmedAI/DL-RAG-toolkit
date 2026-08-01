"""
Shared plotting utilities for training/validation curves and confusion matrices.
Used by all 5 intermediate-level models to keep result reporting consistent.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None,
                          title="Training Curves", save_path=None):
    """
    Plots loss (and optionally accuracy) curves over epochs.

    Args:
        train_losses (list[float]): training loss per epoch
        val_losses (list[float]): validation loss per epoch
        train_accs (list[float], optional): training accuracy per epoch
        val_accs (list[float], optional): validation accuracy per epoch
        title (str): plot title
        save_path (str or Path, optional): if given, saves the figure here
    """
    has_acc = train_accs is not None and val_accs is not None
    n_plots = 2 if has_acc else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss subplot
    axes[0].plot(epochs, train_losses, label="Train Loss", marker='o')
    axes[0].plot(epochs, val_losses, label="Val Loss", marker='o')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy subplot (if provided)
    if has_acc:
        axes[1].plot(epochs, train_accs, label="Train Acc", marker='o')
        axes[1].plot(epochs, val_accs, label="Val Acc", marker='o')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{title} — Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Plots a confusion matrix as a heatmap.

    Args:
        cm (np.ndarray): confusion matrix, shape (n_classes, n_classes)
        class_names (list[str]): class labels for axis ticks
        title (str): plot title
        save_path (str or Path, optional): if given, saves the figure here
    """
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8),
                                     max(5, len(class_names) * 0.7)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()
    plt.close()