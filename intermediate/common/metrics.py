"""
Shared evaluation metrics for all 5 intermediate-level models.
Wraps sklearn metrics so each model's train.py stays consistent and short.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def compute_batch_accuracy(outputs, labels):
    """
    Computes accuracy for a single batch.

    Args:
        outputs (torch.Tensor): raw model outputs (logits), shape (batch, n_classes)
        labels (torch.Tensor): ground truth labels, shape (batch,)

    Returns:
        float: accuracy for this batch
    """
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def evaluate_model(model, dataloader, device, class_names=None):
    """
    Runs full evaluation over a dataloader and returns metrics + predictions.

    Args:
        model (nn.Module): trained model, already in eval mode or will be set here
        dataloader (DataLoader): test/val dataloader
        device (torch.device): cpu or cuda
        class_names (list[str], optional): for classification_report labeling

    Returns:
        dict with keys: accuracy, precision, recall, f1, confusion_matrix, report, y_true, y_pred
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_evaluation_summary(results, model_name="Model"):
    """
    Pretty-prints an evaluation results dict from evaluate_model().
    """
    print(f"\n{'='*50}")
    print(f"{model_name} — Test Set Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"\nClassification Report:\n{results['report']}")