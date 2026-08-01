"""
Transfer Learning: pretrained ResNet18 (ImageNet weights) fine-tuned on CIFAR-10 subset.
Only the final block + classifier head are unfrozen — standard practice for
CPU-feasible fine-tuning, and demonstrates the core transfer learning pattern
(freeze generic features, retrain task-specific layers).
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TransferLearningModel(nn.Module):
    """
    ResNet18 backbone with ImageNet pretrained weights.
    Freezes all layers except layer4 (final conv block) and the classifier head.

    Args:
        num_classes (int): number of output classes for the new task
        freeze_until (str): name of the last layer to freeze; everything after
            this point (layer4 + fc) is trainable. Default freezes through layer3.
    """

    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)

        if freeze_backbone:
            # Freeze everything first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze the final residual block (layer4) — lets the model
            # adapt higher-level features to the new task without retraining
            # low-level filters (edges/textures), which transfer well as-is
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # Replace the final classifier head — always trainable, task-specific
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def get_trainable_params(self):
        """Returns count of trainable vs total params — useful for results.md."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total