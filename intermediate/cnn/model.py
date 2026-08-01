"""
Convolutional Neural Network (CNN) for image classification.
Dataset: FashionMNIST (small, grayscale, CPU-friendly — CIFAR-10 would be too slow on CPU).
"""

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """
    Small CNN: 2 conv blocks (conv → batchnorm → relu → maxpool) + 2 FC layers.
    Designed to train fast on CPU while still demonstrating standard CNN patterns.

    Args:
        num_classes (int): number of output classes
        in_channels (int): input image channels (1 for grayscale FashionMNIST)
    """

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x