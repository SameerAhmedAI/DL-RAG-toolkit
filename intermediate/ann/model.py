"""
Artificial Neural Network (ANN) for tabular binary/multiclass classification.
Dataset: sklearn's Breast Cancer Wisconsin dataset (small, tabular, CPU-friendly).
"""

import torch
import torch.nn as nn


class ANNModel(nn.Module):
    """
    Simple feedforward ANN with 2 hidden layers, dropout, and batch norm.

    Args:
        input_dim (int): number of input features
        hidden_dims (list[int]): sizes of hidden layers, e.g. [64, 32]
        num_classes (int): number of output classes
        dropout (float): dropout probability
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=2, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)