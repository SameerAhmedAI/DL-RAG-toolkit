"""
LSTM for text classification.
Same task/dataset as the RNN model — architecture is the only variable,
enabling a direct, meaningful comparison in results.md.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Embedding -> LSTM -> final hidden state -> classifier.

    Args:
        vocab_size (int): size of the vocabulary
        embed_dim (int): embedding dimension
        hidden_dim (int): LSTM hidden state size
        num_classes (int): number of output classes
        num_layers (int): number of stacked LSTM layers
        dropout (float): dropout before the classifier
    """

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_classes=4, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)                  # (batch, seq_len, embed_dim)
        _, (hidden, cell) = self.lstm(embedded)         # hidden: (num_layers, batch, hidden_dim)
        last_hidden = hidden[-1]                         # (batch, hidden_dim)
        out = self.dropout(last_hidden)
        return self.classifier(out)