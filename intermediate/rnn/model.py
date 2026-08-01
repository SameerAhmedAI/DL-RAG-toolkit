"""
RNN (vanilla) for text classification.
Paired deliberately with the LSTM model on the same task/dataset for direct comparison.
"""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    Embedding -> vanilla RNN -> final hidden state -> classifier.

    Args:
        vocab_size (int): size of the vocabulary
        embed_dim (int): embedding dimension
        hidden_dim (int): RNN hidden state size
        num_classes (int): number of output classes
        num_layers (int): number of stacked RNN layers
        dropout (float): dropout before the classifier
    """

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_classes=4, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, nonlinearity='tanh'
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)                  # (batch, seq_len, embed_dim)
        _, hidden = self.rnn(embedded)                 # hidden: (num_layers, batch, hidden_dim)
        last_hidden = hidden[-1]                        # (batch, hidden_dim)
        out = self.dropout(last_hidden)
        return self.classifier(out)