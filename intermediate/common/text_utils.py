"""
Shared text preprocessing utilities for RNN and LSTM models.
Builds a simple vocabulary and converts text to padded integer sequences.
No external tokenizer dependency — keeps the stack lean and avoids version drift.
"""

import re
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def simple_tokenize(text):
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class Vocabulary:
    """Maps tokens to integer indices, built from a training corpus."""

    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.token2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2token = {0: "<pad>", 1: "<unk>"}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))

        most_common = [
            tok for tok, freq in counter.most_common(self.max_vocab_size)
            if freq >= self.min_freq
        ]

        for tok in most_common:
            idx = len(self.token2idx)
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

    def encode(self, text, max_len=100):
        tokens = simple_tokenize(text)[:max_len]
        ids = [self.token2idx.get(tok, 1) for tok in tokens]  # 1 = <unk>
        return ids

    def __len__(self):
        return len(self.token2idx)


class TextClassificationDataset(Dataset):
    """Wraps encoded text sequences + labels for DataLoader use."""

    def __init__(self, texts, labels, vocab, max_len=100):
        self.vocab = vocab
        self.max_len = max_len
        self.encoded = [torch.tensor(vocab.encode(t, max_len), dtype=torch.long)
                         for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]


def collate_batch(batch):
    """Pads variable-length sequences in a batch to the same length."""
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, labels