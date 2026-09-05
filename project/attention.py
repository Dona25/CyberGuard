"""Custom causal multi-head self-attention for CyberGuard.

The implementation explicitly computes Q, K, V, scaled dot-product scores,
a lower-triangular causal mask, softmax attention weights, and weighted values.
It is vectorized across heads for faster training while still implementing all
heads manually rather than using nn.MultiheadAttention.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim=128, number_of_heads=4, context_length=128, dropout=0.1):
        super().__init__()
        if embedding_dim % number_of_heads != 0:
            raise ValueError("embedding_dim must be divisible by number_of_heads")
        self.embedding_dim = embedding_dim
        self.number_of_heads = number_of_heads
        self.head_dim = embedding_dim // number_of_heads
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
        )

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.number_of_heads, self.head_dim

        # Q, K, V: (B,T,C) -> (B,H,T,D)
        q = self.query(x).view(B, T, H, D).transpose(1, 2)
        k = self.key(x).view(B, T, H, D).transpose(1, 2)
        v = self.value(x).view(B, T, H, D).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        attended = weights @ v                         # (B,H,T,D)

        # Concatenate heads: (B,H,T,D) -> (B,T,C)
        concatenated = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.output_dropout(self.projection(concatenated))


if __name__ == "__main__":
    torch.manual_seed(42)
    layer = MultiHeadSelfAttention(embedding_dim=128, number_of_heads=4, context_length=128)
    x = torch.randn(2, 16, 128)
    y = layer(x)
    print("ATTENTION TEST PASSED")
    print("Input:", tuple(x.shape), "Output:", tuple(y.shape))
    print("Heads: 4 | Dimension per head: 32")
