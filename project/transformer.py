"""From-scratch Transformer language model for CyberGuard."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Multi-Head Attention -> Residual+Norm -> FFN -> Residual+Norm."""
    def __init__(self, embedding_dim, number_of_heads, context_length, ffn_dim, dropout):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embedding_dim, number_of_heads, context_length, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, ffn_dim, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x


class CyberGuardTransformer(nn.Module):
    def __init__(self, vocab_size, context_length=128, embedding_dim=128,
                 number_of_heads=4, number_of_layers=3, ffn_dim=256,
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.context_length = context_length
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, number_of_heads, context_length, ffn_dim, dropout)
            for _ in range(number_of_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        if T > self.context_length:
            raise ValueError(f"Sequence length {T} exceeds context length {self.context_length}")
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        hidden_states = self.final_norm(x)
        logits = self.output_layer(hidden_states)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=self.pad_id)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=300, temperature=0.8, top_k=20):
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.context_length:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = values[:, [-1]]
                logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


if __name__ == "__main__":
    model = CyberGuardTransformer(vocab_size=100)
    x = torch.randint(0, 100, (2, 32))
    logits, loss = model(x, x)
    print("TRANSFORMER TEST PASSED")
    print("Input:", tuple(x.shape), "Logits:", tuple(logits.shape), "Loss:", float(loss.detach()))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
