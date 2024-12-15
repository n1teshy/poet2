import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import dropout


class Head(nn.Module):
    def __init__(self, model_dim: int, head_dim: int):
        super().__init__()
        self.kW = nn.Linear(model_dim, head_dim)
        self.vW = nn.Linear(model_dim, head_dim)
        self.qW = nn.Linear(model_dim, head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, k: torch.Tensor, v: torch.testing, q: torch.Tensor, mask: torch.Tensor
    ):
        B, T, C = q.shape
        k = self.kW(k)  # (B, Te, C) @ (C, h) -> (B, Te, h)
        q = self.qW(q)  # (B, Td, C) @ (C, h) -> (B, Td, h)
        attn = (
            q @ k.transpose(-1, -2) * C**-0.5
        )  # (B, Td, h) @ (B, h, Te) -> (B, Td, Te)
        attn = attn.masked_fill(mask[:, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        v = self.vW(v)  # (B, Te, C) @ (C, h) -> (B, Te, h)
        out = attn @ v  # (B, Td, Te) @ (B, Te, h) -> (B, Td, h)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(model_dim, head_dim) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v, q, mask):
        out = torch.cat([h(k=k, q=q, v=v, mask=mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        assert (
            model_dim % num_heads == 0
        ), "embedding size must be divisible by number of heads"
        head_dim = model_dim // num_heads
        self.self_attn = MultiheadAttention(model_dim, num_heads, head_dim)
        self.ln1 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffwd = FeedForward(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x, mask):
        y = self.self_attn(k=x, q=x, v=x, mask=mask)
        y = self.dropout(self.ln1(x + y))
        y = self.ln2(x + self.ffwd(y))
        return y


class Decoder(nn.Module):
    def __init__(self, model_dim: int, blocks: int, heads: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(model_dim, heads, is_decoder=False) for _ in range(blocks)]
        )

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return x
