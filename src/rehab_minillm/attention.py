from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention implemented directly from Q, K and V.

    This intentionally avoids a ready-made GPT block so the core mechanism remains inspectable.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=config.bias,
        )
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        if channels != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, got {channels}")
        if seq_len > self.causal_mask.size(-1):
            raise ValueError("Sequence length exceeds configured context length")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, C) -> (B, H, T, D)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.out_proj(y)
        return self.resid_dropout(y)
