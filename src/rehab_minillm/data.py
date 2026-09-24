from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped next-token dataset from an int32 token file."""

    def __init__(self, token_path: str | Path, context_length: int) -> None:
        self.tokens = np.memmap(token_path, dtype=np.int32, mode="r")
        self.context_length = context_length
        if len(self.tokens) <= context_length:
            raise ValueError("Token file is too small for the configured context length")

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - 1) // self.context_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.context_length
        end = start + self.context_length + 1
        if end > len(self.tokens):
            start = len(self.tokens) - self.context_length - 1
            end = len(self.tokens)
        chunk = np.asarray(self.tokens[start:end], dtype=np.int64).copy()
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def random_batch(
    token_path: str | Path,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.memmap(token_path, dtype=np.int32, mode="r")
    if len(data) <= context_length + 1:
        raise ValueError("Not enough tokens to sample a batch")

    max_start = len(data) - context_length - 1
    starts = [random.randint(0, max_start) for _ in range(batch_size)]
    xs = [torch.from_numpy(np.asarray(data[i : i + context_length], dtype=np.int64).copy()) for i in starts]
    ys = [torch.from_numpy(np.asarray(data[i + 1 : i + 1 + context_length], dtype=np.int64).copy()) for i in starts]
    return torch.stack(xs).to(device), torch.stack(ys).to(device)
