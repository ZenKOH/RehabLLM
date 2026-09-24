from __future__ import annotations

import math
from pathlib import Path

import torch

from .data import random_batch
from .model import RehabMiniLLM


@torch.no_grad()
def estimate_loss(
    model: RehabMiniLLM,
    token_path: str | Path,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        x, y = random_batch(token_path, batch_size, context_length, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))
