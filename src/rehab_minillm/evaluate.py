from __future__ import annotations

import math
import random
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
    seed: int = 42,
) -> float:
    if eval_batches < 1 or batch_size < 1:
        raise ValueError("eval_batches and batch_size must be positive")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    rng = random.Random(seed)
    try:
        for _ in range(eval_batches):
            x, y = random_batch(token_path, batch_size, context_length, device, rng=rng)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        raise ValueError("Cannot compute perplexity from non-finite loss")
    return math.exp(loss)
