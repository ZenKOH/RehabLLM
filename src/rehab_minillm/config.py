from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8000
    context_length: int = 512
    n_layers: int = 8
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.vocab_size <= 0 or self.context_length <= 0:
            raise ValueError("vocab_size and context_length must be positive")
        if self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("n_layers and n_heads must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 50_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_batches: int = 50
    checkpoint_interval: int = 2500
    seed: int = 42
    amp: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    model = ModelConfig(**raw.get("model", {}))
    model.validate()
    training = TrainingConfig(**raw.get("training", {}))
    return ExperimentConfig(model=model, training=training)
