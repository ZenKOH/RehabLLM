from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_manifest(
    config: ExperimentConfig,
    train_tokens: str | Path,
    val_tokens: str | Path,
    model_parameters: int,
    device: torch.device,
) -> dict[str, Any]:
    train_path = Path(train_tokens)
    val_path = Path(val_tokens)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "model_parameters": model_parameters,
        "config": {"model": asdict(config.model), "training": asdict(config.training)},
        "datasets": {
            "train": {
                "path": str(train_path),
                "bytes": train_path.stat().st_size,
                "sha256": sha256_file(train_path),
            },
            "val": {
                "path": str(val_path),
                "bytes": val_path.stat().st_size,
                "sha256": sha256_file(val_path),
            },
        },
    }


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "
", encoding="utf-8")
