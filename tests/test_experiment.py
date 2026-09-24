from pathlib import Path

import numpy as np
import torch

from rehab_minillm.config import ExperimentConfig, ModelConfig, TrainingConfig
from rehab_minillm.experiment import build_run_manifest, sha256_file


def test_run_manifest_hashes_datasets(tmp_path: Path):
    train = tmp_path / "train.bin"
    val = tmp_path / "val.bin"
    np.asarray([1, 2, 3, 4], dtype=np.int32).tofile(train)
    np.asarray([5, 6, 7, 8], dtype=np.int32).tofile(val)
    cfg = ExperimentConfig(
        model=ModelConfig(vocab_size=32, context_length=4, n_layers=1, n_heads=1, d_model=8, d_ff=16),
        training=TrainingConfig(max_steps=1),
    )
    manifest = build_run_manifest(cfg, train, val, model_parameters=123, device=torch.device("cpu"))
    assert manifest["datasets"]["train"]["sha256"] == sha256_file(train)
    assert manifest["datasets"]["val"]["sha256"] == sha256_file(val)
    assert manifest["model_parameters"] == 123
