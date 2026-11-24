import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from rehab_minillm.config import ModelConfig
from rehab_minillm.data import random_batch
from rehab_minillm.evaluate import estimate_loss, perplexity
from rehab_minillm.model import RehabMiniLLM
from rehab_minillm.sft import estimate_sft_loss


def test_sampled_evaluation_is_reproducible_and_does_not_perturb_training_rng(tmp_path):
    path = tmp_path / "test.bin"
    (np.arange(100) % 32).astype(np.int32).tofile(path)
    model = RehabMiniLLM(ModelConfig(
        vocab_size=32, context_length=8, n_layers=1, n_heads=2, d_model=8, d_ff=16,
    )).train()
    rng_state = random.getstate()
    a = estimate_loss(model, path, 2, 8, 3, torch.device("cpu"))
    assert random.getstate() == rng_state
    random.random()
    b = estimate_loss(model, path, 2, 8, 3, torch.device("cpu"))
    assert a == b
    assert model.training
    with pytest.raises(FileNotFoundError):
        estimate_loss(model, tmp_path / "absent.bin", 2, 8, 3, torch.device("cpu"))
    assert model.training


def test_exactly_one_context_plus_target_is_a_valid_batch(tmp_path):
    path = tmp_path / "one.bin"
    np.arange(9).astype(np.int32).tofile(path)
    x, y = random_batch(path, 1, 8, torch.device("cpu"))
    assert x.tolist() == [list(range(8))]
    assert y.tolist() == [list(range(1, 9))]


def test_perplexity_is_not_silently_clipped():
    assert perplexity(25) == math.exp(25)
    with pytest.raises(ValueError, match="non-finite"):
        perplexity(float("nan"))


def test_sft_validation_weights_by_response_tokens_and_preserves_mode():
    class KnownLoss(torch.nn.Module):
        def forward(self, x, y):
            return None, y[y != -100].float().mean()

    labels = torch.tensor([[-100, -100, -100, 1], [3, 3, 3, -100]])
    loader = DataLoader(TensorDataset(torch.zeros_like(labels), labels), batch_size=1)
    model = KnownLoss().eval()
    result = estimate_sft_loss(model, loader, 10, torch.device("cpu"))
    assert result == 2.5
    assert not model.training


def test_comparison_rejects_unmatched_evaluation_inputs():
    path = Path(__file__).parents[1] / "scripts" / "compare_evaluations.py"
    spec = importlib.util.spec_from_file_location("compare_evaluations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {
        "schema_version": 2, "evaluation_config": {"seed": 42},
        "input_sha256": {"checkpoint": "weights", "tokenizer": "t", "test": "d", "prompts": "p"},
        "generations": [{"id": "question", "prompt": "Explain.", "output": "Answer",
                         "stop_reason": "eos"}],
        "test_loss": 4.2,
    }
    assert module.compare_reports(report, report)["review_status"] == "pending"
    candidate = {**report, "evaluation_config": {"seed": 123}}
    with pytest.raises(ValueError, match="settings differ"):
        module.compare_reports(report, candidate)
    candidate = {**report, "input_sha256": {**report["input_sha256"], "test": "other"}}
    with pytest.raises(ValueError, match="input differs"):
        module.compare_reports(report, candidate)
