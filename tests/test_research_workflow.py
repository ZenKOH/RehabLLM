"""Real tokenizer → SFT → checkpoint → matched evaluation → comparison smoke test."""
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from rehab_minillm.config import ExperimentConfig, ModelConfig, TrainingConfig
from rehab_minillm.model import RehabMiniLLM
from rehab_minillm.sft import train_sft

ROOT = Path(__file__).parents[1]


def test_real_sft_and_evaluation_workflow(tmp_path, real_tokenizer):
    config = ExperimentConfig(
        model=ModelConfig(vocab_size=real_tokenizer.vocab_size, context_length=512,
                          n_layers=1, n_heads=2, d_model=16, d_ff=32, dropout=0),
        training=TrainingConfig(batch_size=2, gradient_accumulation_steps=1, max_steps=2,
                                warmup_steps=0, eval_interval=1, eval_batches=2,
                                checkpoint_interval=1, amp=False),
    )
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
    base = tmp_path / "base.pt"
    model = RehabMiniLLM(config.model)
    torch.save({"model_config": asdict(config.model), "model_state": model.state_dict()}, base)
    for split, question in (("train", "Explain research."), ("val", "Compare sources.")):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps({
            "instruction": question, "response": "Complete answers require evidence."
        }) + "\n", encoding="utf-8")
    out = tmp_path / "sft"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    audit = tmp_path / "instruction_audit.json"
    subprocess.run([
        sys.executable, "scripts/audit_instructions.py",
        "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"),
        "--tokenizer", real_tokenizer.model_path, "--context-length", "512", "--out", str(audit),
    ], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert json.loads(audit.read_text())["ready"]
    train_sft(config_path, base, real_tokenizer.model_path, tmp_path / "train.jsonl",
              tmp_path / "val.jsonl", out)
    final = torch.load(out / "final.pt", map_location="cpu")
    assert final["step"] == 2
    assert any(not torch.equal(model.state_dict()[k], value)
               for k, value in final["model_state"].items())
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["split_overlap_checked"]
    assert len(manifest["input_sha256"]["tokenizer"]) == 64
    prompts = ROOT / "data" / "eval_prompts.jsonl"
    tokens = tmp_path / "test.bin"
    np.array(real_tokenizer.encode("Research evidence. " * 200), dtype=np.int32).tofile(tokens)
    for name, checkpoint in (("base", base), ("candidate", out / "final.pt")):
        subprocess.run([
            sys.executable, "scripts/evaluate_model.py", "--checkpoint", str(checkpoint),
            "--tokenizer", real_tokenizer.model_path, "--test", str(tokens),
            "--prompts", str(prompts), "--prompt-format", "instruction",
            "--max-new-tokens", "4", "--eval-batches", "1", "--batch-size", "1",
            "--out", str(tmp_path / f"{name}.json"),
        ], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    comparison = tmp_path / "comparison.json"
    subprocess.run([
        sys.executable, "scripts/compare_evaluations.py", "--baseline", str(tmp_path / "base.json"),
        "--candidate", str(tmp_path / "candidate.json"), "--out", str(comparison),
    ], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    report = json.loads(comparison.read_text())
    assert report["review_status"] == "pending"
    assert len(report["comparisons"]) == 14
