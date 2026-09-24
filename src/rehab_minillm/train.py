from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .data import random_batch
from .evaluate import estimate_loss, perplexity
from .experiment import build_run_manifest, write_run_manifest
from .model import RehabMiniLLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, config: ExperimentConfig) -> float:
    train = config.training
    if step < train.warmup_steps:
        return train.learning_rate * (step + 1) / max(1, train.warmup_steps)
    if step >= train.max_steps:
        return train.min_learning_rate
    ratio = (step - train.warmup_steps) / max(1, train.max_steps - train.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return train.min_learning_rate + coeff * (
        train.learning_rate - train.min_learning_rate
    )


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    out_path = Path(out_dir)
    candidates = list(out_path.glob("step_*.pt"))
    if not candidates:
        return None

    def step_number(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    return max(candidates, key=step_number)


def resolve_resume_checkpoint(
    resume: str | Path | None,
    out_dir: str | Path,
) -> Path | None:
    if resume is None:
        return None
    if str(resume).lower() == "latest":
        return latest_checkpoint(out_dir)

    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def save_checkpoint(
    path: Path,
    model: RehabMiniLLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_config": asdict(config.model),
            "training_config": asdict(config.training),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        path,
    )


def train(
    config_path: str | Path,
    train_tokens: str | Path,
    val_tokens: str | Path,
    out_dir: str | Path,
    resume: str | Path | None = None,
) -> None:
    config = load_config(config_path)
    train_cfg = config.training
    model_cfg = config.model
    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RehabMiniLLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )

    amp_enabled = bool(train_cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_checkpoint(resume, out_dir)
    start_step = 0
    if resume_path is not None:
        payload = torch.load(resume_path, map_location=device)
        checkpoint_model_config = payload.get("model_config")
        if checkpoint_model_config != asdict(model_cfg):
            raise ValueError(
                "Resume checkpoint model configuration does not match the current config."
            )
        start_step = int(payload.get("step", 0))
        if start_step > train_cfg.max_steps:
            raise ValueError(
                f"Checkpoint step {start_step} exceeds configured max_steps "
                f"{train_cfg.max_steps}."
            )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scaler_state = payload.get("scaler_state")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        print(f"resuming_from={resume_path}")
        print(f"resume_step={start_step}")

    manifest = build_run_manifest(
        config,
        train_tokens,
        val_tokens,
        model.parameter_count(),
        device,
    )
    manifest["resume"] = {
        "checkpoint": str(resume_path) if resume_path is not None else None,
        "start_step": start_step,
    }
    write_run_manifest(out_dir / "run_manifest.json", manifest)

    print(f"device={device}")
    print(f"parameters={model.parameter_count():,}")
    print(f"run_manifest={out_dir / 'run_manifest.json'}")

    model.train()
    for step in range(start_step, train_cfg.max_steps):
        lr = cosine_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(train_cfg.gradient_accumulation_steps):
            x, y = random_batch(
                train_tokens,
                train_cfg.batch_size,
                model_cfg.context_length,
                device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                _, loss = model(x, y)
                assert loss is not None
                loss = loss / train_cfg.gradient_accumulation_steps
            running_loss += float(loss.detach().item())
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        current_step = step + 1
        if current_step % 25 == 0:
            print(f"step={current_step} train_loss={running_loss:.4f} lr={lr:.3e}")

        if current_step % train_cfg.eval_interval == 0:
            val_loss = estimate_loss(
                model,
                val_tokens,
                train_cfg.batch_size,
                model_cfg.context_length,
                train_cfg.eval_batches,
                device,
            )
            print(
                f"step={current_step} val_loss={val_loss:.4f} "
                f"perplexity={perplexity(val_loss):.2f}"
            )

        if current_step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(
                out_dir / f"step_{current_step:07d}.pt",
                model,
                optimizer,
                scaler,
                current_step,
                config,
            )

    save_checkpoint(
        out_dir / "final.pt",
        model,
        optimizer,
        scaler,
        train_cfg.max_steps,
        config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RehabMiniLLM")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--train", default="data/processed/train.bin")
    parser.add_argument("--val", default="data/processed/val.bin")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path or 'latest' to resume from the newest step_*.pt in --out.",
    )
    args = parser.parse_args()
    train(args.config, args.train, args.val, args.out, resume=args.resume)


if __name__ == "__main__":
    main()
