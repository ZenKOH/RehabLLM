from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .instruction import InstructionDataset, load_instruction_jsonl
from .model import RehabMiniLLM
from .tokenizer import SentencePieceTokenizer
from .train import cosine_lr, latest_checkpoint, set_seed


def _next_batch(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    iterator,
):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


@torch.no_grad()
def estimate_sft_loss(
    model: RehabMiniLLM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    iterator = iter(loader)
    for _ in range(max(1, batches)):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def save_sft_checkpoint(
    path: Path,
    model: RehabMiniLLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: ExperimentConfig,
    base_checkpoint: str | Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "sft",
            "step": step,
            "base_checkpoint": str(base_checkpoint),
            "model_config": asdict(config.model),
            "training_config": asdict(config.training),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        path,
    )


def train_sft(
    config_path: str | Path,
    base_checkpoint: str | Path,
    tokenizer_path: str | Path,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    out_dir: str | Path,
    resume: str | Path | None = None,
) -> None:
    config = load_config(config_path)
    train_cfg = config.training
    model_cfg = config.model
    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer.vocab_size} does not match "
            f"model vocab size {model_cfg.vocab_size}."
        )

    base_path = Path(base_checkpoint)
    if not base_path.exists():
        raise FileNotFoundError(f"Base checkpoint does not exist: {base_path}")

    payload = torch.load(base_path, map_location=device)
    if payload.get("model_config") != asdict(model_cfg):
        raise ValueError("Base checkpoint model configuration does not match the SFT config.")

    model = RehabMiniLLM(model_cfg).to(device)
    model.load_state_dict(payload["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )

    amp_enabled = bool(train_cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_examples = load_instruction_jsonl(train_jsonl)
    val_examples = load_instruction_jsonl(val_jsonl)
    train_dataset = InstructionDataset(train_examples, tokenizer, model_cfg.context_length)
    val_dataset = InstructionDataset(val_examples, tokenizer, model_cfg.context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    start_step = 0

    resume_path: Path | None = None
    if resume is not None:
        resume_path = latest_checkpoint(out_path) if str(resume).lower() == "latest" else Path(resume)
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"SFT resume checkpoint does not exist: {resume_path}")
            resume_payload = torch.load(resume_path, map_location=device)
            if resume_payload.get("model_config") != asdict(model_cfg):
                raise ValueError("SFT resume checkpoint model config does not match.")
            model.load_state_dict(resume_payload["model_state"])
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            scaler_state = resume_payload.get("scaler_state")
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            start_step = int(resume_payload.get("step", 0))

    manifest = {
        "stage": "sft",
        "device": str(device),
        "parameters": model.parameter_count(),
        "base_checkpoint": str(base_path),
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "start_step": start_step,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "tokenizer": str(tokenizer_path),
        "model_config": asdict(model_cfg),
        "training_config": asdict(train_cfg),
    }
    import json

    (out_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"device={device}")
    print(f"parameters={model.parameter_count():,}")
    print(f"train_examples={len(train_examples):,} val_examples={len(val_examples):,}")
    if resume_path:
        print(f"resuming_from={resume_path} step={start_step}")

    model.train()
    iterator = iter(train_loader)
    for step in range(start_step, train_cfg.max_steps):
        lr = cosine_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(train_cfg.gradient_accumulation_steps):
            (x, y), iterator = _next_batch(train_loader, iterator)
            x = x.to(device)
            y = y.to(device)
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
        if current_step % 10 == 0:
            print(f"step={current_step} sft_train_loss={running_loss:.4f} lr={lr:.3e}")

        if current_step % train_cfg.eval_interval == 0:
            val_loss = estimate_sft_loss(
                model,
                val_loader,
                train_cfg.eval_batches,
                device,
            )
            print(f"step={current_step} sft_val_loss={val_loss:.4f}")

        if current_step % train_cfg.checkpoint_interval == 0:
            save_sft_checkpoint(
                out_path / f"step_{current_step:07d}.pt",
                model,
                optimizer,
                scaler,
                current_step,
                config,
                base_path,
            )

    save_sft_checkpoint(
        out_path / "final.pt",
        model,
        optimizer,
        scaler,
        train_cfg.max_steps,
        config,
        base_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised instruction-tune RehabLLM")
    parser.add_argument("--config", default="configs/v04_sft.yaml")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train", default="data/v04/instructions/train.jsonl")
    parser.add_argument("--val", default="data/v04/instructions/val.jsonl")
    parser.add_argument("--out", default="checkpoints/v04_sft")
    parser.add_argument(
        "--resume",
        default=None,
        help="SFT checkpoint path or 'latest'.",
    )
    args = parser.parse_args()
    train_sft(
        args.config,
        args.base_checkpoint,
        args.tokenizer,
        args.train,
        args.val,
        args.out,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
