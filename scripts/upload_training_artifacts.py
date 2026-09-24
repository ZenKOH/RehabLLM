#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload RehabLLM training artefacts to a private Hugging Face model repo"
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints/l4-v0.3")
    parser.add_argument("--tokenizer-prefix", default="data/processed/rehab_sp")
    parser.add_argument("--eval", default="eval/l4_v0.3.json")
    parser.add_argument("--stats", default="data/curated/gpu_corpus_stats.json")
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required so training artefacts survive the ephemeral Job.")

    api = HfApi(token=token)
    who = api.whoami(token=token)
    owner = who["name"]
    job_id = os.environ.get("JOB_ID", "manual")
    repo_id = args.repo_id or f"{owner}/RehabLLM-17M-v0.3-{job_id}"

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
        token=token,
    )

    stage = Path("hf_artifacts")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    for name in ("final.pt", "run_manifest.json"):
        source = checkpoint_dir / name
        if source.exists():
            shutil.copy2(source, stage / name)

    for suffix in (".model", ".vocab"):
        source = Path(str(args.tokenizer_prefix) + suffix)
        if source.exists():
            shutil.copy2(source, stage / source.name)

    for source_path in (
        Path(args.eval),
        Path(args.stats),
        Path("configs/gpu_l4.yaml"),
        Path("MODEL_CARD.md"),
    ):
        if source_path.exists():
            shutil.copy2(source_path, stage / source_path.name)

    (stage / "README.md").write_text(
        "# RehabLLM 17M v0.3 GPU training run\n\n"
        "Private training artefacts produced by the RehabLLM v0.3 substantial GPU pipeline.\n\n"
        "This is a research model, not a clinically validated system and not medical advice.\n",
        encoding="utf-8",
    )

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=stage,
        commit_message=f"Upload RehabLLM GPU training artefacts from Job {job_id}",
        token=token,
    )
    print(f"uploaded_to=https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
