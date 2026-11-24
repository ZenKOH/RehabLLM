#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from rehab_minillm.evaluate import estimate_loss, perplexity
from rehab_minillm.experiment import git_commit, sha256_file
from rehab_minillm.generate import load_checkpoint
from rehab_minillm.inference import generate_response
from rehab_minillm.tokenizer import SentencePieceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RehabMiniLLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--test", default="data/processed/test.bin")
    parser.add_argument("--prompts", default="data/eval_prompts.jsonl")
    parser.add_argument("--out", default="eval/results.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--prompt-format", choices=("instruction", "continuation"),
                        default="continuation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    model = load_checkpoint(args.checkpoint, device)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match the checkpoint")
    loss = estimate_loss(
        model,
        args.test,
        args.batch_size,
        model.config.context_length,
        args.eval_batches,
        device,
        seed=args.seed,
    )

    generations = []
    seen_ids = set()
    with Path(args.prompts).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            identifier = item.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                raise ValueError("Benchmark questions require unique, non-empty string IDs")
            seen_ids.add(identifier)
            generated = generate_response(
                model, tokenizer, item["prompt"], context=item.get("context", ""),
                prompt_format=args.prompt_format, seed=args.seed,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
            )
            generations.append(
                {
                    "id": identifier,
                    "domain": item.get("domain"),
                    "prompt": item["prompt"],
                    "context": item.get("context", ""),
                    "expected_behaviour": item.get("expected_behaviour"),
                    **generated,
                    "human_review": None,
                    "notes": None,
                }
            )

    if not generations:
        raise ValueError("Benchmark prompt file is empty")

    result = {
        "schema_version": 2,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "input_sha256": {name: sha256_file(path) for name, path in (
            ("checkpoint", args.checkpoint), ("tokenizer", args.tokenizer),
            ("test", args.test), ("prompts", args.prompts)
        )},
        "evaluation_config": {
            "seed": args.seed, "prompt_format": args.prompt_format,
            "batch_size": args.batch_size, "eval_batches": args.eval_batches,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "top_k": args.top_k, "top_p": args.top_p,
        },
        "test_loss": loss,
        "test_perplexity": perplexity(loss),
        "device": str(device),
        "generations": generations,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"test_loss": loss, "test_perplexity": perplexity(loss), "prompts": len(generations)}, indent=2))


if __name__ == "__main__":
    main()
