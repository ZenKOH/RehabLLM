#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from rehab_minillm.evaluate import estimate_loss, perplexity
from rehab_minillm.generate import load_checkpoint
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    model = load_checkpoint(args.checkpoint, device)
    loss = estimate_loss(
        model,
        args.test,
        args.batch_size,
        model.config.context_length,
        args.eval_batches,
        device,
    )

    generations = []
    with Path(args.prompts).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            ids = tokenizer.encode(item["prompt"], add_bos=True)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=0.8, top_k=50, top_p=0.95)
            generations.append(
                {
                    "id": item.get("id"),
                    "domain": item.get("domain"),
                    "prompt": item["prompt"],
                    "expected_behaviour": item.get("expected_behaviour"),
                    "output": tokenizer.decode(out[0].tolist()),
                    "human_review": None,
                    "notes": None,
                }
            )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "test_loss": loss,
        "test_perplexity": perplexity(loss),
        "device": str(device),
        "generations": generations,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "
", encoding="utf-8")
    print(json.dumps({"test_loss": loss, "test_perplexity": perplexity(loss), "prompts": len(generations)}, indent=2))


if __name__ == "__main__":
    main()
