#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from rehab_minillm.generate import load_checkpoint
from rehab_minillm.inference import generate_response
from rehab_minillm.tokenizer import SentencePieceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an instruction-style RehabLLM answer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    model = load_checkpoint(args.checkpoint, device)

    result = generate_response(
        model, tokenizer, args.instruction, context=args.context, seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(result["output"])
    if result["stop_reason"] != "eos":
        import sys
        print(f"Generation stopped at {result['stop_reason']}; answer may be incomplete.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
