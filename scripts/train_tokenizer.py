#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rehab_minillm.tokenizer import train_sentencepiece


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/pmc_rehab.txt")
    parser.add_argument("--prefix", default="data/processed/rehab_sp")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    args = parser.parse_args()

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    train_sentencepiece(
        args.input,
        prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )
    print(f"tokenizer written to {prefix}.model and {prefix}.vocab")


if __name__ == "__main__":
    main()
