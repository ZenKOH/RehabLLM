#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from rehab_minillm.tokenizer import SentencePieceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenise a text corpus into train/val int32 files")
    parser.add_argument("--input", default="data/raw/pmc_rehab.txt")
    parser.add_argument("--tokenizer", default="data/processed/rehab_sp.model")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    docs = [doc.strip() for doc in text.split("<eos>") if doc.strip()]
    random.Random(args.seed).shuffle(docs)

    split = max(1, int(len(docs) * args.val_fraction)) if len(docs) > 1 else 0
    val_docs = docs[:split]
    train_docs = docs[split:] if split else docs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in (("train", train_docs), ("val", val_docs or train_docs[-1:])):
        token_ids: list[int] = []
        for doc in subset:
            token_ids.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))
        arr = np.asarray(token_ids, dtype=np.int32)
        path = out_dir / f"{name}.bin"
        arr.tofile(path)
        print(f"{name}: {len(subset)} docs, {len(arr):,} tokens -> {path}")


if __name__ == "__main__":
    main()
