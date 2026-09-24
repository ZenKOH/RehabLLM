#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rehab_minillm.tokenizer import SentencePieceTokenizer


def encode_file(input_path: Path, output_path: Path, tokenizer: SentencePieceTokenizer) -> int:
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    documents = [doc.strip() for doc in text.split("<eos>") if doc.strip()]
    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document, add_bos=True, add_eos=True))
    array = np.asarray(token_ids, dtype=np.int32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(output_path)
    return len(array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenise curated train/val/test corpora into int32 files")
    parser.add_argument("--train-text", default="data/curated/train.txt")
    parser.add_argument("--val-text", default="data/curated/val.txt")
    parser.add_argument("--test-text", default="data/curated/test.txt")
    parser.add_argument("--tokenizer", default="data/processed/rehab_sp.model")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    out_dir = Path(args.out_dir)
    for split, source in (("train", args.train_text), ("val", args.val_text), ("test", args.test_text)):
        count = encode_file(Path(source), out_dir / f"{split}.bin", tokenizer)
        print(f"{split}: {count:,} tokens -> {out_dir / f'{split}.bin'}")


if __name__ == "__main__":
    main()
