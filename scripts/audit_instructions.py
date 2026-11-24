#!/usr/bin/env python3
"""CPU preflight for SFT length, complete answers and split isolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rehab_minillm.experiment import sha256_file
from rehab_minillm.instruction import (
    InstructionDataset,
    load_instruction_jsonl,
    validate_instruction_splits,
)
from rehab_minillm.tokenizer import SentencePieceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--out", default="reports/instruction_audit.json")
    args = parser.parse_args()
    report = {"ready": False, "context_length": args.context_length, "splits": {}, "errors": []}
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    train, val = load_instruction_jsonl(args.train), load_instruction_jsonl(args.val)
    try:
        validate_instruction_splits(train, val)
    except ValueError as exc:
        report["errors"].append(str(exc))
    for split, examples in (("train", train), ("val", val)):
        try:
            dataset = InstructionDataset(examples, tokenizer, args.context_length)
            report["splits"][split] = {
                "examples": len(examples),
                "supervised_tokens": sum(int((dataset[i][1] != -100).sum())
                                         for i in range(len(dataset))),
            }
        except ValueError as exc:
            report["errors"].append(f"{split}: {exc}")
    report["input_sha256"] = {name: sha256_file(path) for name, path in (
        ("train", args.train), ("val", args.val), ("tokenizer", args.tokenizer)
    )}
    report["ready"] = not report["errors"]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
