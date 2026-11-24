#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rehab_minillm.instruction import (
    InstructionExample,
    example_fingerprint,
    prompt_fingerprint,
    validate_instruction_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, deduplicate and split RehabLLM instruction data"
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-dir", default="data/v04/instructions")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")

    unique: dict[str, dict[str, Any]] = {}
    for source in args.input:
        with Path(source).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                example = InstructionExample.from_dict(raw)
                fingerprint = example_fingerprint(example)
                unique[fingerprint] = {
                    **raw,
                    "id": example.id or fingerprint[:12],
                    "instruction": example.instruction,
                    "context": example.context,
                    "response": example.response,
                }

    ordered = sorted(unique.items(), key=lambda item: item[0])
    groups = sorted({prompt_fingerprint(InstructionExample.from_dict(row)) for row in unique.values()})
    if len(groups) < 2:
        raise ValueError("Need at least two distinct instruction/context groups to split")
    val_groups = set(groups[:max(1, round(len(groups) * args.val_fraction))])
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for _, row in ordered:
        if prompt_fingerprint(InstructionExample.from_dict(row)) in val_groups:
            val_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows or not val_rows:
        raise ValueError("Need enough instruction examples to create both train and validation sets")
    validate_instruction_splits(
        [InstructionExample.from_dict(row) for row in train_rows],
        [InstructionExample.from_dict(row) for row in val_rows],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input_files": args.input,
        "unique_examples": len(ordered),
        "prompt_groups": len(groups),
        "split_unit": "normalised_instruction_and_context",
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "val_fraction_requested": args.val_fraction,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
