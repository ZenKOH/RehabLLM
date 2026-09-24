#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rehab_minillm.instruction import (
    InstructionExample,
    example_fingerprint,
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

    unique: dict[str, dict[str, str]] = {}
    for source in args.input:
        with Path(source).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                example = InstructionExample.from_dict(raw)
                fingerprint = example_fingerprint(example)
                unique[fingerprint] = {
                    "id": example.id or fingerprint[:12],
                    "instruction": example.instruction,
                    "context": example.context,
                    "response": example.response,
                }

    ordered = sorted(unique.items(), key=lambda item: item[0])
    val_every = max(2, round(1.0 / args.val_fraction))
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []

    for index, (_, row) in enumerate(ordered):
        if index % val_every == 0:
            val_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows or not val_rows:
        raise ValueError("Need enough instruction examples to create both train and validation sets")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input_files": args.input,
        "unique_examples": len(ordered),
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
