#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rehab_minillm.cleaning import clean_document
from rehab_minillm.curation import QualityConfig, quality_metrics, quality_reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean publishing boilerplate from a corpus")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="data/v04/clean_articles.jsonl")
    parser.add_argument("--min-words", type=int, default=200)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quality = QualityConfig(
        min_words=args.min_words,
        max_words=30_000,
        min_alpha_word_fraction=0.70,
        max_symbol_to_word_ratio=0.25,
        min_median_word_length=3.0,
        max_median_word_length=12.0,
        max_duplicate_line_fraction=0.50,
    )
    stats: Counter[str] = Counter()
    removed_chars = 0

    with input_path.open("r", encoding="utf-8") as source, out_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            result = clean_document(str(record.get("text") or ""))
            metrics = quality_metrics(result.text)
            reasons = quality_reasons(metrics, quality)
            if reasons:
                stats.update(f"reject:{reason}" for reason in reasons)
                continue

            record["text"] = result.text
            record["cleaning"] = result.to_dict()
            record["cleaned_words"] = metrics.words
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["accepted"] += 1
            removed_chars += result.removed_chars
            for key, value in result.removed_sections.items():
                stats[f"removed_section:{key}"] += value
            for key, value in result.removed_inline.items():
                stats[f"removed_inline:{key}"] += value
            stats["anchors_removed"] += result.anchors_removed
            stats["html_tags_removed"] += result.html_tags_removed

    summary = {
        "input": str(input_path),
        "output": str(out_path),
        "removed_chars": removed_chars,
        "counts": dict(stats),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
