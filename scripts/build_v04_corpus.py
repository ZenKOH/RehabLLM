#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rehab_minillm.cleaning import clean_document
from rehab_minillm.curation import (
    DedupConfig,
    QualityConfig,
    SplitConfig,
    curate_records,
)
from rehab_minillm.domain import DEFAULT_V04_MIX, classify_record, target_counts


def stable_rank(identifier: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_input_source"] = str(path)
                records.append(record)
    return records


def select_mixture(
    records: list[dict[str, Any]],
    target_docs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = classify_record(record)
        if category is None:
            continue
        record["category"] = category
        groups[category].append(record)

    for group in groups.values():
        group.sort(key=lambda row: stable_rank(str(row.get("id")), seed))

    targets = target_counts(target_docs, DEFAULT_V04_MIX)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for category, quota in targets.items():
        for record in groups.get(category, [])[:quota]:
            selected.append(record)
            selected_ids.add(str(record.get("id")))

    if len(selected) < target_docs:
        leftovers: list[dict[str, Any]] = []
        for group in groups.values():
            leftovers.extend(
                record
                for record in group
                if str(record.get("id")) not in selected_ids
            )
        leftovers.sort(key=lambda row: stable_rank(str(row.get("id")), seed + 1))
        for record in leftovers[: max(0, target_docs - len(selected))]:
            selected.append(record)
            selected_ids.add(str(record.get("id")))

    actual = Counter(str(record.get("category")) for record in selected)
    available = {category: len(group) for category, group in groups.items()}
    return selected, {
        "target_docs": target_docs,
        "target_mix": DEFAULT_V04_MIX,
        "target_counts": targets,
        "available_counts": available,
        "selected_counts": dict(actual),
        "selected_docs": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the cleaned, enriched RehabLLM v0.4 domain corpus"
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-dir", default="data/v04")
    parser.add_argument("--target-docs", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_jsonl([Path(value) for value in args.input])
    cleaned_records: list[dict[str, Any]] = []
    clean_stats: Counter[str] = Counter()

    for record in raw:
        result = clean_document(str(record.get("text") or ""))
        record = dict(record)
        record["text"] = result.text
        record["cleaning"] = result.to_dict()
        cleaned_records.append(record)
        clean_stats["documents"] += 1
        clean_stats["removed_chars"] += result.removed_chars
        for key, value in result.removed_sections.items():
            clean_stats[f"removed_section:{key}"] += value
        for key, value in result.removed_inline.items():
            clean_stats[f"removed_inline:{key}"] += value
        clean_stats["anchors_removed"] += result.anchors_removed
        clean_stats["html_tags_removed"] += result.html_tags_removed

    curated = curate_records(
        cleaned_records,
        quality_config=QualityConfig(
            min_words=200,
            max_words=30_000,
            min_alpha_word_fraction=0.70,
            max_symbol_to_word_ratio=0.25,
            min_median_word_length=3.0,
            max_median_word_length=12.0,
            max_duplicate_line_fraction=0.50,
        ),
        dedup_config=DedupConfig(
            shingle_size=5,
            num_perm=64,
            bands=16,
            jaccard_threshold=0.82,
            seed=args.seed,
        ),
        split_config=SplitConfig(train=0.90, val=0.05, test=0.05, seed=args.seed),
    )

    selected, mixture = select_mixture(curated.accepted, args.target_docs, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(selected)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_handles = {
        split: (out_dir / f"{split}.txt").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }

    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    words = 0

    try:
        with (out_dir / "articles.jsonl").open("w", encoding="utf-8") as handle:
            for record in selected:
                category = classify_record(record)
                if category is None:
                    continue
                record["category"] = category
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                split = str(record["split"])
                split_handles[split].write(str(record["text"]) + "\n<eos>\n")
                source_counts[str(record.get("_input_source") or record.get("source") or "unknown")] += 1
                split_counts[split] += 1
                category_counts[category] += 1
                words += int(record.get("metrics", {}).get("words") or 0)
    finally:
        for handle in split_handles.values():
            handle.close()

    report = {
        "schema_version": 1,
        "inputs": args.input,
        "raw_records": len(raw),
        "post_clean_dedup_records": len(curated.accepted),
        "rejected_records": len(curated.rejected),
        "selected_records": len(selected),
        "selected_words": words,
        "cleaning": dict(clean_stats),
        "curation": curated.stats,
        "mixture": mixture,
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
    }
    (out_dir / "v04_corpus_stats.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
