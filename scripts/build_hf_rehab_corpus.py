#!/usr/bin/env python3
"""Build a rehabilitation-focused corpus from common-pile/pubmed_filtered."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rehab_minillm.curation import (
    QualityConfig,
    SplitConfig,
    deterministic_split,
    normalise_text,
    quality_metrics,
    quality_reasons,
)

DATASET_ID = "common-pile/pubmed_filtered"

REHAB_TERMS = (
    "rehabilitation",
    "neurorehabilitation",
    "physical therapy",
    "physiotherapy",
    "occupational therapy",
    "motor recovery",
    "functional recovery",
    "gait training",
    "balance training",
    "activities of daily living",
)

ROBOTICS_TERMS = (
    "rehabilitation robot",
    "robot-assisted",
    "robotic rehabilitation",
    "robot-assisted therapy",
    "robotic therapy",
    "exoskeleton",
    "end-effector robot",
    "end effector robot",
    "powered gait",
)

NEUROTECH_TERMS = (
    "functional electrical stimulation",
    "neuromuscular electrical stimulation",
    "brain-computer interface",
    "brain computer interface",
    "electromyography",
    "surface emg",
    "motor imagery",
)

ASSISTIVE_TERMS = (
    "assistive technology",
    "prosthesis",
    "prosthetic",
    "orthosis",
    "orthotic",
    "wheelchair",
    "augmentative communication",
)

DEFAULT_QUOTAS = {
    "robotics": 3000,
    "core_rehab": 5000,
    "neurotechnology": 2000,
    "assistive": 2000,
}


def normalise_license(metadata: dict[str, Any]) -> str | None:
    value = str(metadata.get("license") or "").lower()
    if "publicdomain/zero" in value or "cc0" in value:
        return "CC0"
    if "creativecommons.org/licenses/by-sa/" in value or "cc by-sa" in value:
        return "CC-BY-SA"
    if (
        "creativecommons.org/licenses/by/" in value or "cc by" in value
    ) and "by-nc" not in value and "by-nd" not in value:
        return "CC-BY"
    return None


def classify_document(text: str) -> str | None:
    lower = text.lower()
    has_rehab = any(term in lower for term in REHAB_TERMS)
    if has_rehab and any(term in lower for term in ROBOTICS_TERMS):
        return "robotics"
    if has_rehab and any(term in lower for term in NEUROTECH_TERMS):
        return "neurotechnology"
    if has_rehab and any(term in lower for term in ASSISTIVE_TERMS):
        return "assistive"
    if has_rehab:
        return "core_rehab"
    return None


def quotas_from_target(target_docs: int) -> dict[str, int]:
    base_total = sum(DEFAULT_QUOTAS.values())
    quotas = {
        key: max(1, round(target_docs * value / base_total))
        for key, value in DEFAULT_QUOTAS.items()
    }
    difference = target_docs - sum(quotas.values())
    quotas["core_rehab"] += difference
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream and select a RehabLLM corpus from Common Pile PubMed"
    )
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--target-docs", type=int, default=12000)
    parser.add_argument("--max-scanned", type=int, default=4_800_000)
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/curated")
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quotas = quotas_from_target(args.target_docs)
    counts: Counter[str] = Counter()
    licences: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    total_words = 0
    scanned = 0

    quality_config = QualityConfig(
        min_words=250,
        max_words=30000,
        min_alpha_word_fraction=0.70,
        max_symbol_to_word_ratio=0.25,
        min_median_word_length=3.0,
        max_median_word_length=12.0,
        max_duplicate_line_fraction=0.50,
    )
    split_config = SplitConfig(train=0.90, val=0.05, test=0.05, seed=args.seed)

    dataset = load_dataset(args.dataset, split="train", streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    split_files = {
        split: (out_dir / f"{split}.txt").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    records_path = out_dir / "articles.jsonl"

    try:
        with records_path.open("w", encoding="utf-8") as records:
            for row in dataset:
                scanned += 1
                if scanned > args.max_scanned:
                    break

                identifier = str(row.get("id") or "").strip()
                text = normalise_text(str(row.get("text") or ""))
                metadata = row.get("metadata") or {}
                licence = normalise_license(metadata)
                if not identifier or not licence:
                    reject_reasons["missing_id_or_unapproved_license"] += 1
                    continue

                category = classify_document(text)
                if category is None:
                    reject_reasons["not_rehabilitation"] += 1
                    continue
                if counts[category] >= quotas[category]:
                    continue

                metrics = quality_metrics(text)
                reasons = quality_reasons(metrics, quality_config)
                if reasons:
                    reject_reasons.update(reasons)
                    continue

                split = deterministic_split(identifier, split_config)
                record = {
                    "id": identifier,
                    "source": args.dataset,
                    "source_url": metadata.get("url"),
                    "license_verified": licence,
                    "category": category,
                    "split": split,
                    "words": metrics.words,
                    "text": text,
                }
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                split_files[split].write(text + "\n<eos>\n")
                counts[category] += 1
                licences[licence] += 1
                split_counts[split] += 1
                total_words += metrics.words

                if sum(counts.values()) >= args.target_docs:
                    break
    finally:
        for handle in split_files.values():
            handle.close()

    stats = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "target_docs": args.target_docs,
        "accepted_docs": sum(counts.values()),
        "scanned_docs": scanned,
        "category_quotas": quotas,
        "category_counts": dict(counts),
        "split_counts": dict(split_counts),
        "license_counts": dict(licences),
        "rejected": dict(reject_reasons),
        "accepted_words": total_words,
        "note": (
            "Upstream common-pile/pubmed_filtered is already filtered/deduplicated; "
            "this pass adds domain, licence and quality controls."
        ),
    }
    (out_dir / "gpu_corpus_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    if stats["accepted_docs"] < args.target_docs:
        raise SystemExit(
            f"Only collected {stats['accepted_docs']} of {args.target_docs} requested documents "
            f"after scanning {scanned}. Increase --max-scanned or relax quotas."
        )


if __name__ == "__main__":
    main()
