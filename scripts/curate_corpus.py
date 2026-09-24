#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rehab_minillm.curation import (
    DedupConfig,
    QualityConfig,
    SplitConfig,
    curate_records,
    read_jsonl,
)


def load_config(path: str | Path) -> tuple[QualityConfig, DedupConfig, SplitConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return (
        QualityConfig(**raw.get("quality", {})),
        DedupConfig(**raw.get("dedup", {})),
        SplitConfig(**raw.get("splits", {})),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate and split the RehabLLM PMC corpus")
    parser.add_argument("--input", default="data/raw/pmc_rehab.jsonl")
    parser.add_argument("--config", default="configs/curation.yaml")
    parser.add_argument("--out-dir", default="data/curated")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_cfg, dedup_cfg, split_cfg = load_config(args.config)
    result = curate_records(read_jsonl(input_path), quality_cfg, dedup_cfg, split_cfg)

    accepted_path = out_dir / "articles.jsonl"
    rejected_path = out_dir / "rejected.jsonl"
    with accepted_path.open("w", encoding="utf-8") as handle:
        for record in result.accepted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with rejected_path.open("w", encoding="utf-8") as handle:
        for record in result.rejected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    split_files = {split: (out_dir / f"{split}.txt").open("w", encoding="utf-8") for split in ("train", "val", "test")}
    try:
        for record in result.accepted:
            split_files[record["split"]].write(record["text"] + "\n<eos>\n")
    finally:
        for handle in split_files.values():
            handle.close()

    stats = dict(result.stats)
    stats["generated_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "config": {
            "quality": asdict(quality_cfg),
            "dedup": asdict(dedup_cfg),
            "splits": asdict(split_cfg),
        },
        "outputs": {
            "articles": str(accepted_path),
            "rejected": str(rejected_path),
            "stats": str(out_dir / "stats.json"),
            "train": str(out_dir / "train.txt"),
            "val": str(out_dir / "val.txt"),
            "test": str(out_dir / "test.txt"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
