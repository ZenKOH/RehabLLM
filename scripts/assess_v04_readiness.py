#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rehab_minillm.readiness import ReadinessThresholds, assess_v04_readiness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess whether a prepared v0.4 corpus is ready for GPU training"
    )
    parser.add_argument("--stats", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-docs", type=int, default=3500)
    parser.add_argument("--min-robotics-fraction", type=float, default=0.12)
    parser.add_argument("--min-neuro-fraction", type=float, default=0.05)
    parser.add_argument("--min-tech-fraction", type=float, default=0.20)
    args = parser.parse_args()

    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    thresholds = ReadinessThresholds(
        min_selected_docs=args.min_docs,
        min_robotics_fraction=args.min_robotics_fraction,
        min_neurotechnology_fraction=args.min_neuro_fraction,
        min_tech_fraction=args.min_tech_fraction,
    )
    result = assess_v04_readiness(stats, thresholds)
    payload = result.to_dict()

    print("=" * 72)
    print("RehabLLM v0.4 GPU READINESS:", "GO" if result.ready else "HOLD")
    print("=" * 72)
    for name, passed in result.checks.items():
        print(("PASS" if passed else "FAIL"), name)
    print()
    print(json.dumps(payload["metrics"], indent=2))
    if result.reasons:
        print("\nReasons to hold:")
        for reason in result.reasons:
            print("-", reason)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.strict and not result.ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
