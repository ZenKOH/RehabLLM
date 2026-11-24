#!/usr/bin/env python3
"""Create a matched, human-reviewable baseline/candidate comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_reports(baseline: dict, candidate: dict) -> dict:
    for report in (baseline, candidate):
        if report.get("schema_version") != 2:
            raise ValueError("Re-run both evaluations with the schema-v2 evaluator")
        if not report.get("evaluation_config") or not report.get("input_sha256"):
            raise ValueError("Evaluation provenance is missing")
    if baseline["evaluation_config"] != candidate["evaluation_config"]:
        raise ValueError("Evaluation settings differ; use identical prompt format, seed and sampling")
    for key in ("tokenizer", "test", "prompts"):
        left, right = (report["input_sha256"].get(key) for report in (baseline, candidate))
        if not left or left != right:
            raise ValueError(f"Evaluation input differs or is missing: {key}")
    left, right = baseline["generations"], candidate["generations"]
    identities = lambda rows: [(r["id"], r["prompt"], r.get("context", "")) for r in rows]
    if not left or identities(left) != identities(right):
        raise ValueError("Benchmark questions/context/order differ or benchmark is empty")
    return {
        "schema_version": 1,
        "baseline_checkpoint": baseline["input_sha256"]["checkpoint"],
        "candidate_checkpoint": candidate["input_sha256"]["checkpoint"],
        "evaluation_config": baseline["evaluation_config"],
        "test_loss_delta": candidate["test_loss"] - baseline["test_loss"],
        "interpretation": "Loss delta is candidate minus baseline; answer quality requires review.",
        "rubric": ["relevance", "factuality", "uncertainty", "source_fidelity", "safety"],
        "review_status": "pending",
        "comparisons": [{
            "id": a["id"], "prompt": a["prompt"], "context": a.get("context", ""),
            "expected_behaviour": a.get("expected_behaviour"),
            "baseline": {k: a[k] for k in ("output", "stop_reason")},
            "candidate": {k: b[k] for k in ("output", "stop_reason")},
            "human_review": None,
        } for a, b in zip(left, right)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", default="eval/comparison.json")
    args = parser.parse_args()
    report = compare_reports(*[json.loads(Path(p).read_text(encoding="utf-8"))
                               for p in (args.baseline, args.candidate)])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['comparisons'])} matched prompts for human review to {output}")


if __name__ == "__main__":
    main()
