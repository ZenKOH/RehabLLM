#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rehab_minillm.cleaning import clean_document, detect_contamination
from rehab_minillm.domain import classify_record


def _records_from_text(path: Path) -> Iterable[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    for index, chunk in enumerate(raw.split("<eos>")):
        text = chunk.strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        yield {
            "id": f"{path.stem}-{index}-{digest}",
            "source": str(path),
            "text": text,
        }


def read_records(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        else:
            yield from _records_from_text(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RehabLLM corpus contamination and mix")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-json", default="reports/v04_corpus_autopsy.json")
    parser.add_argument("--out-md", default="reports/v04_corpus_autopsy.md")
    parser.add_argument("--sample-limit", type=int, default=5)
    args = parser.parse_args()

    inputs = [Path(value) for value in args.input]
    marker_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    documents = 0
    contaminated_docs = 0
    original_chars = 0
    cleaned_chars = 0

    for record in read_records(inputs):
        documents += 1
        text = str(record.get("text") or "")
        original_chars += len(text)
        category = classify_record(record) or "unclassified"
        category_counts[category] += 1
        source_counts[str(record.get("source") or "unknown")] += 1

        markers = detect_contamination(text)
        if markers:
            contaminated_docs += 1
        marker_counts.update(markers)

        for marker in markers:
            if len(examples[marker]) < args.sample_limit:
                examples[marker].append(
                    {
                        "id": str(record.get("id") or ""),
                        "snippet": text[:500].replace("\n", " "),
                    }
                )

        cleaned = clean_document(text)
        cleaned_chars += cleaned.cleaned_chars

    report = {
        "schema_version": 1,
        "inputs": [str(path) for path in inputs],
        "documents": documents,
        "contaminated_documents": contaminated_docs,
        "contaminated_document_fraction": contaminated_docs / max(1, documents),
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "marker_counts": dict(marker_counts.most_common()),
        "original_chars": original_chars,
        "cleaned_chars_estimate": cleaned_chars,
        "estimated_removed_fraction": (original_chars - cleaned_chars) / max(1, original_chars),
        "examples": examples,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RehabLLM v0.4 corpus autopsy",
        "",
        f"- Documents: {documents:,}",
        f"- Documents with detected contamination: {contaminated_docs:,} ({report['contaminated_document_fraction']:.1%})",
        f"- Estimated characters removed by v0.4 cleaner: {report['estimated_removed_fraction']:.1%}",
        "",
        "## Domain mix",
        "",
    ]
    for key, value in category_counts.most_common():
        lines.append(f"- {key}: {value:,}")

    lines.extend(["", "## Contamination markers", ""])
    if marker_counts:
        for key, value in marker_counts.most_common():
            lines.append(f"- {key}: {value:,}")
    else:
        lines.append("- No configured contamination markers detected.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report is descriptive. A high marker count identifies training-text noise, not scientific invalidity. v0.4 removes publishing boilerplate and markup while preserving substantive rehabilitation content.",
            "",
        ]
    )
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
