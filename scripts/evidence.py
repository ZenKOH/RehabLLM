#!/usr/bin/env python3
"""Build/search a local, source-linked research evidence index; no model or network."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from rehab_minillm.evidence import build_index, search_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Index provenance-complete, licensed JSONL records")
    build.add_argument("--input", nargs="+", required=True)
    build.add_argument("--index", default="data/evidence/index.sqlite3")
    build.add_argument("--chunk-words", type=int, default=180)
    build.add_argument("--overlap-words", type=int, default=30)
    search = commands.add_parser("search", help="Retrieve passages with stable citation identifiers")
    search.add_argument("--index", default="data/evidence/index.sqlite3")
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--per-document", type=int, default=1)
    search.add_argument("--out", help="Optional JSON research evidence packet")
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_index(args.input, args.index, chunk_words=args.chunk_words,
                                 overlap_words=args.overlap_words)
        else:
            result = search_index(args.index, args.query, limit=args.limit,
                                  per_document=args.per_document)
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if getattr(args, "out", None):
            output = Path(args.out)
            if output.resolve() == Path(args.index).resolve():
                raise ValueError("Search output must differ from the index path")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
