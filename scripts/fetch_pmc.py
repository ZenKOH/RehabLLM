#!/usr/bin/env python3
"""Discover and fetch permissively licensed PMC rehabilitation articles.

Uses NCBI E-Utilities for discovery and the official PMC BioC API for article text.
The default query plan is restricted to CC0, CC BY and CC BY-SA records.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from rehab_minillm.corpus import iter_query_plan

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
TOOL_NAME = "rehab_minillm"


def get_json(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RehabMiniLLM/0.1 (research corpus builder)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def esearch(query: str, retmax: int, email: str, api_key: str | None) -> list[str]:
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
        "tool": TOOL_NAME,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{ESEARCH}?{urllib.parse.urlencode(params)}"
    payload = get_json(url)
    return payload.get("esearchresult", {}).get("idlist", [])


def flatten_bioc(payload: Any) -> str:
    documents = payload if isinstance(payload, list) else [payload]
    chunks: list[str] = []
    for collection in documents:
        docs = collection.get("documents", []) if isinstance(collection, dict) else []
        for document in docs:
            for passage in document.get("passages", []):
                text = (passage.get("text") or "").strip()
                if not text:
                    continue
                section = str(passage.get("infons", {}).get("section_type", "")).lower()
                # References are noisy and add citation strings rather than scientific prose.
                if section in {"ref", "references", "bibliography"}:
                    continue
                chunks.append(text)
    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/pmc_rehab.jsonl")
    parser.add_argument("--text-out", default="data/raw/pmc_rehab.txt")
    parser.add_argument("--retmax-per-query", type=int, default=250)
    parser.add_argument("--max-articles", type=int, default=2500)
    parser.add_argument("--sleep", type=float, default=None)
    args = parser.parse_args()

    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise SystemExit("Set NCBI_EMAIL to a valid contact email before using NCBI E-Utilities.")
    api_key = os.environ.get("NCBI_API_KEY")
    sleep_seconds = args.sleep if args.sleep is not None else (0.12 if api_key else 0.36)

    # pmcid -> accumulated provenance from overlapping topic searches
    discovered: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"topics": set(), "tags": set(), "licences": set(), "queries": set()}
    )

    for topic, licence, query in iter_query_plan():
        print(f"discovering topic={topic.name} licence={licence}")
        ids = esearch(query, args.retmax_per_query, email, api_key)
        for uid in ids:
            pmcid = f"PMC{uid}"
            record = discovered[pmcid]
            record["topics"].add(topic.name)
            record["tags"].update(topic.tags)
            record["licences"].add(licence)
            record["queries"].add(query)
        time.sleep(sleep_seconds)

    pmcids = sorted(discovered)[: args.max_articles]
    out_path = Path(args.out)
    text_path = Path(args.text_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as jout, text_path.open("w", encoding="utf-8") as tout:
        for index, pmcid in enumerate(pmcids, start=1):
            try:
                payload = get_json(BIOC.format(pmcid=pmcid))
                text = flatten_bioc(payload)
                if len(text) < 500:
                    continue
                meta = discovered[pmcid]
                row = {
                    "id": pmcid,
                    "source": "PMC Open Access Subset via BioC API",
                    "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                    "licence": sorted(meta["licences"]),
                    "topics": sorted(meta["topics"]),
                    "tags": sorted(meta["tags"]),
                    "retrieval_method": "NCBI E-Utilities + PMC BioC API",
                    "text": text,
                }
                jout.write(json.dumps(row, ensure_ascii=False) + "\n")
                tout.write(text.replace("\x00", " ") + "\n<eos>\n")
                written += 1
                print(f"[{index}/{len(pmcids)}] wrote {pmcid}")
            except Exception as exc:  # network/API errors should not destroy a long run
                print(f"warning: {pmcid}: {exc}")
            time.sleep(sleep_seconds)

    print(f"wrote {written} articles to {out_path}")
    print(f"wrote plain-text training corpus to {text_path}")


if __name__ == "__main__":
    main()
