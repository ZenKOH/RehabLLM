#!/usr/bin/env python3
"""Discover and fetch permissively licensed PMC rehabilitation articles.

Uses NCBI E-Utilities for discovery, PMC OAI-PMH for article-level rights metadata,
and the official PMC BioC API for article text. The default allow-list is CC0,
CC BY and CC BY-SA.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rehab_minillm.corpus import iter_query_plan

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
OAI = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
TOOL_NAME = "rehab_minillm"
ALLOWED_LICENSES = {"CC0", "CC-BY", "CC-BY-SA"}
EXCLUDED_SECTIONS = {
    "ref",
    "references",
    "bibliography",
    "ack",
    "acknowledgements",
    "acknowledgments",
    "author-contributions",
    "funding",
    "conflict-of-interest",
    "competing-interests",
    "supplementary-material",
}


class PoliteClient:
    """Serial HTTP client with a minimum interval between requests."""

    def __init__(self, min_interval: float = 0.36, timeout: int = 60) -> None:
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def get_bytes(self, url: str, accept: str) -> bytes:
        self._wait()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "RehabMiniLLM/0.2 (research corpus builder)",
                "Accept": accept,
                "Accept-Encoding": "gzip, deflate",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read()
        self._last_request = time.monotonic()
        return payload

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_bytes(url, "application/json").decode("utf-8"))

    def get_xml(self, url: str) -> ET.Element:
        return ET.fromstring(self.get_bytes(url, "application/xml,text/xml"))


def esearch(client: PoliteClient, query: str, retmax: int, email: str, api_key: str | None) -> list[str]:
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
    payload = client.get_json(url)
    return payload.get("esearchresult", {}).get("idlist", [])


def _normalise_license(rights: list[str]) -> str | None:
    joined = " ".join(rights).lower()
    if "creativecommons.org/publicdomain/zero" in joined or re.search(r"\bcc0\b", joined):
        return "CC0"
    if "creativecommons.org/licenses/by-sa/" in joined or "cc by-sa" in joined:
        return "CC-BY-SA"
    if (
        "creativecommons.org/licenses/by/" in joined or re.search(r"\bcc by\b", joined)
    ) and "by-nc" not in joined and "by-nd" not in joined:
        return "CC-BY"
    return None


def oai_metadata(client: PoliteClient, pmcid: str) -> dict[str, Any]:
    numeric = pmcid.removeprefix("PMC")
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{numeric}",
        "metadataPrefix": "oai_dc",
    }
    root = client.get_xml(f"{OAI}?{urllib.parse.urlencode(params)}")
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    metadata = root.find(".//oai:metadata", ns)
    if metadata is None:
        return {"rights": [], "language": [], "title": [], "identifier": []}
    return {
        "rights": [node.text.strip() for node in metadata.findall(".//dc:rights", ns) if node.text],
        "language": [node.text.strip() for node in metadata.findall(".//dc:language", ns) if node.text],
        "title": [node.text.strip() for node in metadata.findall(".//dc:title", ns) if node.text],
        "identifier": [node.text.strip() for node in metadata.findall(".//dc:identifier", ns) if node.text],
    }


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
                section = str(passage.get("infons", {}).get("section_type", "")).strip().lower()
                if section in EXCLUDED_SECTIONS:
                    continue
                chunks.append(text)
    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/pmc_rehab.jsonl")
    parser.add_argument("--retmax-per-query", type=int, default=250)
    parser.add_argument("--max-articles", type=int, default=2500)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.36,
        help="Minimum seconds between requests. PMC OAI-PMH documents a 3 requests/second limit.",
    )
    parser.add_argument("--skip-rights-verification", action="store_true")
    args = parser.parse_args()

    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise SystemExit("Set NCBI_EMAIL to a valid contact email before using NCBI E-Utilities.")
    api_key = os.environ.get("NCBI_API_KEY")
    client = PoliteClient(min_interval=max(args.request_interval, 0.34))

    discovered: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"topics": set(), "tags": set(), "licences": set(), "queries": set()}
    )
    for topic, licence, query in iter_query_plan():
        print(f"discovering topic={topic.name} licence={licence}")
        ids = esearch(client, query, args.retmax_per_query, email, api_key)
        for uid in ids:
            pmcid = f"PMC{uid}"
            record = discovered[pmcid]
            record["topics"].add(topic.name)
            record["tags"].update(topic.tags)
            record["licences"].add(licence)
            record["queries"].add(query)

    pmcids = sorted(discovered)[: args.max_articles]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_rights = 0
    with out_path.open("w", encoding="utf-8") as output:
        for index, pmcid in enumerate(pmcids, start=1):
            try:
                oai = oai_metadata(client, pmcid) if not args.skip_rights_verification else {"rights": []}
                verified = _normalise_license(oai.get("rights", []))
                if not args.skip_rights_verification and verified not in ALLOWED_LICENSES:
                    skipped_rights += 1
                    print(f"[{index}/{len(pmcids)}] skip {pmcid}: rights={oai.get('rights', [])}")
                    continue

                payload = client.get_json(BIOC.format(pmcid=pmcid))
                text = flatten_bioc(payload)
                if len(text) < 500:
                    continue
                meta = discovered[pmcid]
                row = {
                    "id": pmcid,
                    "source": "PMC Open Access Subset",
                    "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                    "license_search_filter": sorted(meta["licences"]),
                    "license_verified": verified,
                    "rights": oai.get("rights", []),
                    "language": oai.get("language", []),
                    "title": oai.get("title", []),
                    "identifiers": oai.get("identifier", []),
                    "topics": sorted(meta["topics"]),
                    "tags": sorted(meta["tags"]),
                    "queries": sorted(meta["queries"]),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "retrieval_method": "NCBI E-Utilities discovery + PMC OAI-PMH rights + PMC BioC full text",
                    "text": text,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                print(f"[{index}/{len(pmcids)}] wrote {pmcid} ({verified or 'unverified'})")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ET.ParseError) as exc:
                print(f"warning: {pmcid}: {exc}")

    print(f"wrote {written} articles to {out_path}")
    if skipped_rights:
        print(f"skipped {skipped_rights} articles because article-level rights did not match the allow-list")


if __name__ == "__main__":
    main()
