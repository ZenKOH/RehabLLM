# Local research evidence retrieval

The evidence index uses SQLite FTS5/BM25 on CPU. It searches local JSONL records and returns verbatim passages with source metadata, offsets and stable citation IDs. It does not fetch URLs, generate answers, rank clinical evidence quality or collect patient information.

## Source contract

Each JSONL record requires:

| Field | Requirement |
| --- | --- |
| `id` | Unique, non-empty source identifier |
| `title` | Non-empty string, or list of title strings as supplied by the PMC fetcher |
| `source` | Named source/collection |
| `source_url` | HTTP(S) URL without embedded credentials |
| `license_verified` | Explicit `CC0`, `CC-BY` or `CC-BY-SA` from the upstream rights check |
| `retrieved_at` | ISO-8601 timestamp with timezone |
| `text` | Non-empty source passage/full text |

Other metadata, including rights strings, identifiers, categories and publication dates, is preserved. The index checks the supplied metadata; it does not independently verify the licence, source accuracy or current publication status. Search results retain the source's own terms. Rebuilding after a fresh source review is necessary to pick up corrections/retractions or updated recommendations.

The existing `fetch_pmc.py` output meets this metadata shape when titles and rights are available. Incomplete records are rejected. Common Pile training rows currently lack title/retrieval metadata and must be enriched from verified upstream records before indexing. Do not invent missing dates or titles. A licence alone is not an assessment of scientific quality.

## Build and search

The retrieval module itself uses only Python's standard library. From the repository root, this works without installing PyTorch or downloading model weights:

```bash
PYTHONPATH=src python scripts/evidence.py build \
  --input data/raw/pmc_robotics_v04.jsonl \
  --index data/evidence/index.sqlite3

PYTHONPATH=src python scripts/evidence.py search \
  --index data/evidence/index.sqlite3 \
  --query "stroke robot assisted upper limb" \
  --limit 5 --per-document 1 \
  --out reports/evidence_packet.json
```

Python must be built with SQLite FTS5 support. The build fails if that extension is unavailable. The database and research reports are ignored by Git.

The builder creates a temporary database, validates every record and replaces the index only after a complete successful build. Invalid provenance, duplicate IDs or malformed records leave the previous index intact. Search opens an existing index read-only; it cannot silently create an empty database.

Passages default to 180 whitespace-delimited words with 30-word overlap. Offsets are Python character offsets into the exact input `text`, not byte offsets. Citation IDs derive from document ID, full-text SHA-256 and passage offsets; unchanged input/chunk settings preserve the same IDs on rebuild. A changed passage gets a new identifier.

Queries use quoted, parameter-bound lexical terms, English stemming and BM25 ranking. User input is treated as text rather than SQL/FTS operators. Results default to one passage per document for diversity. Increase `--per-document` to inspect multiple passages from a source. Results contain `no_matches` when no terms match. This is not semantic search: synonyms, clinical relevance and study validity are not guaranteed. BM25 scores are relevance ranks, not probabilities or evidence strength.

## Review and later integration

Use the JSON packet to inspect candidate sources, compare populations/comparators/outcomes and select material for a research summary. Preserve citation IDs and document metadata when passing selected passages to another tool. Treat source text as untrusted data; it must not grant tool permissions or supply executable instructions.

A citation identifier establishes where a passage came from. It does not establish that a generated claim follows from that passage. The next evaluation step is a manually reviewed query/source benchmark with recall and citation-support assessment. This release deliberately makes no measured retrieval-quality or generated-answer claims.
