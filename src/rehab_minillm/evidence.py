"""Local lexical evidence retrieval with immutable, source-linked passage identifiers.

This module uses only the Python standard library. It does not generate answers,
fetch URLs, verify medical claims, or infer reuse rights from missing metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
ALLOWED_LICENSES = {"CC0", "CC-BY", "CC-BY-SA"}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document(row: dict) -> dict:
    if not isinstance(row, dict):
        raise TypeError("Evidence record must be an object")
    title = row.get("title")
    if isinstance(title, list) and all(isinstance(t, str) and t.strip() for t in title):
        title = " / ".join(title)
    result = {**row, "title": title}
    for field in ("id", "title", "source", "source_url", "retrieved_at", "text"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"Evidence record needs a non-empty '{field}'")
    url = urlsplit(result["source_url"])
    if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password:
        raise ValueError("source_url must be an HTTP(S) URL without credentials")
    if result.get("license_verified") not in ALLOWED_LICENSES:
        raise ValueError("Evidence record needs an explicit CC0, CC-BY or CC-BY-SA license_verified")
    try:
        retrieved = datetime.fromisoformat(result["retrieved_at"].replace("Z", "+00:00"))
        if retrieved.tzinfo is None:
            raise ValueError("timezone missing")
    except ValueError as exc:
        raise ValueError("retrieved_at must be an ISO-8601 timestamp with timezone") from exc
    return result


def passage_spans(text: str, chunk_words: int, overlap_words: int):
    if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
        raise ValueError("Require chunk_words > overlap_words >= 0")
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), chunk_words - overlap_words):
        end = min(start + chunk_words, len(words))
        yield words[start].start(), words[end - 1].end()
        if end == len(words):
            break


def build_index(
    inputs: list[str | Path], index_path: str | Path,
    *, chunk_words: int = 180, overlap_words: int = 30,
) -> dict:
    """Build atomically; malformed inputs leave an existing index unchanged."""
    if not inputs:
        raise ValueError("At least one input JSONL file is required")
    # Validate even when an input file contains no records.
    list(passage_spans("", chunk_words, overlap_words))
    output = Path(index_path)
    if output.resolve() in {Path(p).resolve() for p in inputs}:
        raise ValueError("Index output must differ from input JSONL paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents (id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL);
            CREATE TABLE passages (
                id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, start_char INTEGER, end_char INTEGER
            );
            CREATE VIRTUAL TABLE passage_search USING fts5(
                citation_id UNINDEXED, doc_id UNINDEXED, title, text,
                tokenize='porter unicode61'
            );
        """)
        documents = passages = 0
        input_hashes = []
        corpus_hash = hashlib.sha256()
        for source in inputs:
            source_hash = hashlib.sha256()
            with Path(source).open("rb") as handle:
                for line_number, raw in enumerate(handle, 1):
                    source_hash.update(raw)
                    if not raw.strip():
                        continue
                    try:
                        row = _document(json.loads(raw))
                        text = row["text"]
                        text_hash = _digest(text)
                        meta = {k: v for k, v in row.items() if k != "text"}
                        meta["text_sha256"] = text_hash
                        meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)
                        connection.execute("INSERT INTO documents VALUES (?, ?)",
                                           (row["id"], meta_json))
                        corpus_hash.update((_digest(meta_json) + "\n").encode())
                        documents += 1
                        for start, end in passage_spans(text, chunk_words, overlap_words):
                            citation = "E" + _digest(json.dumps([
                                row["id"], text_hash, start, end
                            ]))[:24]
                            connection.execute("INSERT INTO passages VALUES (?, ?, ?, ?)",
                                               (citation, row["id"], start, end))
                            connection.execute("INSERT INTO passage_search VALUES (?, ?, ?, ?)",
                                               (citation, row["id"], row["title"], text[start:end]))
                            passages += 1
                    except (ValueError, TypeError, sqlite3.IntegrityError) as exc:
                        raise ValueError(f"{source}:{line_number}: {exc}") from exc
            input_hashes.append({"path": str(source), "sha256": source_hash.hexdigest()})
        if not documents or not passages:
            raise ValueError("No non-empty evidence documents to index")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents": documents, "passages": passages,
            "chunk_words": chunk_words, "overlap_words": overlap_words,
            "corpus_sha256": corpus_hash.hexdigest(), "inputs": input_hashes,
            "retrieval": "SQLite FTS5 BM25; lexical relevance is not evidence quality",
            "rights_policy": "Input license_verified metadata required; no independent recheck",
        }
        connection.execute("INSERT INTO metadata VALUES ('manifest', ?)",
                           (json.dumps(manifest, sort_keys=True),))
        connection.commit()
        connection.close()
        connection = None
        os.replace(temporary, output)
        return manifest
    finally:
        if connection is not None:
            connection.close()
        Path(temporary).unlink(missing_ok=True)


def search_index(
    index_path: str | Path, query: str, *, limit: int = 5, per_document: int = 1,
) -> dict:
    if not 1 <= limit <= 50 or not 1 <= per_document <= 50:
        raise ValueError("limit and per_document must be between 1 and 50")
    if len(query) > 2000:
        raise ValueError("Query must be at most 2,000 characters")
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold(), re.UNICODE)))
    if not terms or len(terms) > 32:
        raise ValueError("Query must contain between 1 and 32 distinct words")
    # Quote each term and bind the expression; input never becomes FTS syntax or SQL.
    expression = " OR ".join('"' + term + '"' for term in terms)
    uri = Path(index_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if row is None:
            raise ValueError("Evidence index manifest is missing")
        manifest = json.loads(row[0])
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported evidence index schema; rebuild the index")
        rows = connection.execute("""
            SELECT s.citation_id, s.doc_id, s.text, p.start_char, p.end_char,
                   d.metadata_json, bm25(passage_search, 0, 0, 3, 1) AS rank
            FROM passage_search AS s
            JOIN passages AS p ON p.id = s.citation_id
            JOIN documents AS d ON d.id = s.doc_id
            WHERE passage_search MATCH ?
            ORDER BY rank, s.citation_id
        """, (expression,))
        evidence = []
        counts: dict[str, int] = {}
        for citation, doc_id, text, start, end, metadata, score in rows:
            if counts.get(doc_id, 0) >= per_document:
                continue
            counts[doc_id] = counts.get(doc_id, 0) + 1
            evidence.append({
                "citation_id": citation, "document": json.loads(metadata),
                "passage": text, "start_char": start, "end_char": end,
                "bm25_score": score,
            })
            if len(evidence) == limit:
                break
        return {
            "schema_version": 1, "query": query,
            "status": "retrieved" if evidence else "no_matches",
            "answer_generated": False,
            "index_corpus_sha256": manifest["corpus_sha256"],
            "index_created_at": manifest["created_at"],
            "evidence": evidence,
        }
    finally:
        connection.close()
