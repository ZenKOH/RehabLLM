import hashlib
import json
import sqlite3

import pytest

from rehab_minillm.evidence import build_index, search_index


def record(identifier="fixture-1", **overrides):
    return {
        "id": identifier, "title": ["Synthetic robotics research fixture"],
        "source": "Test fixtures, not clinical evidence",
        "source_url": "https://example.org/" + identifier, "license_verified": "CC0",
        "retrieved_at": "2026-11-24T00:00:00+00:00",
        "text": "Robotics research café. A synthetic passage with no clinical conclusions. " * 4,
        **overrides,
    }


def write_records(tmp_path, rows):
    source = tmp_path / "evidence.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return source


def test_retrieved_passages_are_traceable_and_diverse(tmp_path):
    rows = [record(), record("fixture-2")]
    source = write_records(tmp_path, rows)
    index = tmp_path / "index.sqlite3"
    manifest = build_index([source], index, chunk_words=12, overlap_words=3)
    packet = search_index(index, 'robotics OR "café" *', limit=5)
    assert manifest["documents"] == 2
    assert packet["answer_generated"] is False
    assert len(packet["evidence"]) == 2
    for item in packet["evidence"]:
        doc = next(r for r in rows if r["id"] == item["document"]["id"])
        assert item["passage"] == doc["text"][item["start_char"]:item["end_char"]]
        assert item["document"]["text_sha256"] == hashlib.sha256(doc["text"].encode()).hexdigest()
        assert item["document"]["source_url"] == doc["source_url"]
    assert search_index(index, "unfindablexyz")["status"] == "no_matches"
    citations = [e["citation_id"] for e in packet["evidence"]]
    build_index([source], index, chunk_words=12, overlap_words=3)
    assert citations == [e["citation_id"] for e in search_index(
        index, 'robotics OR "café" *', limit=5)["evidence"]]


@pytest.mark.parametrize("change", [
    {"license_verified": None}, {"license_verified": "CC-BY-NC"}, {"title": []},
    {"source_url": "javascript:alert(1)"}, {"retrieved_at": "2026-11-24"},
    {"text": "  "}, {"source_url": "https://user:secret@example.org/paper"}, {"title": ["", ""]},
])
def test_bad_provenance_does_not_replace_existing_index(tmp_path, change):
    source = write_records(tmp_path, [record()])
    index = tmp_path / "index.sqlite3"
    build_index([source], index)
    original = index.read_bytes()
    write_records(tmp_path, [record(), record("bad-record", **change)])
    with pytest.raises(ValueError, match="evidence.jsonl:2"):
        build_index([source], index)
    assert index.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_duplicate_source_ids_are_rejected(tmp_path):
    source = write_records(tmp_path, [record(), record()])
    with pytest.raises(ValueError, match="UNIQUE"):
        build_index([source], tmp_path / "index.sqlite3")


def test_search_cannot_create_missing_database_and_checks_schema(tmp_path):
    index = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        search_index(index, "research")
    assert not index.exists()
    source = write_records(tmp_path, [record()])
    build_index([source], index)
    with sqlite3.connect(index) as connection:
        connection.execute("UPDATE metadata SET value=? WHERE key='manifest'",
                           (json.dumps({"schema_version": 99}),))
    with pytest.raises(ValueError, match="Unsupported"):
        search_index(index, "research")
