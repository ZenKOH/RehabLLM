from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
PRIME_61 = (1 << 61) - 1


@dataclass(frozen=True)
class QualityConfig:
    min_words: int = 250
    max_words: int = 30_000
    min_alpha_word_fraction: float = 0.70
    max_symbol_to_word_ratio: float = 0.20
    min_median_word_length: float = 3.0
    max_median_word_length: float = 12.0
    max_duplicate_line_fraction: float = 0.35


@dataclass(frozen=True)
class DedupConfig:
    shingle_size: int = 5
    num_perm: int = 64
    bands: int = 16
    jaccard_threshold: float = 0.82
    seed: int = 42

    def validate(self) -> None:
        if self.num_perm % self.bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        if not 0.0 < self.jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold must be in (0, 1]")


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.90
    val: float = 0.05
    test: float = 0.05
    seed: int = 42

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("train + val + test must equal 1")


@dataclass(frozen=True)
class QualityMetrics:
    words: int
    chars: int
    alpha_word_fraction: float
    symbol_to_word_ratio: float
    median_word_length: float
    duplicate_line_fraction: float


@dataclass
class CurationResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    stats: dict[str, Any]


def normalise_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def quality_metrics(text: str) -> QualityMetrics:
    words = WORD_RE.findall(text)
    word_count = len(words)
    alpha_words = sum(any(char.isalpha() for char in word) for word in words)
    symbol_count = sum(not (char.isalnum() or char.isspace()) for char in text)
    lengths = sorted(len(word) for word in words if any(char.isalpha() for char in word))
    if lengths:
        mid = len(lengths) // 2
        median = float(lengths[mid]) if len(lengths) % 2 else (lengths[mid - 1] + lengths[mid]) / 2.0
    else:
        median = 0.0

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if lines:
        counts = Counter(lines)
        duplicate_lines = sum(count for count in counts.values() if count > 1)
        duplicate_fraction = duplicate_lines / len(lines)
    else:
        duplicate_fraction = 1.0

    return QualityMetrics(
        words=word_count,
        chars=len(text),
        alpha_word_fraction=alpha_words / max(1, word_count),
        symbol_to_word_ratio=symbol_count / max(1, word_count),
        median_word_length=median,
        duplicate_line_fraction=duplicate_fraction,
    )


def quality_reasons(metrics: QualityMetrics, config: QualityConfig) -> list[str]:
    reasons: list[str] = []
    if metrics.words < config.min_words:
        reasons.append("too_short")
    if metrics.words > config.max_words:
        reasons.append("too_long")
    if metrics.alpha_word_fraction < config.min_alpha_word_fraction:
        reasons.append("low_alpha_word_fraction")
    if metrics.symbol_to_word_ratio > config.max_symbol_to_word_ratio:
        reasons.append("high_symbol_ratio")
    if metrics.median_word_length < config.min_median_word_length:
        reasons.append("median_word_too_short")
    if metrics.median_word_length > config.max_median_word_length:
        reasons.append("median_word_too_long")
    if metrics.duplicate_line_fraction > config.max_duplicate_line_fraction:
        reasons.append("excess_duplicate_lines")
    return reasons


def exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def word_shingles(text: str, n: int = 5) -> set[int]:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    if len(tokens) < n:
        return {_stable_u64(" ".join(tokens))} if tokens else set()
    return {_stable_u64(" ".join(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


def _permutations(num_perm: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(1, PRIME_61), rng.randrange(0, PRIME_61)) for _ in range(num_perm)]


def minhash_signature(shingles: set[int], config: DedupConfig) -> tuple[int, ...]:
    config.validate()
    if not shingles:
        return tuple([PRIME_61] * config.num_perm)
    signature: list[int] = []
    for a, b in _permutations(config.num_perm, config.seed):
        signature.append(min(((a * (value % PRIME_61) + b) % PRIME_61) for value in shingles))
    return tuple(signature)


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / max(1, union)


class NearDuplicateIndex:
    def __init__(self, config: DedupConfig) -> None:
        config.validate()
        self.config = config
        self.rows = config.num_perm // config.bands
        self.buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
        self.shingles: list[set[int]] = []

    def _band_keys(self, signature: tuple[int, ...]) -> Iterator[tuple[int, str]]:
        for band in range(self.config.bands):
            start = band * self.rows
            chunk = signature[start : start + self.rows]
            payload = ",".join(map(str, chunk))
            yield band, hashlib.blake2b(payload.encode("ascii"), digest_size=8).hexdigest()

    def find_duplicate(self, shingles: set[int]) -> tuple[int, float] | None:
        signature = minhash_signature(shingles, self.config)
        candidates: set[int] = set()
        for key in self._band_keys(signature):
            candidates.update(self.buckets.get(key, ()))
        best: tuple[int, float] | None = None
        for index in candidates:
            score = jaccard(shingles, self.shingles[index])
            if score >= self.config.jaccard_threshold and (best is None or score > best[1]):
                best = (index, score)
        return best

    def add(self, shingles: set[int]) -> int:
        index = len(self.shingles)
        signature = minhash_signature(shingles, self.config)
        self.shingles.append(shingles)
        for key in self._band_keys(signature):
            self.buckets[key].append(index)
        return index


def deterministic_split(identifier: str, config: SplitConfig) -> str:
    config.validate()
    digest = hashlib.sha256(f"{config.seed}:{identifier}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < config.train:
        return "train"
    if value < config.train + config.val:
        return "val"
    return "test"


def curate_records(
    records: Iterable[dict[str, Any]],
    quality_config: QualityConfig | None = None,
    dedup_config: DedupConfig | None = None,
    split_config: SplitConfig | None = None,
) -> CurationResult:
    quality_config = quality_config or QualityConfig()
    dedup_config = dedup_config or DedupConfig()
    split_config = split_config or SplitConfig()
    exact_seen: dict[str, str] = {}
    near_index = NearDuplicateIndex(dedup_config)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for raw in records:
        identifier = str(raw.get("id") or raw.get("pmcid") or "").strip()
        text = normalise_text(str(raw.get("text") or ""))
        metrics = quality_metrics(text)
        reasons = quality_reasons(metrics, quality_config)
        if not identifier:
            reasons.append("missing_identifier")
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            rejected.append({"id": identifier, "reasons": reasons, "metrics": asdict(metrics)})
            continue

        content_hash = exact_hash(text)
        if content_hash in exact_seen:
            reason_counts["exact_duplicate"] += 1
            rejected.append(
                {
                    "id": identifier,
                    "reasons": ["exact_duplicate"],
                    "duplicate_of": exact_seen[content_hash],
                    "metrics": asdict(metrics),
                }
            )
            continue

        shingles = word_shingles(text, dedup_config.shingle_size)
        duplicate = near_index.find_duplicate(shingles)
        if duplicate is not None:
            duplicate_index, score = duplicate
            reason_counts["near_duplicate"] += 1
            rejected.append(
                {
                    "id": identifier,
                    "reasons": ["near_duplicate"],
                    "duplicate_of": accepted[duplicate_index]["id"],
                    "jaccard": round(score, 6),
                    "metrics": asdict(metrics),
                }
            )
            continue

        record = dict(raw)
        record["text"] = text
        record["content_sha256"] = content_hash
        record["metrics"] = asdict(metrics)
        record["split"] = deterministic_split(identifier, split_config)
        exact_seen[content_hash] = identifier
        near_index.add(shingles)
        accepted.append(record)

    split_counts = Counter(record["split"] for record in accepted)
    topic_counts: Counter[str] = Counter()
    licence_counts: Counter[str] = Counter()
    for record in accepted:
        topic_counts.update(record.get("topics") or [])
        licences = record.get("license_verified") or record.get("license") or []
        if isinstance(licences, str):
            licences = [licences]
        licence_counts.update(licences)

    stats = {
        "input_records": len(accepted) + len(rejected),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "reject_reasons": dict(sorted(reason_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "topics": dict(topic_counts.most_common()),
        "licences": dict(licence_counts.most_common()),
        "accepted_words": sum(record["metrics"]["words"] for record in accepted),
        "accepted_chars": sum(record["metrics"]["chars"] for record in accepted),
    }
    return CurationResult(accepted=accepted, rejected=rejected, stats=stats)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
