from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

REHAB_TERMS = (
    "rehabilitation",
    "neurorehabilitation",
    "physical therapy",
    "physiotherapy",
    "occupational therapy",
    "motor recovery",
    "functional recovery",
    "gait training",
    "balance training",
    "activities of daily living",
)

ROBOTICS_TERMS = (
    "rehabilitation robot",
    "robot-assisted",
    "robot assisted",
    "robotic rehabilitation",
    "robot-assisted therapy",
    "robotic therapy",
    "exoskeleton",
    "end-effector robot",
    "end effector robot",
    "powered gait",
    "wearable robot",
    "robotic gait",
    "robotic arm",
    "robotic hand",
)

NEUROTECH_TERMS = (
    "functional electrical stimulation",
    "neuromuscular electrical stimulation",
    "brain-computer interface",
    "brain computer interface",
    "electromyography",
    "surface emg",
    "motor imagery",
    "neuromodulation",
)

ASSISTIVE_TERMS = (
    "assistive technology",
    "prosthesis",
    "prosthetic",
    "orthosis",
    "orthotic",
    "wheelchair",
    "augmentative communication",
)

DEFAULT_V04_MIX = {
    "core_rehab": 0.45,
    "robotics": 0.30,
    "neurotechnology": 0.15,
    "assistive": 0.10,
}


def classify_document(text: str) -> str | None:
    lower = text.lower()
    has_rehab = any(term in lower for term in REHAB_TERMS)
    if has_rehab and any(term in lower for term in ROBOTICS_TERMS):
        return "robotics"
    if has_rehab and any(term in lower for term in NEUROTECH_TERMS):
        return "neurotechnology"
    if has_rehab and any(term in lower for term in ASSISTIVE_TERMS):
        return "assistive"
    if has_rehab:
        return "core_rehab"
    return None


def classify_record(record: dict[str, Any]) -> str | None:
    explicit = str(record.get("category") or "").strip()
    if explicit in DEFAULT_V04_MIX:
        return explicit

    tags = {str(tag).lower() for tag in (record.get("tags") or [])}
    topics = {str(topic).lower() for topic in (record.get("topics") or [])}
    joined = " ".join(tags | topics)
    if any(token in joined for token in ("robot", "exoskeleton", "gait_robot")):
        return "robotics"
    if any(token in joined for token in ("fes", "bci", "neurotechnology", "emg")):
        return "neurotechnology"
    if any(token in joined for token in ("assistive", "prosthe", "orthot", "wheelchair")):
        return "assistive"
    return classify_document(str(record.get("text") or ""))


def target_counts(total: int, mix: dict[str, float] | None = None) -> dict[str, int]:
    mix = mix or DEFAULT_V04_MIX
    if total <= 0:
        raise ValueError("total must be positive")
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError("mixture weights must sum to 1")

    counts = {key: int(total * weight) for key, weight in mix.items()}
    counts["core_rehab"] += total - sum(counts.values())
    return counts


def count_categories(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        category = classify_record(record)
        counter[category or "unclassified"] += 1
    return dict(counter)
