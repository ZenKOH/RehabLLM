from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReadinessThresholds:
    min_selected_docs: int = 3500
    min_robotics_fraction: float = 0.12
    min_neurotechnology_fraction: float = 0.05
    min_tech_fraction: float = 0.20
    min_val_docs: int = 100
    min_test_docs: int = 100
    max_cleaning_removed_fraction: float = 0.30


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_v04_readiness(
    stats: dict[str, Any],
    thresholds: ReadinessThresholds | None = None,
) -> ReadinessResult:
    thresholds = thresholds or ReadinessThresholds()

    selected = int(stats.get("selected_records") or 0)
    categories = stats.get("category_counts") or {}
    splits = stats.get("split_counts") or {}
    cleaning = stats.get("cleaning") or {}

    robotics = int(categories.get("robotics") or 0)
    neuro = int(categories.get("neurotechnology") or 0)
    val_docs = int(splits.get("val") or 0)
    test_docs = int(splits.get("test") or 0)

    robotics_fraction = robotics / max(1, selected)
    neuro_fraction = neuro / max(1, selected)
    tech_fraction = (robotics + neuro) / max(1, selected)

    original_chars = int(cleaning.get("original_chars") or 0)
    removed_chars = int(cleaning.get("removed_chars") or 0)
    removed_fraction = removed_chars / max(1, original_chars)

    checks = {
        "enough_documents": selected >= thresholds.min_selected_docs,
        "robotics_representation": robotics_fraction >= thresholds.min_robotics_fraction,
        "neurotechnology_representation": neuro_fraction
        >= thresholds.min_neurotechnology_fraction,
        "combined_technology_representation": tech_fraction >= thresholds.min_tech_fraction,
        "validation_size": val_docs >= thresholds.min_val_docs,
        "test_size": test_docs >= thresholds.min_test_docs,
        "cleaning_not_excessive": (
            original_chars > 0
            and removed_fraction <= thresholds.max_cleaning_removed_fraction
        ),
    }

    reasons: list[str] = []
    labels = {
        "enough_documents": (
            f"Need at least {thresholds.min_selected_docs:,} selected documents; "
            f"found {selected:,}."
        ),
        "robotics_representation": (
            f"Need robotics fraction >= {thresholds.min_robotics_fraction:.0%}; "
            f"found {robotics_fraction:.1%} ({robotics:,} documents)."
        ),
        "neurotechnology_representation": (
            f"Need neurotechnology fraction >= {thresholds.min_neurotechnology_fraction:.0%}; "
            f"found {neuro_fraction:.1%} ({neuro:,} documents)."
        ),
        "combined_technology_representation": (
            f"Need robotics + neurotechnology >= {thresholds.min_tech_fraction:.0%}; "
            f"found {tech_fraction:.1%}."
        ),
        "validation_size": (
            f"Need at least {thresholds.min_val_docs:,} validation documents; found {val_docs:,}."
        ),
        "test_size": (
            f"Need at least {thresholds.min_test_docs:,} test documents; found {test_docs:,}."
        ),
        "cleaning_not_excessive": (
            "Cleaning statistics are missing or the cleaner removed more than "
            f"{thresholds.max_cleaning_removed_fraction:.0%} of source characters."
        ),
    }
    for key, passed in checks.items():
        if not passed:
            reasons.append(labels[key])

    metrics: dict[str, float | int] = {
        "selected_documents": selected,
        "robotics_documents": robotics,
        "neurotechnology_documents": neuro,
        "robotics_fraction": robotics_fraction,
        "neurotechnology_fraction": neuro_fraction,
        "combined_technology_fraction": tech_fraction,
        "validation_documents": val_docs,
        "test_documents": test_docs,
        "cleaning_removed_fraction": removed_fraction,
    }

    return ReadinessResult(
        ready=all(checks.values()),
        checks=checks,
        metrics=metrics,
        reasons=reasons,
    )
