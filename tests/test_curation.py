from rehab_minillm.curation import (
    DedupConfig,
    QualityConfig,
    SplitConfig,
    curate_records,
    deterministic_split,
    normalise_text,
)


def test_normalise_text_collapses_whitespace():
    assert normalise_text("A   rehab\r\n\r\n\r\nrobot  \n") == "A rehab\n\nrobot"


def test_exact_and_near_duplicates_are_removed():
    base_tokens = [f"rehab{i}" for i in range(400)]
    base = " ".join(base_tokens)
    near_tokens = base_tokens.copy()
    near_tokens[200] = "rehab_changed"
    near = " ".join(near_tokens)
    unique = " ".join(f"stroke{i}" for i in range(400))
    records = [
        {"id": "PMC1", "text": base, "topics": ["robotics"], "license_verified": "CC-BY"},
        {"id": "PMC2", "text": base, "topics": ["robotics"], "license_verified": "CC-BY"},
        {"id": "PMC3", "text": near, "topics": ["robotics"], "license_verified": "CC-BY"},
        {"id": "PMC4", "text": unique, "topics": ["stroke"], "license_verified": "CC0"},
    ]
    result = curate_records(
        records,
        quality_config=QualityConfig(min_words=10, max_duplicate_line_fraction=1.0),
        dedup_config=DedupConfig(jaccard_threshold=0.80),
        split_config=SplitConfig(train=0.8, val=0.1, test=0.1),
    )
    assert [row["id"] for row in result.accepted] == ["PMC1", "PMC4"]
    reasons = [row["reasons"][0] for row in result.rejected]
    assert "exact_duplicate" in reasons
    assert "near_duplicate" in reasons


def test_split_is_deterministic():
    config = SplitConfig(train=0.8, val=0.1, test=0.1, seed=7)
    assert deterministic_split("PMC123", config) == deterministic_split("PMC123", config)
