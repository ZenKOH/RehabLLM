from rehab_minillm.readiness import assess_v04_readiness


def test_readiness_passes_for_healthy_enriched_corpus():
    stats = {
        "selected_records": 4500,
        "category_counts": {
            "core_rehab": 2200,
            "robotics": 1200,
            "neurotechnology": 600,
            "assistive": 500,
        },
        "split_counts": {"train": 4050, "val": 225, "test": 225},
        "cleaning": {
            "original_chars": 20_000_000,
            "removed_chars": 2_000_000,
        },
    }
    result = assess_v04_readiness(stats)
    assert result.ready
    assert result.checks["robotics_representation"]


def test_readiness_holds_for_v03_like_domain_mix():
    stats = {
        "selected_records": 3546,
        "category_counts": {
            "core_rehab": 3016,
            "robotics": 32,
            "neurotechnology": 104,
            "assistive": 394,
        },
        "split_counts": {"train": 3191, "val": 165, "test": 190},
        "cleaning": {
            "original_chars": 20_000_000,
            "removed_chars": 1_000_000,
        },
    }
    result = assess_v04_readiness(stats)
    assert not result.ready
    assert not result.checks["robotics_representation"]
    assert not result.checks["combined_technology_representation"]
