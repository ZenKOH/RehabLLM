from rehab_minillm.domain import classify_document, target_counts


def test_domain_classifier_prioritises_robotics():
    text = "Stroke rehabilitation used a robot-assisted upper-limb device for motor training."
    assert classify_document(text) == "robotics"


def test_v04_target_counts_sum_to_total():
    counts = target_counts(6000)
    assert sum(counts.values()) == 6000
    assert counts["robotics"] == 1800
    assert counts["core_rehab"] == 2700
