from rehab_minillm.corpus import ROBOTICS_ENRICHMENT_QUERIES, iter_query_plan


def test_robotics_enrichment_plan_is_targeted_and_licensed():
    names = {topic.name for topic in ROBOTICS_ENRICHMENT_QUERIES}
    assert "robot_assisted_gait" in names
    assert "bci_robotic_rehabilitation" in names
    plan = list(iter_query_plan(ROBOTICS_ENRICHMENT_QUERIES))
    assert len(plan) == len(ROBOTICS_ENRICHMENT_QUERIES) * 3
    assert all("NOT articletyperetraction" in query for _, _, query in plan)
