from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# Intentionally excludes NC and ND licences for the default reusable training corpus.
PERMISSIVE_PMC_LICENSE_FILTERS: dict[str, str] = {
    "CC0": '"cc0 license"[filter]',
    "CC-BY": '"cc by license"[filter]',
    "CC-BY-SA": '"cc by-sa license"[filter]',
}


@dataclass(frozen=True)
class TopicQuery:
    name: str
    query: str
    weight: float = 1.0
    tags: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_REHAB_QUERIES: tuple[TopicQuery, ...] = (
    TopicQuery(
        "rehabilitation_robotics_general",
        '(rehabilitation OR neurorehabilitation) AND (robot OR robotic OR robotics OR exoskeleton OR "end effector")',
        2.0,
        ("robotics", "rehabilitation"),
    ),
    TopicQuery(
        "stroke_robotics",
        '(stroke AND rehabilitation) AND (robot OR robotic OR exoskeleton OR "end effector")',
        1.7,
        ("stroke", "robotics"),
    ),
    TopicQuery(
        "upper_limb_robotics",
        '(("upper limb" OR arm OR hand) AND rehabilitation) AND (robot OR robotic OR exoskeleton OR "end effector")',
        1.5,
        ("upper-limb", "robotics"),
    ),
    TopicQuery(
        "gait_robotics",
        '(gait AND rehabilitation) AND (robot OR robotic OR exoskeleton OR treadmill)',
        1.5,
        ("gait", "robotics"),
    ),
    TopicQuery(
        "spinal_cord_injury_robotics",
        '("spinal cord injury" AND rehabilitation) AND (robot OR robotic OR exoskeleton)',
        1.4,
        ("sci", "robotics"),
    ),
    TopicQuery(
        "functional_electrical_stimulation",
        '(rehabilitation OR neurorehabilitation) AND ("functional electrical stimulation" OR FES)',
        1.2,
        ("fes", "neurotechnology"),
    ),
    TopicQuery(
        "brain_computer_interface_rehab",
        '(rehabilitation OR neurorehabilitation) AND ("brain computer interface" OR "brain-computer interface" OR BCI)',
        1.1,
        ("bci", "neurotechnology"),
    ),
    TopicQuery(
        "assistive_technology",
        '(rehabilitation AND ("assistive technology" OR prosthesis OR prosthetic OR orthosis OR orthotic))',
        1.0,
        ("assistive-technology",),
    ),
    TopicQuery(
        "neurological_rehabilitation",
        '(rehabilitation OR neurorehabilitation) AND (stroke OR "spinal cord injury" OR "traumatic brain injury" OR Parkinson OR "cerebral palsy")',
        1.2,
        ("neurological-rehabilitation",),
    ),
    TopicQuery(
        "rehab_outcomes",
        '(rehabilitation AND ("activities of daily living" OR gait OR balance OR "motor function" OR participation OR functioning))',
        0.9,
        ("outcomes", "functioning"),
    ),
    TopicQuery(
        "therapy_and_rehab",
        '((physiotherapy OR "physical therapy" OR "occupational therapy") AND rehabilitation)',
        0.8,
        ("therapy", "rehabilitation"),
    ),
)


def build_pmc_query(topic_query: str, license_filter: str) -> str:
    return (
        f"({topic_query}) AND {license_filter} "
        'NOT "pmc embargo"[filter] NOT hasretractionin NOT articletypeexpressionofconcern'
    )


def iter_query_plan(
    topics: Iterable[TopicQuery] = DEFAULT_REHAB_QUERIES,
) -> Iterable[tuple[TopicQuery, str, str]]:
    for topic in topics:
        for licence, licence_filter in PERMISSIVE_PMC_LICENSE_FILTERS.items():
            yield topic, licence, build_pmc_query(topic.query, licence_filter)
