"""
Trivial baselines. Any real attribution method must beat them,
otherwise it measures the structure of the dataset, not causality.
"""
from __future__ import annotations

import random

from .load import Case


def random_order(case: Case) -> list[int]:
    cand = case.candidates
    random.Random(case.case_id).shuffle(cand)
    return cand


def nearest_first(case: Case) -> list[int]:
    """The step nearest to the target first."""
    return case.candidates[::-1]


def earliest_first(case: Case) -> list[int]:
    return case.candidates


def last_user_first(case: Case) -> list[int]:
    """The last user message first, then by proximity."""
    cand = case.candidates[::-1]
    return sorted(cand, key=lambda i: case.steps[i].role != "user")


def injection_marker_first(case: Case) -> list[int]:
    """Steps with the contains_injection_marker flag from the AgentDojo annotation first, then in order.
    The flag is in the data itself, so this is a hint, not a method."""
    return sorted(case.candidates, key=lambda i: not case.steps[i].flags.get("injection_marker"))


ALL = {
    "random": random_order,
    "nearest to target": nearest_first,
    "earliest": earliest_first,
    "last user message": last_user_first,
}
