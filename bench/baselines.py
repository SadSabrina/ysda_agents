"""
Тривиальные методы. Любой настоящий метод атрибуции обязан их обыграть,
иначе он измеряет структуру датасета, а не причинность.
"""
from __future__ import annotations

import random

from .load import Case


def random_order(case: Case) -> list[int]:
    cand = case.candidates
    random.Random(case.case_id).shuffle(cand)
    return cand


def nearest_first(case: Case) -> list[int]:
    """Ближайший к цели шаг первым."""
    return case.candidates[::-1]


def earliest_first(case: Case) -> list[int]:
    return case.candidates


def last_user_first(case: Case) -> list[int]:
    """Последнее сообщение пользователя, дальше по близости."""
    cand = case.candidates[::-1]
    return sorted(cand, key=lambda i: case.steps[i].role != "user")


def injection_marker_first(case: Case) -> list[int]:
    """Шаги с флагом contains_injection_marker из разметки AgentDojo, дальше по порядку.
    Флаг лежит в самих данных, то есть это подсказка, а не метод."""
    return sorted(case.candidates, key=lambda i: not case.steps[i].flags.get("injection_marker"))


ALL = {
    "случайный": random_order,
    "ближайший к цели": nearest_first,
    "самый ранний": earliest_first,
    "последний user": last_user_first,
}
