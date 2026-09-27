"""
sanity_kit — минимальный набор проверок для утверждений вида
«вот это повлияло на решение агента».

Три функции, три вопроса:
  necessity  — а без этого компонента результат был бы другим?
  repeats    — сколько прогонов нужно, чтобы порядок вообще был определён?
  redundancy — не делят ли несколько компонентов один и тот же вклад?

Лекция ШАД, 30.09.2026.
"""

from .necessity import Necessity, measure_necessity, wilson_ci
from .repeats import bootstrap_order_prob, min_repeats
from .redundancy import cofire_matrix, redundant_groups, shapley_dilution

__all__ = [
    "Necessity", "measure_necessity", "wilson_ci",
    "bootstrap_order_prob", "min_repeats",
    "cofire_matrix", "redundant_groups", "shapley_dilution",
]
