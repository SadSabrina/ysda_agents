"""
Statistics for reporting a method: intervals and the number of runs.

A point estimate without an interval is not a result. An order that does not
reproduce when the experiment is repeated is not a result either.
"""
from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist

import numpy as np


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson confidence interval for the proportion k/n.

    The normal approximation is wrong for small n and proportions near the boundary; Wilson is not:
    wilson_ci(0, 3) = (0.0, 0.56), not (0, 0).
    """
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile bootstrap interval for the mean."""
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    return tuple(np.percentile(values[idx].mean(axis=1), [2.5, 97.5]))


def bootstrap_order_prob(rates: dict[str, float], n_per: int, *,
                         n_boot: int = 200_000, seed: int = 0) -> dict[str, float]:
    """Probability that n_per runs per component give the correct order.

    rates — true frequencies, e.g. PN measured on a large sample.

    Three mutually exclusive shares, summing to 1:
      p_correct — the order is strict and matches the true one;
      p_tie     — at least two components got the same count, i.e.
                  the data do not determine the order at all;
      p_wrong   — the order is strict but wrong.

    A tie is not "almost right" but the absence of an answer, so it is separate.
    """
    p = np.array(list(rates.values()), dtype=float)
    truth = np.argsort(-p)
    counts = np.random.default_rng(seed).binomial(n_per, p[None, :], size=(n_boot, len(p)))
    ties = (np.diff(np.sort(counts, axis=1), axis=1) == 0).any(axis=1)
    matches = (np.argsort(-counts, axis=1, kind="stable") == truth[None, :]).all(axis=1)
    return {"p_correct": float((matches & ~ties).mean()),
            "p_tie": float(ties.mean()),
            "p_wrong": float((~matches & ~ties).mean())}


def min_repeats(p1: float, p2: float, *, conf: float = 0.95, power: float = 0.80) -> int:
    """Runs per component needed to tell two frequencies apart: the standard
    sample size formula for comparing two proportions, two-sided test."""
    z_a = NormalDist().inv_cdf(0.5 + conf / 2)
    z_b = NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * sqrt(2 * pbar * (1 - pbar)) + z_b * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return ceil(num / (p1 - p2) ** 2)
