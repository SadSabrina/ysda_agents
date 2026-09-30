"""
Evaluation of a step attribution method.

A method is a function case -> list of candidate indices, from most likely
to least likely. Anything not in the list counts as not found.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .load import Case
from .stats import bootstrap_ci

Method = Callable[[Case], list[int]]


def ranks(cases: list[Case], method: Method) -> np.ndarray:
    """Rank of the first gold cause in the method's output, one-based; inf means not found."""
    out = []
    for c in cases:
        ranking = method(c)
        hit = [r for r, i in enumerate(ranking, 1) if i in c.causes]
        out.append(hit[0] if hit else np.inf)
    return np.array(out, dtype=float)


def evaluate(cases: list[Case], methods: dict[str, Method]) -> pd.DataFrame:
    """top-1, top-3 and MRR of each method with bootstrap intervals."""
    rows = []
    for name, method in methods.items():
        r = ranks(cases, method)
        row = {"method": name, "n": len(cases)}
        for metric, values in (("top-1", (r <= 1).astype(float)),
                               ("top-3", (r <= 3).astype(float)),
                               ("MRR", 1 / r)):
            lo, hi = bootstrap_ci(values)
            row[metric] = f"{values.mean():.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")
