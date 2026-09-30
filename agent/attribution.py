"""
Per-tool attribution on top of AgentSHAP.analyze() results.

occlusion is ported from agentshap_tutorial_rus.ipynb and reuses the
coalitions that analyze() has already run: not a single extra model call.

Also here: the recovered closed form of the Shapley normalization, and the
contribution with the baseline taken from the measured noise floor instead
of the unreachable 1.
"""
from __future__ import annotations

import pandas as pd


def coalitions(results_df: pd.DataFrame, tool_names: list[str]) -> pd.DataFrame:
    """analyze() rows as a tool-presence matrix."""
    present = results_df["tools_available"].astype(str).str.split("+")
    return pd.DataFrame([[int(t in p) for t in tool_names] for p in present],
                        columns=tool_names, index=results_df.index)


def leave_one_out(results_df: pd.DataFrame, tool_names: list[str]) -> pd.Series:
    """similarity of coalitions N \\ {i}: the index is the missing tool i."""
    X = coalitions(results_df, tool_names)
    loo = X.sum(axis=1) == len(tool_names) - 1
    missing = (1 - X[loo]).idxmax(axis=1)
    return pd.Series(results_df.loc[loo, "similarity"].to_numpy(), index=missing.to_numpy(),
                     name="similarity").reindex(tool_names)


def tool_occlusion_importance(results_df: pd.DataFrame, tool_names: list[str]) -> pd.Series:
    """Effect of removing one tool: 1 - similarity(N \\ {i})."""
    return (1 - leave_one_out(results_df, tool_names)).rename("occlusion")




def shapley_from_occlusion(occ: pd.Series) -> pd.Series:
    """The library's normalization, recovered:

        phi_i = (occ_i - min occ) / (sum occ - n * min occ)

    The minimal tool is forced to zero, whatever its true contribution.
    Hence the systematic violation of the dummy axiom.
    """
    o = occ.to_numpy(dtype=float)
    return pd.Series((o - o.min()) / (o.sum() - len(o) * o.min()),
                     index=occ.index, name="shapley_closed_form")


def corrected_contributions(results_df: pd.DataFrame, tool_names: list[str],
                            noise_floor: float) -> pd.Series:
    """Contribution with the measured baseline: max(0, nu - similarity(N \\ {i})).

    For a tool the agent did not call, similarity stays near nu and the
    contribution drops to zero: dummy holds by construction.
    Not normalized: dividing near-zeros by their near-zero sum
    inflates the noise back up to 1.
    """
    return (noise_floor - leave_one_out(results_df, tool_names)).clip(lower=0).rename("corrected")
