"""
Атрибуция по инструментам поверх результатов AgentSHAP.analyze().

occlusion перенесён из agentshap_tutorial_rus.ipynb и переиспользует
коалиции, которые analyze() уже прогнал: ни одного дополнительного
вызова модели.

Здесь же восстановленная закрытая форма шеплевской нормировки и вклад
с baseline из измеренного noise floor вместо недостижимой единицы.
"""
from __future__ import annotations

import pandas as pd


def coalitions(results_df: pd.DataFrame, tool_names: list[str]) -> pd.DataFrame:
    """Строки analyze() как матрица присутствия инструментов."""
    present = results_df["tools_available"].astype(str).str.split("+")
    return pd.DataFrame([[int(t in p) for t in tool_names] for p in present],
                        columns=tool_names, index=results_df.index)


def leave_one_out(results_df: pd.DataFrame, tool_names: list[str]) -> pd.Series:
    """similarity коалиций N \\ {i}: индекс — отсутствующий инструмент i."""
    X = coalitions(results_df, tool_names)
    loo = X.sum(axis=1) == len(tool_names) - 1
    missing = (1 - X[loo]).idxmax(axis=1)
    return pd.Series(results_df.loc[loo, "similarity"].to_numpy(), index=missing.to_numpy(),
                     name="similarity").reindex(tool_names)


def tool_occlusion_importance(results_df: pd.DataFrame, tool_names: list[str]) -> pd.Series:
    """Эффект удаления одного инструмента: 1 - similarity(N \\ {i})."""
    return (1 - leave_one_out(results_df, tool_names)).rename("occlusion")




def shapley_from_occlusion(occ: pd.Series) -> pd.Series:
    """Восстановленная нормировка библиотеки:

        phi_i = (occ_i - min occ) / (sum occ - n * min occ)

    Минимальный инструмент обнуляется принудительно, каким бы ни был
    его настоящий вклад. Отсюда системное нарушение аксиомы dummy.
    """
    o = occ.to_numpy(dtype=float)
    return pd.Series((o - o.min()) / (o.sum() - len(o) * o.min()),
                     index=occ.index, name="shapley_closed_form")


def corrected_contributions(results_df: pd.DataFrame, tool_names: list[str],
                            noise_floor: float) -> pd.Series:
    """Вклад с измеренным baseline: max(0, nu - similarity(N \\ {i})).

    Для инструмента, который агент не вызывал, similarity держится около nu,
    и вклад садится в ноль: dummy выполняется по построению.
    Не нормируется: деление почти нулей на их почти нулевую сумму
    заново раздувает шум до единицы.
    """
    return (noise_floor - leave_one_out(results_df, tool_names)).clip(lower=0).rename("corrected")
