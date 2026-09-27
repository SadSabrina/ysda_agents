"""
Атрибуция по инструментам поверх результатов AgentSHAP.analyze().

occlusion и tool-LIME перенесены из agentshap_tutorial_rus.ipynb.
Обе переиспользуют коалиции, которые analyze() уже прогнал, — ни одного
дополнительного вызова модели.

Здесь же — восстановленная закрытая форма шеплевской нормировки и
исправленная версия вклада, в которой baseline берётся из измеренного
noise floor, а не из недостижимой единицы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def coalitions(results_df: pd.DataFrame, tool_names: list[str]) -> pd.DataFrame:
    """Разложить строки analyze() в матрицу присутствия инструментов."""
    rows = []
    for _, row in results_df.iterrows():
        present = set(str(row["tools_available"]).split("+"))
        rows.append([1 if t in present else 0 for t in tool_names])
    return pd.DataFrame(rows, columns=tool_names, index=results_df.index)


def tool_occlusion_importance(results_df: pd.DataFrame, tool_names: list[str]) -> pd.Series:
    """Прямой эффект удаления одного инструмента: 1 - similarity.

    Берутся только коалиции, где отсутствует ровно один инструмент.
    Это тот же примитив, поверх которого строится Шепли, но без
    теоретико-игрового взвешивания.
    """
    scores = {}
    for _, row in results_df.iterrows():
        present = set(str(row["tools_available"]).split("+"))
        missing = set(tool_names) - present
        if len(missing) != 1:
            continue
        scores[missing.pop()] = 1 - row["similarity"]
    return pd.Series(scores, name="occlusion").sort_values(ascending=False)


def tool_lime_importance(results_df: pd.DataFrame, tool_names: list[str],
                         alpha: float = 1.0) -> pd.Series:
    """Локальная линейная суррогатная модель по присутствию инструментов."""
    tool_names = list(tool_names)
    X = coalitions(results_df, tool_names).to_numpy(dtype=float)
    y = results_df["similarity"].to_numpy(dtype=float)
    weights = X.sum(axis=1) / len(tool_names)
    model = Ridge(alpha=alpha)
    model.fit(X, y, sample_weight=weights)
    return pd.Series(model.coef_, index=tool_names, name="lime").sort_values(ascending=False)


def predicted_lime_slope(n_tools: int, alpha: float = 1.0) -> float:
    """Предсказанный наклон lime по occlusion: w/(w+alpha), w = (n-1)/n.

    Для n=4, alpha=1 это ровно 3/7.
    """
    w = (n_tools - 1) / n_tools
    return w / (w + alpha)


def shapley_from_occlusion(occ: pd.Series) -> pd.Series:
    """Восстановленная нормировка библиотеки: сдвиг минимума в ноль и деление на сумму.

    phi_i = (occ_i - min occ) / (sum occ - n * min occ)

    Побочный эффект, который и ломает аксиому dummy: минимальный инструмент
    обнуляется принудительно, каким бы ни был его настоящий вклад.
    """
    o = occ.to_numpy(dtype=float)
    n = len(o)
    denom = o.sum() - n * o.min()
    if abs(denom) < 1e-12:
        return pd.Series(np.zeros(n), index=occ.index, name="shapley_closed_form")
    return pd.Series((o - o.min()) / denom, index=occ.index, name="shapley_closed_form")


def corrected_contributions(results_df: pd.DataFrame, tool_names: list[str],
                            noise_floor: float, *, normalize: bool = False) -> pd.Series:
    """Вклад с исправленным baseline.

    Вместо недостижимой единицы берём измеренный потолок близости nu:

        contribution_i = max(0, nu - similarity(N \\ {i}))

    Для инструмента, который агент не вызывал, similarity почти равна nu,
    значит вклад садится в ноль — аксиома dummy начинает выполняться
    по построению, а не случайно.

    normalize=True делит на сумму (как в библиотеке). При всех нулевых
    вкладах возвращаются нули, а не деление на ноль.
    """
    scores = {}
    for _, row in results_df.iterrows():
        present = set(str(row["tools_available"]).split("+"))
        missing = set(tool_names) - present
        if len(missing) != 1:
            continue
        scores[missing.pop()] = max(0.0, noise_floor - float(row["similarity"]))

    s = pd.Series(scores, name="corrected").reindex(tool_names).fillna(0.0)
    if normalize:
        total = s.sum()
        s = s / total if total > 1e-12 else s * 0.0
    return s.sort_values(ascending=False)


def compare(results_df: pd.DataFrame, shapley_values: dict, tool_names: list[str],
            noise_floor: float | None = None, alpha: float = 1.0) -> pd.DataFrame:
    """Свести все методы в одну таблицу."""
    occ = tool_occlusion_importance(results_df, tool_names)
    cols = {
        "shapley": pd.Series(shapley_values),
        "occlusion": occ,
        "lime": tool_lime_importance(results_df, tool_names, alpha=alpha),
        "shapley_closed_form": shapley_from_occlusion(occ),
    }
    if noise_floor is not None:
        cols["corrected"] = corrected_contributions(results_df, tool_names, noise_floor,
                                                    normalize=True)
    return pd.DataFrame(cols).sort_values("occlusion", ascending=False)


def dummy_check(results_df: pd.DataFrame, tool_names: list[str],
                noise_floor: float, noise_sd: float = 0.0, k: float = 2.0) -> pd.DataFrame:
    """Проверка аксиомы dummy на промпте, где ни один инструмент не нужен.

    Сравниваем сырые вклады (без нормировки!) с шумовым порогом k * sd.

    Нормировать здесь нельзя: деление почти нулевых величин на их почти
    нулевую сумму заново раздувает шум до единицы. Именно это и делает
    библиотека, и именно поэтому нарушение dummy выглядит так убедительно.
    """
    raw_old = tool_occlusion_importance(results_df, tool_names).reindex(tool_names)
    raw_new = corrected_contributions(results_df, tool_names, noise_floor,
                                      normalize=False).reindex(tool_names)
    threshold = k * noise_sd
    return pd.DataFrame({
        "occlusion_baseline_1": raw_old,
        "corrected_baseline_nu": raw_new,
        "порог шума": threshold,
        "dummy держится": raw_new <= threshold,
    }).sort_values("occlusion_baseline_1", ascending=False)
