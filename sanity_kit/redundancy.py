"""
Детектор избыточности.

Теорема об избыточности: если два компонента взаимозаменяемы, то аксиомы
симметрии и эффективности заставляют выдать им равный положительный вклад,
хотя необходимость каждого равна нулю. При k взаимозаменяемых компонентах
каждый получает 1/k — и чем больше перекрытие, тем ничтожнее выглядит каждый.

Практический смысл: если в вашем сенсоре два правила всегда срабатывают
вместе, «важность правила» не определена, сколько её ни мерь.
"""
from __future__ import annotations

import numpy as np


def cofire_matrix(fires: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Матрица совместного срабатывания.

    fires : {имя компонента: булев вектор по событиям}
    Возвращает (имена, матрицу), где M[i, j] = P(j сработал | i сработал).
    Матрица несимметрична: правило-надмножество это показывает.
    """
    names = list(fires)
    X = np.array([np.asarray(fires[k], dtype=bool) for k in names])
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        ni = X[i].sum()
        for j in range(n):
            M[i, j] = (X[i] & X[j]).sum() / ni if ni else np.nan
    return names, M


def redundant_groups(fires: dict[str, np.ndarray], *, threshold: float = 0.98) -> list[list[str]]:
    """Группы компонентов, которые практически всегда срабатывают вместе.

    Внутри такой группы индивидуальная атрибуция бессмысленна:
    Шепли поделит вклад поровну, а необходимость каждого будет нулевой.
    """
    names, M = cofire_matrix(fires)
    n = len(names)
    seen: set[int] = set()
    groups: list[list[str]] = []
    for i in range(n):
        if i in seen:
            continue
        grp = [i]
        for j in range(i + 1, n):
            if j not in seen and M[i, j] >= threshold and M[j, i] >= threshold:
                grp.append(j)
        if len(grp) > 1:
            seen.update(grp)
            groups.append([names[k] for k in grp])
    return groups


def shapley_dilution(k: int) -> float:
    """Какой вклад получит каждый из k взаимозаменяемых компонентов.

    Ровно 1/k, при том что PN каждого равна нулю.
    """
    if k < 1:
        raise ValueError("k >= 1")
    return 1.0 / k


def warn(fires: dict[str, np.ndarray], *, threshold: float = 0.98) -> str:
    groups = redundant_groups(fires, threshold=threshold)
    if not groups:
        return "Избыточных групп не найдено: индивидуальная атрибуция осмысленна."
    out = ["Найдена избыточность — индивидуальные вклады внутри групп не интерпретируемы:", ""]
    for g in groups:
        out.append(f"  {' == '.join(g)}   каждый получит ~{shapley_dilution(len(g)):.2f}, PN каждого ~0")
    return "\n".join(out)
