"""
Сколько прогонов нужно, чтобы порядок важностей вообще был определён.

Короткий ответ: больше, чем кажется. На реальных данных агента с четырьмя
инструментами оценка по трём прогонам воспроизводит истинный порядок
в 15,2 % случаев, а в 70 % где-нибудь получается ничья.
"""
from __future__ import annotations

import numpy as np

_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def bootstrap_order_prob(
    rates: dict[str, float],
    n_per: int,
    *,
    n_boot: int = 200_000,
    seed: int = 0,
) -> dict[str, float]:
    """С какой вероятностью n_per прогонов на компонент дадут верный порядок.

    Параметры
    ---------
    rates : истинные частоты, например измеренные PN на большой выборке
        {"get_news": 0.519, "get_stock_price": 0.266, "get_weather": 0.169}
    n_per : сколько прогонов на компонент делает наивный эксперимент.

    Возвращает
    ----------
    Три взаимоисключающие доли, в сумме 1:
    p_correct : порядок строгий (ничьих нет) и совпал с истинным
    p_tie     : хотя бы два компонента получили одинаковый счёт,
                то есть порядок данными не определён вовсе
    p_wrong   : порядок строгий, но неверный

    Ничья — это не «почти правильно», а отсутствие ответа, поэтому
    она вынесена в отдельную категорию, а не засчитана в p_correct.
    """
    names = list(rates)
    p = np.array([rates[k] for k in names], dtype=float)
    truth = np.argsort(-p)

    rng = np.random.default_rng(seed)
    # счётчик успехов на каждый компонент в каждой симуляции
    counts = rng.binomial(n_per, p[None, :], size=(n_boot, len(names)))

    srt = np.sort(counts, axis=1)
    ties = (np.diff(srt, axis=1) == 0).any(axis=1)

    order = np.argsort(-counts, axis=1, kind="stable")
    matches = (order == truth[None, :]).all(axis=1)

    correct = matches & ~ties          # строгий и верный
    wrong = ~matches & ~ties           # строгий и неверный

    return {
        "p_correct": float(correct.mean()),
        "p_tie": float(ties.mean()),
        "p_wrong": float(wrong.mean()),
        "n_per": n_per,
        "n_boot": n_boot,
    }


def min_repeats(p1: float, p2: float, *, conf: float = 0.95, power: float = 0.80) -> int:
    """Сколько прогонов на компонент, чтобы отличить две частоты.

    Обычная формула для двух долей. Даёт тот размер выборки, при котором
    разницу p1 - p2 удаётся увидеть с заданной мощностью.

    >>> min_repeats(0.52, 0.27)      # get_news против get_stock_price
    58
    >>> min_repeats(0.27, 0.17)      # get_stock_price против get_weather
    291
    """
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("частоты должны быть строго между 0 и 1")
    if p1 == p2:
        raise ValueError("одинаковые частоты неразличимы при любом n")
    z_a = _Z[conf]
    z_b = _Z[round(power, 2)] if round(power, 2) in _Z else _Z[0.80]
    pbar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) +
           z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


def order_is_claimable(rates: dict[str, float], n_per: int, *, threshold: float = 0.80,
                       n_boot: int = 50_000, seed: int = 0) -> bool:
    """Можно ли при таком n_per вообще заявлять порядок.

    Порог 0.80 — «в четырёх случаях из пяти порядок воспроизведётся».
    """
    return bootstrap_order_prob(rates, n_per, n_boot=n_boot, seed=seed)["p_correct"] >= threshold
