"""
Контрфактуальный тест на необходимость.

Атрибуция (SHAP, LIME, occlusion) отвечает на вопрос о достаточности:
воспроизводит ли компонент результат сам по себе. Почти всегда спрашивают
о другом — о необходимости: изменился бы результат без этого компонента.
Здесь меряется именно необходимость, напрямую, через вмешательство.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Callable, Hashable, Iterable, Sequence

_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал Уилсона для доли k/n.

    Нормальное приближение при малых n и долях у границы врёт;
    Уилсон — нет. При k=0, n=3 честно отдаёт (0.0, 0.56), а не (0, 0).
    """
    if n <= 0:
        return (0.0, 1.0)
    if conf not in _Z:
        raise ValueError(f"conf должен быть одним из {sorted(_Z)}")
    z = _Z[conf]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Necessity:
    """Оценка PN (probability of necessity) для одного компонента."""

    component: Hashable
    n_runs: int
    n_flips: int
    conf: float = 0.95

    @property
    def pn(self) -> float:
        return self.n_flips / self.n_runs if self.n_runs else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_ci(self.n_flips, self.n_runs, self.conf)

    @property
    def resolved(self) -> bool:
        """False, если интервал накрывает и 0, и заметную величину:
        данных не хватает, чтобы вообще что-то утверждать."""
        lo, hi = self.ci
        return not (lo <= 0.0 and hi >= 0.5)

    def __str__(self) -> str:
        lo, hi = self.ci
        flag = "" if self.resolved else "   << не определено данными"
        return (f"{str(self.component):<24} PN={self.pn:6.1%}  "
                f"[{lo:5.1%}, {hi:5.1%}]  n={self.n_runs}{flag}")


def measure_necessity(
    components: Sequence[Hashable],
    run: Callable[[frozenset], object],
    *,
    n_repeats: int = 5,
    same: Callable[[object, object], bool] | None = None,
    conf: float = 0.95,
) -> list[Necessity]:
    """Измерить необходимость каждого компонента прямым вмешательством.

    Параметры
    ---------
    components : компоненты, которые можно убрать — инструменты агента,
        правила детектора, шаги пайплайна.
    run : run(active) -> outcome. Принимает frozenset активных компонентов,
        возвращает исход. Исход — что угодно сравнимое: строка ответа,
        вердикт deny/allow, множество вызванных инструментов.
    n_repeats : прогонов на компонент. Меньше 5 почти никогда не хватает;
        сколько нужно на самом деле — см. repeats.min_repeats.
    same : предикат равенства исходов. По умолчанию ==.

    Возвращает список Necessity, отсортированный по убыванию PN.

    Стоимость: (len(components) + 1) * n_repeats запусков.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats >= 1")
    same = same or (lambda a, b: a == b)
    full = frozenset(components)

    baselines = [run(full) for _ in range(n_repeats)]

    results: list[Necessity] = []
    for c in components:
        reduced = full - {c}
        flips = 0
        for i in range(n_repeats):
            after = run(reduced)
            # сравниваем со своим прогоном базлайна, а не с одним общим:
            # генерация стохастична, один базлайн занижает шум
            if not same(baselines[i], after):
                flips += 1
        results.append(Necessity(c, n_repeats, flips, conf))

    return sorted(results, key=lambda r: r.pn, reverse=True)


def report(results: Iterable[Necessity]) -> str:
    """Печатная сводка с явной отметкой неразрешённых оценок."""
    rows = list(results)
    lines = ["Необходимость (PN), вмешательством:", ""]
    lines += [str(r) for r in rows]
    unresolved = [r for r in rows if not r.resolved]
    if unresolved:
        lines += ["", f"{len(unresolved)} из {len(rows)} не определены данными. "
                      "Увеличьте n_repeats или не утверждайте порядок."]
    return "\n".join(lines)
