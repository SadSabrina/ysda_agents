"""
Измерение noise floor.

Baseline в AgentSHAP зашит как 1.0: предполагается, что ответ на полном
наборе инструментов совпадает сам с собой. Но temperature=0.5 захардкожена
в token_shap/base.py, поэтому два прогона с одним и тем же тулсетом дают
разные ответы, и настоящий потолок близости равен

    nu = E[ cos( ответ(N), ответ'(N) ) ]  <  1

Разница (1 - nu) — постоянное положительное смещение, которое нормировка
размазывает по всем инструментам, включая неиспользованные. Отсюда
нарушение аксиомы dummy.

ВАЖНО про метрику. Чтобы nu был сопоставим с колонкой similarity из
analyze(), считать его надо ровно так же, как это делает библиотека
(base.py, _get_df_per_combination):

    vectors = vectorizer.vectorize([baseline] + responses)   # фит на всём корпусе
    sims    = vectorizer.calculate_similarity(vectors[0], vectors[1:])

TfidfTextVectorizer.vectorize() переобучает TF-IDF на том, что ему дали.
Если сравнивать тексты попарно, словарь строится по двум документам,
и число получается несопоставимым с тем, что выдаёт analyze().
Поэтому здесь генерируются все ответы, а векторизация одна на весь корпус.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev


@dataclass
class NoiseFloor:
    nu: float
    sd: float
    n_runs: int
    samples: list[float] = field(repr=False, default_factory=list)
    responses: list[str] = field(repr=False, default_factory=list)
    tool_usage: list[dict] = field(repr=False, default_factory=list)

    @property
    def bias(self) -> float:
        """Смещение, которое размазывается по вкладам: 1 - nu."""
        return 1.0 - self.nu

    def expected_dummy_share(self, n_tools: int) -> float:
        """Грубая оценка паразитного вклада на один неиспользованный инструмент."""
        return self.bias / n_tools

    @property
    def behavioural_noise(self) -> float:
        """Доля прогонов, где агент вызвал не тот же набор инструментов, что в первом.

        Текстовый шум — это одно, а разные наборы вызванных инструментов
        на одном и том же промпте — другое, и куда неприятнее.
        """
        if len(self.tool_usage) < 2:
            return float("nan")
        first = frozenset(k for k, v in self.tool_usage[0].items() if v > 0)
        diff = sum(1 for u in self.tool_usage[1:]
                   if frozenset(k for k, v in u.items() if v > 0) != first)
        return diff / (len(self.tool_usage) - 1)

    def __str__(self) -> str:
        lines = [
            f"noise floor  nu = {self.nu:.4f} ± {self.sd:.4f}   ({self.n_runs} прогонов)",
            f"смещение   1 - nu = {self.bias:.4f}",
        ]
        bn = self.behavioural_noise
        if bn == bn:  # not nan
            lines.append(f"поведенческий шум: {bn:.1%} прогонов с другим набором инструментов")
        return "\n".join(lines)


def measure_noise_floor(agent_shap, prompt: str, *, n_runs: int = 25,
                        verbose: bool = True) -> NoiseFloor:
    """Померить потолок близости независимых генераций с полным тулсетом.

    Параметры
    ---------
    agent_shap : объект AgentSHAP (bundle.agent_shap). Из него берутся
        и инструменты, и векторизатор, чтобы метрика совпала с analyze().
    prompt     : один промпт. Для устойчивой оценки стоит повторить на нескольких
        и усреднить — см. measure_noise_floor_multi.
    n_runs     : сколько генераций. Первая работает базлайном, остальные
        с ней сравниваются, то есть замеров получается n_runs - 1.

    Стоимость: n_runs вызовов модели (а не 2 * n_runs).
    """
    if n_runs < 2:
        raise ValueError("нужно минимум 2 прогона: один базлайн и один для сравнения")

    responses: list[str] = []
    usages: list[dict] = []
    for i in range(n_runs):
        text, usage = agent_shap._execute_with_tools(prompt, agent_shap.tools)
        responses.append(text)
        usages.append(usage)
        if verbose:
            called = sorted(k for k, v in usage.items() if v > 0)
            print(f"  прогон {i + 1:>2}/{n_runs}: {len(text):>4} симв., инструменты: {called or '—'}")

    # Ровно как в base.py::_get_df_per_combination — одна векторизация на корпус
    vectors = agent_shap.vectorizer.vectorize(responses)
    sims = agent_shap.vectorizer.calculate_similarity(vectors[0], vectors[1:])
    sims = [float(s) for s in sims]

    return NoiseFloor(
        nu=mean(sims),
        sd=stdev(sims) if len(sims) > 1 else 0.0,
        n_runs=n_runs,
        samples=sims,
        responses=responses,
        tool_usage=usages,
    )


def measure_noise_floor_multi(agent_shap, prompts: list[str], *, n_runs: int = 10,
                              verbose: bool = True) -> NoiseFloor:
    """То же самое на нескольких промптах: nu усредняется по ним.

    Один промпт может оказаться нетипично стабильным или нетипично шумным.
    Для числа, которое пойдёт на слайд, лучше брать несколько.
    """
    all_sims: list[float] = []
    all_resp: list[str] = []
    all_usage: list[dict] = []
    for p in prompts:
        if verbose:
            print(f"\n{p!r}")
        nf = measure_noise_floor(agent_shap, p, n_runs=n_runs, verbose=verbose)
        all_sims += nf.samples
        all_resp += nf.responses
        all_usage += nf.tool_usage
        if verbose:
            print(f"  -> nu = {nf.nu:.4f}")

    return NoiseFloor(
        nu=mean(all_sims),
        sd=stdev(all_sims) if len(all_sims) > 1 else 0.0,
        n_runs=len(prompts) * n_runs,
        samples=all_sims,
        responses=all_resp,
        tool_usage=all_usage,
    )
