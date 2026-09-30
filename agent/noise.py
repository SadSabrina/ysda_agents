"""
Измерение noise floor.

Baseline в AgentSHAP зашит как 1.0: предполагается, что ответ на полном
наборе инструментов совпадает сам с собой. Но temperature=0.5 захардкожена
в token_shap/base.py, поэтому два прогона с одним и тем же тулсетом дают
разные ответы, и настоящий потолок близости

    nu = E[ cos( ответ(N), ответ'(N) ) ]  <  1

Разница (1 - nu) — постоянное положительное смещение, которое нормировка
размазывает по всем инструментам, включая неиспользованные.

Про метрику. Чтобы nu был сопоставим с колонкой similarity из analyze(),
считать его надо так же, как библиотека (agent_shap.py,
_get_df_per_combination_with_usage): TF-IDF обучается на всём корпусе ответов
разом, первый ответ — бейзлайн, остальные сравниваются с ним. Попарная
векторизация строила бы словарь по двум документам и давала другое число.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev


@dataclass
class NoiseFloor:
    nu: float
    sd: float
    samples: list[float]
    tool_sets: list[list[str]]
    prompts: list[str]
    model: str

    @property
    def bias(self) -> float:
        """Смещение, которое размазывается по вкладам: 1 - nu."""
        return 1.0 - self.nu

    @property
    def behavioural_noise(self) -> float:
        """Доля прогонов, где набор вызванных инструментов отличается от первого
        прогона того же промпта. Текстовый шум — одно, другой набор вызовов
        на одном и том же промпте — другое, и куда неприятнее."""
        per_prompt = len(self.tool_sets) // len(self.prompts)
        diffs = total = 0
        for k in range(len(self.prompts)):
            runs = self.tool_sets[k * per_prompt:(k + 1) * per_prompt]
            diffs += sum(set(r) != set(runs[0]) for r in runs[1:])
            total += len(runs) - 1
        return diffs / total

    def __str__(self) -> str:
        return (f"noise floor  nu = {self.nu:.4f} ± {self.sd:.4f}   "
                f"({len(self.samples)} сравнений, {len(self.prompts)} промпта, {self.model})\n"
                f"смещение   1 - nu = {self.bias:.4f}\n"
                f"поведенческий шум: {self.behavioural_noise:.1%} прогонов с другим набором инструментов")

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> "NoiseFloor":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def measure_noise_floor(agent_shap, prompts: list[str], *, n_runs: int = 10) -> NoiseFloor:
    """n_runs генераций с полным тулсетом на каждый промпт.

    Стоимость: len(prompts) * n_runs вызовов агента.
    """
    sims: list[float] = []
    tool_sets: list[list[str]] = []
    for prompt in prompts:
        responses = []
        for _ in range(n_runs):
            text, usage = agent_shap._execute_with_tools(prompt, agent_shap.tools)
            responses.append(text)
            tool_sets.append(sorted(k for k, v in usage.items() if v > 0))
        vectors = agent_shap.vectorizer.vectorize(responses) # векторизованные ответы 

        # близость меж ответами, тут считаем как
        # «близость ответа к одному случайному бейзлайну».
        sims += [float(s) for s in agent_shap.vectorizer.calculate_similarity(vectors[0], vectors[1:])] 


    return NoiseFloor(nu=mean(sims), sd=stdev(sims), samples=sims, tool_sets=tool_sets,
                      prompts=list(prompts), model=agent_shap.model.model_name)

# Что есть vectorize и calculate_similarity в AGENTSHAP:

# def vectorize(self, texts):
#     self.vectorizer = TfidfVectorizer().fit(texts)      # TF-IDF обучается на тех текстах, что передали
#     return self.vectorizer.transform(texts).toarray()

# def calculate_similarity(self, base_vector, comparison_vectors):
#     return cosine_similarity(...)                        # sklearn, косинус между TF-IDF векторами
