"""
Measuring the noise floor.

The baseline in AgentSHAP is hardcoded as 1.0: it assumes that the answer
with the full tool set equals itself. But temperature=0.5 is hardcoded in
token_shap/base.py, so two runs with the same tool set give different
answers, and the real similarity ceiling is

    nu = E[ cos( answer(N), answer'(N) ) ]  <  1

The gap (1 - nu) is a constant positive bias that the normalization spreads
over all tools, including unused ones.

On the metric. For nu to be comparable with the similarity column from
analyze(), it has to be computed the same way the library does it
(agent_shap.py, _get_df_per_combination_with_usage): TF-IDF is fitted on the
whole corpus of answers at once, the first answer is the baseline, and the
rest are compared with it. Pairwise vectorization would build the vocabulary
from two documents and give a different number.
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
        """The bias that gets spread over the contributions: 1 - nu."""
        return 1.0 - self.nu

    @property
    def behavioural_noise(self) -> float:
        """Share of runs where the set of called tools differs from the first
        run of the same prompt. Text noise is one thing; a different set of
        calls on the same prompt is another, and much worse."""
        per_prompt = len(self.tool_sets) // len(self.prompts)
        diffs = total = 0
        for k in range(len(self.prompts)):
            runs = self.tool_sets[k * per_prompt:(k + 1) * per_prompt]
            diffs += sum(set(r) != set(runs[0]) for r in runs[1:])
            total += len(runs) - 1
        return diffs / total

    def __str__(self) -> str:
        return (f"noise floor  nu = {self.nu:.4f} ± {self.sd:.4f}   "
                f"({len(self.samples)} comparisons, {len(self.prompts)} prompts, {self.model})\n"
                f"bias       1 - nu = {self.bias:.4f}\n"
                f"behavioural noise: {self.behavioural_noise:.1%} of runs with a different tool set")

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> "NoiseFloor":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def measure_noise_floor(agent_shap, prompts: list[str], *, n_runs: int = 10) -> NoiseFloor:
    """n_runs generations with the full tool set for each prompt.

    Cost: len(prompts) * n_runs agent calls.
    """
    sims: list[float] = []
    tool_sets: list[list[str]] = []
    for prompt in prompts:
        responses = []
        for _ in range(n_runs):
            text, usage = agent_shap._execute_with_tools(prompt, agent_shap.tools)
            responses.append(text)
            tool_sets.append(sorted(k for k, v in usage.items() if v > 0))
        vectors = agent_shap.vectorizer.vectorize(responses) # vectorized answers 

        # similarity between answers, computed here as
        # "similarity of an answer to one random baseline".
        sims += [float(s) for s in agent_shap.vectorizer.calculate_similarity(vectors[0], vectors[1:])] 


    return NoiseFloor(nu=mean(sims), sd=stdev(sims), samples=sims, tool_sets=tool_sets,
                      prompts=list(prompts), model=agent_shap.model.model_name)

# What vectorize and calculate_similarity are in AGENTSHAP:

# def vectorize(self, texts):
#     self.vectorizer = TfidfVectorizer().fit(texts)      # TF-IDF is fitted on the texts passed in
#     return self.vectorizer.transform(texts).toarray()

# def calculate_similarity(self, base_vector, comparison_vectors):
#     return cosine_similarity(...)                        # sklearn, cosine between TF-IDF vectors
