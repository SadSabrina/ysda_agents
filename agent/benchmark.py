"""
Разбор уже прогнанного бенчмарка: 300 промптов, по одному прогону на каждый.

Перенесено из agentshap_tutorial_rus.ipynb. Прогон стоил времени и вызовов
модели; в репозитории лежит его результат целиком (`data/benchmark_results.csv`),
включая полный текст ответов, — чтобы переанализировать без обращения к модели.

Главный сюжет здесь не про агента, а про метрику: наивное сравнение строк
даёт 142/300, сравнение множеств — 295/300. Из 158 «ошибок» настоящих
оказывается пять. Разница целиком в том, как написано сравнение.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "benchmark_results.csv"


def load_runs(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Прочитать результаты прогона. Пустая строка = ни один инструмент не вызван."""
    return pd.read_csv(path).fillna("")


def _to_set(s: str) -> frozenset:
    return frozenset(s.split("+")) if s else frozenset()


@dataclass
class MatchAnalysis:
    n: int
    naive_string_match: int
    set_match: int
    order_artifact: int
    mismatches: pd.DataFrame

    @property
    def naive_accuracy(self) -> float:
        return self.naive_string_match / self.n

    @property
    def true_accuracy(self) -> float:
        return self.set_match / self.n

    def __str__(self) -> str:
        fake = self.set_match - self.naive_string_match
        return (
            f"промптов: {self.n}\n"
            f"наивное сравнение строк: {self.naive_string_match}/{self.n} "
            f"({self.naive_accuracy:.1%})\n"
            f"сравнение множеств:      {self.set_match}/{self.n} "
            f"({self.true_accuracy:.1%})\n"
            f"артефакт порядка:        {self.order_artifact}\n"
            f"настоящих расхождений:   {len(self.mismatches)}\n\n"
            f"Из {self.n - self.naive_string_match} «ошибок» настоящих — "
            f"{len(self.mismatches)}. Остальные {fake} создала метрика, а не агент."
        )


def match_analysis(runs_df: pd.DataFrame) -> MatchAnalysis:
    """Сравнить ожидаемые и вызванные инструменты тремя способами.

    Наивно строки сравнивать нельзя: "a+b" и "b+a" — один и тот же набор,
    а строки разные. Это и даёт 153 ложных «ошибки» из 158.
    """
    rel = runs_df["relevant_tools"].apply(_to_set)
    cal = runs_df["called_tools"].apply(_to_set)

    naive = int((runs_df["relevant_tools"] == runs_df["called_tools"]).sum())
    same = rel == cal
    order_artifact = int((same & (runs_df["relevant_tools"] != runs_df["called_tools"])).sum())

    return MatchAnalysis(
        n=len(runs_df),
        naive_string_match=naive,
        set_match=int(same.sum()),
        order_artifact=order_artifact,
        mismatches=runs_df.loc[~same, ["prompt", "relevant_tools", "called_tools"]].copy(),
    )


def tool_frequency_summary(runs_df: pd.DataFrame, tool_names: list[str]) -> pd.DataFrame:
    """Recall и false positive rate по каждому инструменту.

    Каждый инструмент ожидается в 130 промптах из 300 и не ожидается в 170 —
    этого хватает, чтобы отличить настоящий низкий base rate от шума.
    На десяти промптах одна ошибка сразу даёт 10-25 %.
    """
    rel = runs_df["relevant_tools"].apply(_to_set)
    cal = runs_df["called_tools"].apply(_to_set)

    rows = {}
    for tool in tool_names:
        expected = rel.apply(lambda s, t=tool: t in s)
        called = cal.apply(lambda s, t=tool: t in s)
        n_exp, n_not = int(expected.sum()), int((~expected).sum())
        rows[tool] = {
            "recall": (expected & called).sum() / n_exp if n_exp else float("nan"),
            "false_positive_rate": (~expected & called).sum() / n_not if n_not else float("nan"),
            "n_expected": n_exp,
            "n_not_expected": n_not,
        }
    return pd.DataFrame(rows).T.sort_values("false_positive_rate", ascending=False)


def classify_mismatches(mismatches: pd.DataFrame) -> pd.DataFrame:
    """Разложить настоящие расхождения на два типа.

    silent_miss   — инструмент ожидался, но не вызван ни один
    spurious_call — не ожидался ни один, но агент что-то вызвал

    Второй тип — это та же паразитная активность, которую AgentSHAP
    потом припишет ненулевым вкладом: аксиома dummy нарушается
    и в поведении агента, и в его объяснении.
    """
    out = mismatches.copy()

    def kind(row):
        want, got = row["relevant_tools"], row["called_tools"]
        if want and not got:
            return "silent_miss"
        if got and not want:
            return "spurious_call"
        return "wrong_set"

    out["тип"] = out.apply(kind, axis=1)
    return out
