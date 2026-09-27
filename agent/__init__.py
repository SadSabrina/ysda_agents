"""
Агент, на котором всё разбирается: локальная Ollama + четыре инструмента
+ AgentSHAP поверх.

Перенесено сюда из agentshap_tutorial_rus.ipynb (XAI-open_materials),
чтобы ноутбуки лекции импортировали настройку, а не повторяли её.

    from agent import build_agent_shap
    bundle = build_agent_shap()
    results, shapley = bundle.agent_shap.analyze(prompt, sampling_ratio=0.0)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model import (CANDIDATE_URLS, MODEL_NAME, OLLAMA_URL, build_model,
                    check_ollama, find_ollama, raw_completion)
from .noise import NoiseFloor, measure_noise_floor, measure_noise_floor_multi
from .tokenshap import (DEFAULT_PATH, HARDCODED_TEMPERATURE, TokenShapAPI,
                        ensure_tokenshap, load_tokenshap)
from .tools import TOOL_NAMES, build_tools

__all__ = [
    "AgentBundle", "build_agent_shap",
    "OLLAMA_URL", "CANDIDATE_URLS", "MODEL_NAME", "build_model", "check_ollama",
    "find_ollama", "raw_completion",
    "load_tokenshap", "ensure_tokenshap", "TokenShapAPI", "HARDCODED_TEMPERATURE",
    "build_tools", "TOOL_NAMES",
    "measure_noise_floor", "measure_noise_floor_multi", "NoiseFloor",
]


@dataclass
class AgentBundle:
    api: TokenShapAPI
    model: object
    tools: list
    agent_shap: object
    vectorizer: object

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


def build_agent_shap(*, tokenshap_path: Path | str | None = None,
                     model_name: str = MODEL_NAME, url: str | None = None,
                     max_iterations: int = 5, debug: bool = False,
                     check: bool = True) -> AgentBundle:
    """Собрать всё разом: библиотека, модель, инструменты, AgentSHAP.

    url=None — искать Ollama автоматически среди типовых адресов.
    """
    if url is None:
        status = find_ollama(model_name)
        print(status["message"])
        if not status["ok"]:
            raise RuntimeError(status["message"])
        url = status["url"]
    elif check:
        status = check_ollama(url, model_name)
        print(status["message"])
        if not status["ok"]:
            raise RuntimeError(status["message"])

    api = load_tokenshap(tokenshap_path or DEFAULT_PATH)
    model = build_model(api, url=url, model_name=model_name)
    tools = build_tools(api.create_function_tool)
    vectorizer = api.TfidfTextVectorizer()

    agent_shap = api.AgentSHAP(
        model=model,
        tools=tools,
        vectorizer=vectorizer,
        max_iterations=max_iterations,
        debug=debug,
    )
    return AgentBundle(api=api, model=model, tools=tools,
                       agent_shap=agent_shap, vectorizer=vectorizer)
