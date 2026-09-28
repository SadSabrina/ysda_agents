"""
Агент, на котором всё разбирается: локальная Ollama + четыре инструмента
+ AgentSHAP поверх.

Перенесено из agentshap_tutorial_rus.ipynb (XAI-open_materials).

    from agent import build_agent_shap
    bundle = build_agent_shap()
    results, shapley = bundle.agent_shap.analyze(prompt, sampling_ratio=0.0)
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import MODEL_NAME, OLLAMA_URL, build_model
from .noise import NoiseFloor, measure_noise_floor
from .tokenshap import HARDCODED_TEMPERATURE, load_tokenshap
from .tools import TOOL_NAMES, build_tools

__all__ = [
    "AgentBundle", "build_agent_shap", "MODEL_NAME", "OLLAMA_URL",
    "HARDCODED_TEMPERATURE", "TOOL_NAMES", "NoiseFloor", "measure_noise_floor",
]


@dataclass
class AgentBundle:
    model: object
    tools: list
    agent_shap: object

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


def build_agent_shap(*, model_name: str = MODEL_NAME, url: str = OLLAMA_URL,
                     max_iterations: int = 5) -> AgentBundle:
    """Библиотека, модель, инструменты и AgentSHAP одним вызовом."""
    api = load_tokenshap()
    model = build_model(api, url=url, model_name=model_name)
    tools = build_tools(api.create_function_tool)
    agent_shap = api.AgentSHAP(model=model, tools=tools,
                               vectorizer=api.TfidfTextVectorizer(),
                               max_iterations=max_iterations)
    return AgentBundle(model=model, tools=tools, agent_shap=agent_shap)
