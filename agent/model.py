"""
Модель агента: локальная Ollama через OpenAI-совместимый endpoint.

Настоящий ключ не нужен — Ollama принимает любую строку.
"""
from __future__ import annotations

OLLAMA_URL = "http://localhost:11434"

# ~1.9 ГБ, поддерживает tool calling, помещается в 4 ГБ.
MODEL_NAME = "qwen2.5:3b"


def build_model(api, *, url: str = OLLAMA_URL, model_name: str = MODEL_NAME):
    """OpenAIModel из TokenSHAP, указывающий на локальную Ollama."""
    return api.OpenAIModel(model_name=model_name, api_key="ollama", base_url=f"{url}/v1")
