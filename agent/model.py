"""
Agent model: local Ollama through an OpenAI-compatible endpoint.

No real key is needed: Ollama accepts any string.
"""
from __future__ import annotations

OLLAMA_URL = "http://localhost:11434"

# ~1.9 GB, supports tool calling, fits in 4 GB.
MODEL_NAME = "qwen2.5:3b"


def build_model(api, *, url: str = OLLAMA_URL, model_name: str = MODEL_NAME):
    """OpenAIModel from TokenSHAP, pointed at the local Ollama."""
    return api.OpenAIModel(model_name=model_name, api_key="ollama", base_url=f"{url}/v1")
