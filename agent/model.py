"""
Модель агента: локальная Ollama через OpenAI-совместимый endpoint.

Настоящий ключ не нужен — Ollama принимает любую строку. Это важно для
аудитории: ноутбуки запускаются без платного доступа куда бы то ни было.
"""
from __future__ import annotations

from typing import Any

import requests

OLLAMA_URL = "http://localhost:11434"

# Ollama слушает на localhost машины. Если код идёт из контейнера или VM,
# localhost там свой — нужен адрес хоста. Порядок перебора: сначала честный
# localhost, потом типовые имена хоста из контейнера.
CANDIDATE_URLS = [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
    "http://host.docker.internal:11434",
    "http://host.lima.internal:11434",
]

# ~1.9 ГБ, поддерживает tool calling, помещается в 4 ГБ.
# Альтернативы: "llama3.2:3b" (~2 ГБ), "qwen2.5" / "llama3.1" (7-8B, надёжнее
# на сложных вызовах инструментов, но и медленнее).
MODEL_NAME = "qwen2.5:3b"


def _probe(url: str, timeout: int = 3) -> list[str] | None:
    """Вернуть список моделей, если по этому адресу живая Ollama, иначе None.

    Проверяем именно код 200 и разбор JSON: прокси может ответить 403
    с текстом об allowlist, и без raise_for_status это выглядит как успех.
    """
    try:
        resp = requests.get(f"{url}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return None
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return None


def find_ollama(model_name: str = MODEL_NAME, *, timeout: int = 3) -> dict[str, Any]:
    """Найти живую Ollama среди типовых адресов."""
    tried = []
    for url in CANDIDATE_URLS:
        models = _probe(url, timeout)
        tried.append(url)
        if models is None:
            continue
        if not any(model_name in m for m in models):
            return {"ok": False, "url": url, "models": models,
                    "message": f"Ollama найдена на {url}, но модели {model_name!r} нет. "
                               f"Запустите: ollama pull {model_name}"}
        return {"ok": True, "url": url, "models": models,
                "message": f"Ollama на {url}, модель {model_name!r} готова"}

    return {"ok": False, "url": None, "models": [],
            "message": "Ollama не найдена ни по одному адресу: "
                       + ", ".join(tried)
                       + ". Запустите `ollama serve`. Если код идёт не с той же машины, "
                         "где стоит Ollama, доступа может не быть вовсе."}


def check_ollama(url: str = OLLAMA_URL, model_name: str = MODEL_NAME, *, timeout: int = 3) -> dict[str, Any]:
    """Проверить конкретный адрес. Для автопоиска используйте find_ollama()."""
    models = _probe(url, timeout)
    if models is None:
        return {"ok": False, "url": url, "models": [],
                "message": f"Ollama недоступна на {url}. Запустите: ollama serve"}
    if not any(model_name in m for m in models):
        return {"ok": False, "url": url, "models": models,
                "message": f"Модель {model_name!r} не скачана. Запустите: ollama pull {model_name}"}
    return {"ok": True, "url": url, "models": models,
            "message": f"Модель {model_name!r} готова"}


def build_model(api, *, url: str = OLLAMA_URL, model_name: str = MODEL_NAME):
    """Создать OpenAIModel, указывающий на локальную Ollama.

    api — результат load_tokenshap().
    """
    return api.OpenAIModel(model_name=model_name, api_key="ollama", base_url=f"{url}/v1")


def raw_completion(model, prompt: str, tools: list, *, model_name: str = MODEL_NAME,
                   temperature: float = 0.5):
    """Прямой вызов клиента в обход AgentSHAP и generate_with_tools.

    Библиотека выбрасывает message.content, как только распарсит tool_calls,
    а нам нужен именно он: текст, который модель написала до или вместе
    с вызовом инструмента.
    """
    resp = model.client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=[t.definition for t in tools],
        temperature=temperature,
    )
    return resp.choices[0].message
