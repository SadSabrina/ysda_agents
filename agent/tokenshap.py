"""
Загрузка TokenSHAP (в нём же живёт AgentSHAP).

Репозиторий клонируется рядом при первом запуске: это единственное место,
где материалы лекции ходят за пределы своего репозитория. Код библиотеки
нарочно не копируется к нам — разбирается ровно то, что лежит у авторов.

Импортируем напрямую из .py-файлов, минуя пакетный __init__.py: он тянет
torchvision и прочее тяжёлое. token_shap/base.py импортирует torch на уровне
модуля, но нужен он только классу HuggingFaceModel, которым мы не пользуемся,
поэтому на время импорта вместо torch подставляется пустой модуль.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://github.com/GenAISHAP/TokenSHAP.git"
REPO_COMMIT = "28b2f2c"          # версия, на которой проверены все числа лекции
PATH = Path(__file__).resolve().parent.parent / "TokenSHAP"

# Температура, зашитая в библиотеке: base.py, OpenAIModel.generate() и
# generate_with_tools(). Изменить её нельзя, отсюда недостижимый baseline.
HARDCODED_TEMPERATURE = 0.5


@dataclass
class TokenShapAPI:
    OpenAIModel: type
    TfidfTextVectorizer: type
    AgentSHAP: type
    create_function_tool: object


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tokenshap() -> TokenShapAPI:
    if not PATH.exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(PATH)], check=True)
        subprocess.run(["git", "-C", str(PATH), "checkout", "-q", REPO_COMMIT], check=True)

    # sklearn и scipy при импорте проверяют, загружен ли torch, и на пустом
    # модуле падают. Поэтому они импортируются до заглушки, а сама заглушка
    # живёт в sys.modules только на время импорта base.py
    import sklearn.feature_extraction.text  # noqa: F401
    import sklearn.metrics.pairwise  # noqa: F401
    sys.modules["torch"] = types.ModuleType("torch")
    base = _load("token_shap.base", PATH / "token_shap" / "base.py")
    del sys.modules["torch"]
    tools = _load("token_shap.tools", PATH / "token_shap" / "tools.py")
    agent = _load("token_shap.agent_shap", PATH / "token_shap" / "agent_shap.py")
    return TokenShapAPI(
        OpenAIModel=base.OpenAIModel,
        TfidfTextVectorizer=base.TfidfTextVectorizer,
        AgentSHAP=agent.AgentSHAP,
        create_function_tool=tools.create_function_tool,
    )
