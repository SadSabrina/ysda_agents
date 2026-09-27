"""
Загрузка TokenSHAP (в нём же живёт AgentSHAP).

Импортируем напрямую из .py-файлов, минуя пакетный __init__.py: он тянет
torchvision и прочие тяжёлые зависимости, которые здесь не нужны.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://github.com/GenAISHAP/TokenSHAP.git"

# Корень репозитория, а не текущая папка: иначе клон появится внутри notebooks/
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "TokenSHAP"


@dataclass
class TokenShapAPI:
    """Классы, которые нам нужны из библиотеки."""

    OpenAIModel: type
    TfidfTextVectorizer: type
    AgentSHAP: type
    create_function_tool: object
    path: Path


def ensure_tokenshap(path: Path | str = DEFAULT_PATH, *, quiet: bool = True) -> Path:
    """Клонировать репозиторий, если его ещё нет рядом."""
    path = Path(path)
    if (path / "token_shap" / "base.py").exists():
        return path
    cmd = ["git", "clone", REPO_URL, str(path)]
    if quiet:
        cmd.append("-q")
    subprocess.run(cmd, check=True)
    return path


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"не удалось загрузить {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tokenshap(path: Path | str = DEFAULT_PATH, *, clone_if_missing: bool = True) -> TokenShapAPI:
    """Вернуть нужные классы TokenSHAP, минуя тяжёлый __init__."""
    path = Path(path)
    if clone_if_missing:
        path = ensure_tokenshap(path)
    if not (path / "token_shap" / "base.py").exists():
        raise FileNotFoundError(
            f"TokenSHAP не найден в {path.resolve()}. "
            f"Клонируйте: git clone {REPO_URL}"
        )

    try:
        base = _load("token_shap.base", path / "token_shap" / "base.py")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{exc.name} не установлен. token_shap/base.py импортирует его на уровне модуля, "
            f"поэтому обход __init__.py от этого не спасает. Поставьте: pip install {exc.name}"
        ) from exc
    tools_mod = _load("token_shap.tools", path / "token_shap" / "tools.py")
    agent_mod = _load("token_shap.agent_shap", path / "token_shap" / "agent_shap.py")

    return TokenShapAPI(
        OpenAIModel=base.OpenAIModel,
        TfidfTextVectorizer=base.TfidfTextVectorizer,
        AgentSHAP=agent_mod.AgentSHAP,
        create_function_tool=tools_mod.create_function_tool,
        path=path,
    )


# Температура, зашитая в библиотеке. Ни generate(), ни generate_with_tools()
# не дают её изменить, и OpenAIModel.__init__ её не принимает. Отсюда
# недостижимый baseline: два прогона с полным тулсетом не совпадают.
HARDCODED_TEMPERATURE = 0.5
