"""
Loading TokenSHAP (AgentSHAP lives in it too).

The repository is cloned next to ours on first run: this is the only place
where the lecture materials reach outside their own repository. The library
code is deliberately not copied here: we analyse exactly what the authors
published.

We import directly from the .py files, bypassing the package __init__.py: it
pulls in torchvision and other heavy things. token_shap/base.py imports torch
at module level, but only the HuggingFaceModel class needs it, and we do not
use that class, so an empty module stands in for torch during the import.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://github.com/GenAISHAP/TokenSHAP.git"
REPO_COMMIT = "28b2f2c"          # version on which all lecture numbers were checked
PATH = Path(__file__).resolve().parent.parent / "TokenSHAP"

# Temperature hardcoded in the library: base.py, OpenAIModel.generate() and
# generate_with_tools(). It cannot be changed, hence the unreachable baseline.
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

    # sklearn and scipy check on import whether torch is loaded, and crash on
    # an empty module. So they are imported before the stub, and the stub
    # itself lives in sys.modules only while base.py is being imported
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
