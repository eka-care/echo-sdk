"""Static prompt templates bundled with the SDK.

Unlike `BasePromptProvider` (which fetches dynamic AgentConfigs from external
providers like Langfuse), templates here are framework-internal text fragments
loaded synchronously from YAML files co-located in this package.
"""

from functools import lru_cache
from pathlib import Path

import yaml

_TEMPLATES_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_template(name: str) -> str:
    """Return the `content` field of templates/{name}.yaml.

    Cached per-name so the YAML is parsed once per process.
    """
    path = _TEMPLATES_DIR / f"{name}.yaml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data["content"]


__all__ = ["load_template"]
