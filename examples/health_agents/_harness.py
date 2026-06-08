from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from echo.llm import LLMConfig

_FIXTURES = Path(__file__).parent / "fixtures"

#: Doctor template (user prompt) used across the examples.
DEFAULT_TEMPLATE = "default_clinical_template"


def load_transcript(name: str = "sample_transcript.txt") -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8").strip()


def example_llm_config() -> LLMConfig:
    return LLMConfig()


def banner(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def rule(label: str = "") -> None:
    if label:
        print(f"--- {label} " + "-" * max(0, 67 - len(label)))
    else:
        print("-" * 72)


def describe_config(cfg: LLMConfig) -> None:
    print(f"provider={cfg.provider}  model={cfg.model}  temperature={cfg.temperature}")
    print()


def short(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + " …"


def env_token(var: str = "EKA_BEARER_TOKEN") -> str | None:
    token = os.getenv(var)
    return token.strip() if token else None
