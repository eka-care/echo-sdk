from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from echo.agents.config import AgentConfig

from .schemas import UserPrompt

_PROMPTS_DIR = Path(__file__).parent
_SYSTEM_DIR = _PROMPTS_DIR / "system_prompts"
_USER_DIR = _PROMPTS_DIR / "user_prompts"

_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _substitute(text: str, variables: Dict[str, Any]) -> str:
    if not text or not variables:
        return text

    def _repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        return str(variables[key]) if key in variables else match.group(0)

    return _VAR_PATTERN.sub(_repl, text)


class HealthPrompts:
    def __init__(
        self,
        system_dir: Optional[Path] = None,
        user_dir: Optional[Path] = None,
    ) -> None:
        self.system_dir = Path(system_dir) if system_dir else _SYSTEM_DIR
        self.user_dir = Path(user_dir) if user_dir else _USER_DIR

    def load_system_prompt(self, name: str) -> AgentConfig:
        data = self._read_yaml(self.system_dir / f"{name}.yaml")
        return AgentConfig(**data)

    def load_user_prompt(self, name: str) -> UserPrompt:
        data = self._read_yaml(self.user_dir / f"{name}.yaml")
        data.setdefault("name", name)
        return UserPrompt(**data)

    def compose(
        self,
        system: AgentConfig,
        user: Optional[UserPrompt] = None,
        **variables: Any,
    ) -> AgentConfig:
        merged = system.model_copy(deep=True)

        if user is not None and user.content.strip():
            merged.task.description = (
                f"{merged.task.description}\n\n"
                "## Template (use these section headings, in this order)\n\n"
                f"{user.content.strip()}"
            )

        persona = merged.persona
        persona.role = _substitute(persona.role or "", variables) or None
        persona.goal = _substitute(persona.goal or "", variables) or None
        persona.backstory = _substitute(persona.backstory or "", variables) or None
        merged.task.description = _substitute(merged.task.description, variables)
        if merged.task.expected_output:
            merged.task.expected_output = _substitute(
                merged.task.expected_output, variables
            )
        return merged

    def build_agent_config(
        self,
        system_name: str,
        user_name: Optional[str] = None,
        **variables: Any,
    ) -> AgentConfig:
        system = self.load_system_prompt(system_name)
        user = self.load_user_prompt(user_name) if user_name else None
        return self.compose(system, user, **variables)

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Prompt file must be a YAML mapping: {path}")
        return data


@lru_cache(maxsize=1)
def _default_store() -> HealthPrompts:
    return HealthPrompts()


def load_system_prompt(name: str) -> AgentConfig:
    return _default_store().load_system_prompt(name)


def load_user_prompt(name: str) -> UserPrompt:
    return _default_store().load_user_prompt(name)


def build_agent_config(
    system_name: str,
    user_name: Optional[str] = None,
    **variables: Any,
) -> AgentConfig:
    return _default_store().build_agent_config(system_name, user_name, **variables)
