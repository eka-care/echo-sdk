"""File-based prompt provider (on-prem default — plan B2/B4).

Resolution for prompt ``name`` under ``ECHO_PROMPT_DIR`` (default ./prompts),
with ``/`` in names flattened to ``_`` (same convention as voice2rx's checked-in
.md fallbacks, so one folder serves both):

  1. {dir}/{flat}/{version}.yaml        — versioned layout; ``production`` file
     in the folder names the default version (falls back to highest version)
  2. {dir}/{flat}.yaml                  — flat YAML
  3. {dir}/{flat}.md                    — plain markdown; whole file is the task

YAML fields: ``prompt`` (required), ``role``/``goal``/``backstory`` (persona),
``expected_output``, ``version``.

Variable compilation reproduces Langfuse ``{{var}}`` semantics: ONLY ``{{key}}``
placeholders are replaced; every other brace (JSON examples in prompt bodies)
is left untouched. ``str.format`` would be wrong here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from echo.prompts.schemas import AgentPrompt, PromptPersona, PromptTask
from echo.prompts.base import BasePromptProvider, FetchedPrompt, PromptFetchError


def compile_langfuse_style(raw: str, variables: Optional[Dict[str, Any]] = None) -> str:
    """Replace {{key}} placeholders only — literal braces survive."""
    result = raw
    for key, value in (variables or {}).items():
        result = result.replace("{{" + str(key) + "}}", "" if value is None else str(value))
    return result


class FilePromptProvider(BasePromptProvider):
    def __init__(self, prompt_dir: Optional[str] = None):
        self.prompt_dir = Path(
            prompt_dir or os.getenv("ECHO_PROMPT_DIR", "./prompts")
        ).resolve()

    def _flat(self, name: str) -> str:
        return name.replace("/", "_")

    def _resolve(self, name: str, version: Optional[str]) -> Tuple[Path, Optional[str]]:
        flat = self._flat(name)
        versioned_dir = self.prompt_dir / flat
        if versioned_dir.is_dir():
            if version:
                path = versioned_dir / f"{version}.yaml"
                if path.is_file():
                    return path, version
                raise PromptFetchError(f"Prompt '{name}' version {version} not found")
            pointer = versioned_dir / "production"
            if pointer.is_file():
                v = pointer.read_text().strip()
                path = versioned_dir / f"{v}.yaml"
                if path.is_file():
                    return path, v
            versions = sorted(
                (p for p in versioned_dir.glob("*.yaml")),
                key=lambda p: (len(p.stem), p.stem),
            )
            if versions:
                return versions[-1], versions[-1].stem
            raise PromptFetchError(f"Prompt folder '{flat}/' has no versions")
        flat_yaml = self.prompt_dir / f"{flat}.yaml"
        if flat_yaml.is_file():
            return flat_yaml, None
        flat_md = self.prompt_dir / f"{flat}.md"
        if flat_md.is_file():
            return flat_md, None
        raise PromptFetchError(
            f"Prompt '{name}' not found under {self.prompt_dir} "
            f"(tried {flat}/, {flat}.yaml, {flat}.md)"
        )

    async def get_prompt(
        self,
        name: str,
        version: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FetchedPrompt:
        try:
            path, resolved_version = self._resolve(name, version)
            raw = path.read_text(encoding="utf-8")

            if path.suffix == ".yaml":
                data = yaml.safe_load(raw) or {}
                body = data.get("prompt", "")
                task_description = compile_langfuse_style(body, prompt_variables)
                agent_prompt = AgentPrompt(
                    persona=PromptPersona(
                        role=data.get("role"),
                        goal=data.get("goal"),
                        backstory=data.get("backstory"),
                    ),
                    task=PromptTask(
                        description=task_description,
                        expected_output=data.get("expected_output"),
                    ),
                )
                resolved_version = resolved_version or (
                    str(data["version"]) if data.get("version") else None
                )
            else:  # .md — whole file is the task description
                task_description = compile_langfuse_style(raw, prompt_variables)
                agent_prompt = AgentPrompt(
                    persona=PromptPersona(),
                    task=PromptTask(description=task_description),
                )

            return FetchedPrompt(
                name=name, version=resolved_version, agent_prompt=agent_prompt
            )
        except PromptFetchError:
            raise
        except Exception as e:
            raise PromptFetchError(f"Failed to fetch '{name}': {e}")
