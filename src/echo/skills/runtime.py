"""Runtime configuration for skills and helpers to materialize them."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from echo.prompts.factory import get_prompt_provider
from echo.prompts.schemas import PromptConfig

from .skill import Skill

logger = logging.getLogger(__name__)


class SkillRuntimeConfig(BaseModel):
    """A skill attached to an agent. Resolved into an ``echo.skills.Skill`` at run time.

    Carries the prompt descriptor (resolved into the skill's ``instructions``)
    plus the flat tool-name list that the host has already validated against
    the agent's tool registry.
    """

    name: str
    description: str = ""
    prompt_config: PromptConfig
    tool_names: list[str] = Field(default_factory=list)


async def build_skills_from_runtime(
    configs: Optional[List[SkillRuntimeConfig]],
    provider=None,
) -> Optional[List[Skill]]:
    """Fetch each skill's prompt in parallel and map to ``Skill`` instances.

    A skill whose prompt fetch fails is dropped (logged); the caller gets
    the surviving skills rather than an exception, so one bad config does
    not break the whole turn.

    Args:
        configs: Runtime skill configs. ``None`` / empty returns ``None``.
        provider: Prompt provider. Defaults to ``get_prompt_provider()``.

    Returns:
        List of ``Skill`` instances, or ``None`` if no skills survived.
    """
    if not configs:
        return None

    provider = provider or get_prompt_provider()
    prompts = await asyncio.gather(
        *(
            provider.get_prompt(
                name=s.prompt_config.name,
                version=s.prompt_config.version,
                prompt_variables=s.prompt_config.prompt_variables,
            )
            for s in configs
        ),
        return_exceptions=True,
    )

    skills: list[Skill] = []
    for s, prompt in zip(configs, prompts):
        if isinstance(prompt, Exception):
            logger.error(
                "Skill prompt fetch failed; dropping skill name=%s error=%s",
                s.name,
                prompt,
            )
            continue
        skills.append(
            Skill(
                name=s.name,
                description=s.description,
                instructions=prompt.agent_prompt.task.description,
                tool_names=s.tool_names,
            )
        )
    return skills or None
