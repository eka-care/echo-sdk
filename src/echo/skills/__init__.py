"""Skills for Echo SDK.

Houses the `Skill` definition, runtime construction, and the LLM-driven
activation meta-tools (`LoadSkillTool`, `UnloadSkillTool`) that `BaseAgent`
auto-injects when configured with `skill_activation="llm"`. The meta-tools
live here (with the skill domain) rather than under `tools/`, and import the
tool framework from `echo.tools`.
"""

from .meta_tools import LoadSkillTool, UnloadSkillTool
from .runtime import SkillRuntimeConfig, build_skills_from_runtime
from .skill import Skill

__all__ = [
    "Skill",
    "SkillRuntimeConfig",
    "build_skills_from_runtime",
    "LoadSkillTool",
    "UnloadSkillTool",
]
