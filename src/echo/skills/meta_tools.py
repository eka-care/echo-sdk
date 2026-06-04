"""
Meta-tools for LLM-driven skill activation.

When `BaseAgent` is configured with `skill_activation="llm"` and one or more
skills are registered, the agent auto-injects a `load_skill` and an
`unload_skill` tool into the per-turn tool list. These are how the LLM itself
expresses "load this capability" / "unload this capability" — analogous to
the way it picks any other tool, but with the side effect of changing the
agent's active set on the next turn.

Each instance is bound to a specific `BaseAgent`. They translate LLM tool
calls into the agent's programmatic activation API (`activate_skill` /
`deactivate_skill`).

Note on hooks: meta-tool calls invoke the activation API with `context=None`.
A skill's `on_activate` / `on_deactivate` hooks therefore receive `None` for
context when activated via the LLM. Hosts that need access to the live
ConversationContext during activation should use programmatic (manual-mode)
activation instead.
"""

from typing import TYPE_CHECKING, Any, Dict

from echo.tools.core import BaseTool

if TYPE_CHECKING:
    from echo.agents.base import BaseAgent


class LoadSkillTool(BaseTool):
    """LLM-callable tool that adds a registered skill to the agent's active set."""

    name = "load_skill"
    description = (
        "Add one of the agent's registered skills to the active set so its "
        "instructions and tools become available on subsequent turns. Use "
        "this when the user's intent matches a skill listed in "
        "<available_skills>. To swap from one skill to another, call "
        "unload_skill on the outgoing skill in the same turn (in parallel) "
        "or beforehand."
    )

    def __init__(self, agent: "BaseAgent") -> None:
        self._agent = agent

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill's registered name. Must match a name "
                        "listed in <available_skills>."
                    ),
                }
            },
            "required": ["name"],
        }

    async def run(self, name: str = "", **kwargs: Any) -> Dict[str, Any]:
        if not name:
            return {
                "status": "error",
                "message": "Missing required argument: name",
            }
        try:
            await self._agent.activate_skill(name, context=None)
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        return {"status": "loaded", "skill": name}


class UnloadSkillTool(BaseTool):
    """LLM-callable tool that removes a skill from the agent's active set."""

    name = "unload_skill"
    description = (
        "Remove a skill from the active set. Its instructions and tools are "
        "no longer visible on subsequent turns. Use this when the user's "
        "intent shifts away from the active skill, or to make room before "
        "loading a different skill."
    )

    def __init__(self, agent: "BaseAgent") -> None:
        self._agent = agent

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill's registered name to unload.",
                }
            },
            "required": ["name"],
        }

    async def run(self, name: str = "", **kwargs: Any) -> Dict[str, Any]:
        if not name:
            return {
                "status": "error",
                "message": "Missing required argument: name",
            }
        await self._agent.deactivate_skill(name, context=None)
        return {"status": "unloaded", "skill": name}
