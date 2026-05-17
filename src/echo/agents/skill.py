"""
Skill primitive for Echo SDK.

A Skill is a lazy-loaded bundle of (instructions + tool name references +
description) that can be dynamically attached to or detached from an agent
mid-conversation. The agent maintains an "active set" of skills and
recomposes its system prompt and tool list each turn from the active set.

Tools live in a single name-keyed registry on the agent. A skill does not
own tool instances — it references them by name (`tool_names`). When the
skill activates, the agent looks up those names in its registry; visibility
is computed as a name-set union across base + active skills + meta tools,
so a tool referenced by two active skills appears exactly once.

Activation modes (set on the agent, not the skill):
- "manual": host application calls agent.activate_skill(name) / deactivate_skill(name).
- "llm": agent auto-injects load_skill / unload_skill meta-tools so the LLM
  decides when to load and unload.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from echo.models.user_conversation import ConversationContext


class Skill:
    """A bundle of (instructions + tool name references + description)
    attachable to an agent.

    Direct instantiation is the 90% path. Subclass only if you need lifecycle
    hooks (on_activate / on_deactivate) — for example, to fetch a tenant-scoped
    auth token at activation time or log skill switches to telemetry.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        tool_names: Optional[List[str]] = None,
    ) -> None:
        if not name:
            raise ValueError("Skill.name is required")
        if not description:
            raise ValueError("Skill.description is required")
        if not instructions:
            raise ValueError("Skill.instructions is required")

        if tool_names is not None:
            for tn in tool_names:
                if not isinstance(tn, str) or not tn:
                    raise ValueError(
                        f"Skill.tool_names must contain non-empty strings; got {tn!r}"
                    )

        self.name = name
        self.description = description
        self.instructions = instructions
        self.tool_names: List[str] = list(tool_names or [])

    async def on_activate(self, context: Optional["ConversationContext"]) -> None:
        """Called when the skill is added to the agent's active set.

        Override in a subclass to run setup logic (auth, telemetry, etc.).
        `context` is None when the skill is activated before any conversation
        has started (e.g., during agent setup); otherwise it is the live
        ConversationContext of the in-flight turn.
        """
        return None

    async def on_deactivate(self, context: Optional["ConversationContext"]) -> None:
        """Called when the skill is removed from the agent's active set.

        Override in a subclass to run teardown logic. Same `context` contract
        as on_activate.
        """
        return None

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, tool_names={self.tool_names!r})"
