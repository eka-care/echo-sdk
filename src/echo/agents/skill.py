"""
Skill primitive for Echo SDK.

A Skill is a lazy-loaded bundle of (instructions + tools + description) that
can be dynamically attached to or detached from an agent mid-conversation.
The agent maintains an "active set" of skills and recomposes its system
prompt and tool list each turn from the active set.

Activation modes (set on the agent, not the skill):
- "manual": host application calls agent.activate_skill(name) / deactivate_skill(name).
- "llm": agent auto-injects load_skill / unload_skill meta-tools so the LLM
  decides when to load and unload. (PR2; not implemented in PR1.)

The primitive is policy-neutral: there is no `kind` or `additive` flag. The
LLM (or host) decides whether loading skill X should also unload skill Y by
the verbs it uses. Swap is just unload(old) + load(new) — providers that
support parallel tool calls let the LLM express atomic swap in one turn.
"""

from typing import TYPE_CHECKING, List, Optional

from echo.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from echo.models.user_conversation import ConversationContext


class Skill:
    """A bundle of (instructions + tools + description) attachable to an agent.

    Direct instantiation is the 90% path. Subclass only if you need lifecycle
    hooks (on_activate / on_deactivate) — for example, to fetch a tenant-scoped
    auth token at activation time or log skill switches to telemetry.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        if not name:
            raise ValueError("Skill.name is required")
        if not description:
            raise ValueError("Skill.description is required")
        if not instructions:
            raise ValueError("Skill.instructions is required")

        self.name = name
        self.description = description
        self.instructions = instructions
        self.tools: List[BaseTool] = list(tools or [])

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
        return (
            f"Skill(name={self.name!r}, "
            f"tools={[t.name for t in self.tools]!r})"
        )
