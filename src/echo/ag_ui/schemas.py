"""
AG-UI request schemas extending what ag_ui.core provides.

For the run entry point, callers should pass `ag_ui.core.RunAgentInput`
directly — there's no value in re-defining it here.

For resume, AG-UI doesn't standardize a body shape (resume is a runtime
concern, not a protocol concern), so we define one compatible with the
BaseAgent.resume_run_with_ag_ui() API.
"""

from typing import Any, List, Optional

from pydantic import BaseModel


class AgUiResumeInput(BaseModel):
    """Body of a /resume call.

    The host endpoint validates this on input and forwards to
    BaseAgent.resume_run_with_ag_ui(). The host is also responsible for
    rehydrating ConversationContext and AgUiState from the saved
    PausedRun before invoking resume.
    """

    thread_id: str
    run_id: str
    tool_call_id: str
    tool_result: Any
    # FE re-declares UI tools on resume so the agent knows what's still
    # available. Optional; if omitted, no UI tools are registered (the
    # next agent turn will treat all tool calls as backend).
    ui_tool_names: Optional[List[str]] = None
