"""AG-UI request schemas extending what ag_ui.core provides."""

from typing import Any

from pydantic import BaseModel


class AgUiResumeInput(BaseModel):
    """Body of a /resume call, forwarded to AgUiAgent.resume_stream()."""

    thread_id: str
    run_id: str
    tool_call_id: str
    tool_result: Any
