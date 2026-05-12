"""AG-UI request schemas extending what ag_ui.core provides."""

from typing import Any, List, Optional

from pydantic import BaseModel


class AgUiResumeInput(BaseModel):
    """Body of a /resume call, forwarded to BaseAgent.resume_run_with_ag_ui()."""

    thread_id: str
    run_id: str
    tool_call_id: str
    tool_result: Any
    ui_tool_names: Optional[List[str]] = None
