"""Client tool category.

A client tool runs in the user's page (a note editor, ...), not on the server.
The host app owns the transport and passes it to every run as
``tool_context["call_client_tool"]``. `BaseClientTool` is the category root for
tools whose schema is written in code; `ClientTool` wraps a declaration the
page sends at run time.
"""

from .base_client_tool import BaseClientTool, CallClientTool, ClientTool

__all__ = ["BaseClientTool", "CallClientTool", "ClientTool"]
