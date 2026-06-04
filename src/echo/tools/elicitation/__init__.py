"""Elicitation tool category.

`BaseElicitationTool` is the category root for tools that collect structured
user input (always PAUSE/VISIBLE). The elicitation *payload* schemas
(`ElicitationDetails`/`Response`/...) live in `tools/core` because they are
shared with the MCP wrapper and the LLM providers.
"""

from .base_elicitation_tool import BaseElicitationTool

__all__ = ["BaseElicitationTool"]
