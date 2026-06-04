"""Tool framework for Echo SDK.

Import policy (tools/): each subpackage owns and exposes its own public API;
this top-level package does NOT re-aggregate them. Import from the owning
subpackage:

- ``from echo.tools.core import BaseTool, ElicitationResponse, ...``
- ``from echo.tools.elicitation import BaseElicitationTool``
- ``from echo.tools.mcp import MCPTool, MCPConnectionManager, MCPServerConfig, ...``
- ``from echo.tools.system import SystemTool``   (echo-internal)

This keeps ``import echo.tools`` lean and dependency-light — notably it does
NOT drag in the optional ``mcp``/``httpx`` deps that ``echo.tools.mcp`` needs.

Concrete *domain* tools live with their domain, not here:
- skill tools  → ``echo.skills``
- postgres tool → ``echo.databases.postgres``

The single exception to "no re-aggregation" is ``BaseTool`` — the one
universal, dependency-free contract — re-exported here for convenience.
"""

from .core import BaseTool

__all__ = ["BaseTool"]
