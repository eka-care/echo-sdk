"""System tools: echo-internal, loop-managed tool base.

`SystemTool` is intentionally NOT re-exported from `echo.tools` — it is not
part of the public extension surface (see its `__init_subclass__` guard).
"""

from .system_tool import SystemTool

__all__ = ["SystemTool"]
