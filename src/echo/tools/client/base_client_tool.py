"""
Base client tool interface for Echo SDK.

Client tools are implemented by the user's page. The tool sends the call
through the host's ``call_client_tool`` and turns the page's JSON-RPC reply
(an MCP ``CallToolResult`` or an error) into the text the LLM reads.
"""

from typing import Any, Awaitable, Callable, Dict, Optional

import orjson

from ..core.base_tool import BaseTool
from ..core.schemas import Observability

# call_client_tool(tool_name, arguments) -> the page's JSON-RPC reply:
# {"result": CallToolResult} or {"error": {"code", "message", "data"?}}.
# Delivery failures and timeouts are the host's to report, as an error reply.
CallClientTool = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]

MAX_RESULT_BYTES = 262_144


def _dumps(value: Any) -> str:
    return orjson.dumps(value).decode("utf-8")


def _error(code: str, message: str) -> str:
    return _dumps({"isError": True, "code": code, "message": message})


def reply_to_llm_text(reply: Dict[str, Any]) -> str:
    """Turn the page's JSON-RPC reply into the tool result the LLM reads."""
    if "error" in reply:
        error = reply.get("error") or {}
        return _dumps(
            {
                "isError": True,
                "code": error.get("code"),
                "message": error.get("message"),
                **({"data": error["data"]} if "data" in error else {}),
            }
        )

    result = reply.get("result") or {}
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        body: Dict[str, Any] = dict(structured)
        if result.get("isError"):
            body = {"isError": True, **body}
        return _dumps(body)

    texts = [
        c.get("text", "")
        for c in result.get("content") or []
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    text = "\n".join(texts)
    return f"[tool error] {text}" if result.get("isError") else text


class BaseClientTool(BaseTool):
    """
    Base class for client tools.

    Subclasses set ``name``, ``description`` and ``input_schema``; ``run`` is
    the same for all of them. Silent toward the client: the page shows the
    status of its own tools.
    """

    observability = Observability.SILENT

    @property
    def meta(self) -> Dict[str, Any]:
        return {"tool_class": "client_tool"}

    async def run(
        self, tool_context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        call = (tool_context or {}).get("call_client_tool")
        if not callable(call):
            return _error(
                "client_unreachable", "The user's page is not connected."
            )
        text = reply_to_llm_text(await call(self.name, kwargs))
        if len(text.encode("utf-8")) > MAX_RESULT_BYTES:
            return _error(
                "result_too_large", "Result too large; ask for less data."
            )
        return text


class ClientTool(BaseClientTool):
    """A client tool built from the MCP ``Tool`` declaration the page sends."""

    def __init__(self, declaration: Dict[str, Any]):
        self.name = declaration["name"]
        description = declaration["description"]
        output_schema = declaration.get("outputSchema")
        if isinstance(output_schema, dict):
            description = (
                f"{description}\n\nReturns JSON matching this schema: "
                f"{_dumps(output_schema)}"
            )
        self.description = description
        self._input_schema = declaration["inputSchema"]

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._input_schema
