"""
MCP (Model Context Protocol) tool wrapper for Echo SDK.

Wraps tools discovered from MCP servers behind the BaseTool interface,
enabling seamless use with all LLM providers and framework adapters.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from .base_tool import BaseTool

if TYPE_CHECKING:
    from mcp import ClientSession


class MCPTool(BaseTool):
    """
    Wrapper for tools discovered from an MCP server.
    Takes an MCP tool definition and client session, exposing it as a
    standard BaseTool that works with all Echo LLM providers and adapters.
    """

    def __init__(
        self,
        session: "ClientSession",
        tool_name: str,
        tool_description: str,
        input_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an MCP tool wrapper.

        Args:
            session: Active MCP ClientSession for making tool calls
            tool_name: Name of the tool from MCP server
            tool_description: Description of the tool from MCP server
            input_schema: JSON schema for tool inputs from MCP server.
                         If None, defaults to a generic query schema.
        """
        self._session = session
        self.name = tool_name
        self.description = tool_description
        self._input_schema = input_schema or {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Get the input schema from the MCP tool definition.

        Returns:
            Dict with type, properties, and required fields
        """
        return self._input_schema

    async def run(self, **kwargs) -> str:
        """
        Execute the MCP tool asynchronously.

        Args:
            **kwargs: Tool arguments to pass to the MCP server

        Returns:
            str: Tool result from MCP server
        """
        result = await self._session.call_tool(self.name, arguments=kwargs)
        # MCP returns content as a list of content blocks
        if hasattr(result, "content") and result.content:
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
                else:
                    texts.append(str(block))
            return "\n".join(texts)
        return str(result)

    @property
    def session(self) -> "ClientSession":
        """Get the underlying MCP session."""
        return self._session
