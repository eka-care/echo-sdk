"""
MCP (Model Context Protocol) tool wrapper for Echo SDK.

Wraps tools discovered from MCP servers behind the BaseTool interface,
enabling seamless use with all LLM providers and framework adapters.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from .base_tool import BaseTool
from .schemas import MCPExecutionError

if TYPE_CHECKING:
    from .mcp_connection_manager import MCPConnectionManager


class MCPTool(BaseTool):
    """
    Wrapper for tools discovered from an MCP server.
    Uses manager delegation for automatic reconnection and retry logic.
    """

    def __init__(
        self,
        manager: "MCPConnectionManager",
        server_id: str,
        tool_name: str,
        tool_description: str,
        input_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an MCP tool wrapper.

        Args:
            manager: MCPConnectionManager for executing tools with retry
            server_id: Unique identifier for the MCP server
            tool_name: Name of the tool from MCP server
            tool_description: Description of the tool from MCP server
            input_schema: JSON schema for tool inputs from MCP server.
                         If None, defaults to a generic query schema.
        """
        self._manager = manager
        self._server_id = server_id
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
        Execute the MCP tool asynchronously via manager with automatic retry.

        Args:
            **kwargs: Tool arguments to pass to the MCP server

        Returns:
            str: Tool result from MCP server
        """
        try:
            result = await self._manager.execute_tool(
                tool_name=self.name, arguments=kwargs
            )

            # Parse result (MCP returns content as a list of content blocks)
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

        except MCPExecutionError as e:
            return f"Tool execution failed: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
