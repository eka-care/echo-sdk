"""
MCP (Model Context Protocol) tool wrapper for Echo SDK.

Wraps tools discovered from MCP servers behind the BaseTool interface,
enabling seamless use with all LLM providers and framework adapters.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base_tool import BaseTool
from .schemas import ElicitationDetails, MCPExecutionError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .mcp_connection_manager import MCPConnectionManager


class MCPTool(BaseTool):
    """
    Wrapper for tools discovered from an MCP server. Delegates execution
    to the manager, which opens a fresh MCP session per call.
    """

    def __init__(
        self,
        manager: "MCPConnectionManager",
        tool_name: str,
        tool_description: str,
        input_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an MCP tool wrapper.

        Args:
            manager: MCPConnectionManager for executing tools
            tool_name: Name of the tool from MCP server
            tool_description: Description of the tool from MCP server
            input_schema: JSON schema for tool inputs from MCP server.
                         If None, defaults to a generic query schema.
        """
        self._manager = manager
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

    async def run(
        self, tool_context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str | ElicitationDetails:
        """Execute the MCP tool asynchronously.

        tool_context is accepted for BaseTool interface compatibility but is
        not consulted; the manager opens a fresh MCP session per call.
        """
        try:
            meta = kwargs.pop("meta", {})
            result = await self._manager.execute_tool(
                tool_name=self.name,
                arguments=kwargs,
                meta=meta,
            )

            structured_content = (
                result.structuredContent if hasattr(result, "structuredContent") else {}
            )
            if structured_content and structured_content.get("is_elicitation"):
                elicitation_details = ElicitationDetails(**structured_content)
                return elicitation_details

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
            logger.error("MCP tool '%s' execution failed: %s", self.name, e)
            return f"Tool execution failed: {str(e)}"
        except Exception as e:
            logger.critical(
                "Unexpected error executing MCP tool '%s': %s",
                self.name, e, exc_info=True,
            )
            return f"Unexpected error: {str(e)}"
