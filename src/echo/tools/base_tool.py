"""
Base tool interface for Echo SDK.

Provides a framework-agnostic interface for tools with adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    """
    Abstract base class for Echo tools.

    Provides framework-agnostic tool definition with adapters for
    different execution frameworks (CrewAI, LangChain, etc.).
    """

    name: str = ""
    description: str = ""

    @property
    def is_elicitation(self) -> bool:
        """
        Whether the tool is an elicitation tool.
        """
        return False

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """
        Get the input schema for this tool.

        Override in subclasses to define custom parameters.

        Returns:
            Dict with type, properties, and required fields
        """
        pass

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """
        Execute the tool's core logic asynchronously.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            str: Tool output
        """
        pass

    def to_dict(self) -> Dict[str, str]:
        """Get tool metadata as dict."""
        return {
            "name": self.name,
            "description": self.description,
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """
        Get tool schema for Anthropic direct API.

        Returns:
            Dict with name, description, and input_schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Get tool schema for OpenAI.

        Returns:
            Dict with type: function wrapper and parameters
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_gemini_schema(self) -> Dict[str, Any]:
        """
        Get tool schema for Google Gemini.

        Returns:
            Dict with name, description, and parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

    def to_bedrock_schema(self) -> Dict[str, Any]:
        """
        Get tool schema for AWS Bedrock Converse API.

        Returns:
            Dict with toolSpec containing name, description, and inputSchema.json
        """
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": self.input_schema},
            }
        }

    def to_crewai_tool(self) -> Any:
        """
        Convert to CrewAI BaseTool.

        Returns:
            CrewAI-compatible tool instance

        Raises:
            ImportError: If crewai is not installed
        """
        try:
            from crewai.tools import BaseTool as CrewAIBaseTool
            from pydantic import Field
        except ImportError:
            raise ImportError(
                "crewai is required for to_crewai_tool(). "
                "Install with: pip install crewai"
            )

        # Capture self for closure
        tool = self

        class WrappedTool(CrewAIBaseTool):
            name: str = tool.name
            description: str = tool.description
            # Define input field for CrewAI
            query: str = Field(default="", description="The user query to search for")

            def _run(self, query: str = "", **kwargs) -> str:
                return tool.run(query=query, **kwargs)

        return WrappedTool()

    def to_langchain_tool(self) -> Any:
        """
        Convert to LangChain Tool.

        Returns:
            LangChain-compatible tool instance

        Raises:
            ImportError: If langchain is not installed
        """
        try:
            from langchain.tools import Tool
        except ImportError:
            raise ImportError(
                "langchain is required for to_langchain_tool(). "
                "Install with: pip install langchain"
            )

        return Tool(
            name=self.name,
            description=self.description,
            func=lambda **kwargs: self.run(**kwargs),
        )
