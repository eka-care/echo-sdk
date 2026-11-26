"""BaseLLM.invoke_tool: an unknown tool name tells the model the real names."""

from typing import Any, Dict

from echo.llm.base import BaseLLM
from echo.models.user_conversation import ToolCall
from echo.tools.core import BaseTool


class _StubTool(BaseTool):
    def __init__(self, name: str):
        self.name = name
        self.description = "stub"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> Any:
        return "ok"


class _StubLLM(BaseLLM):
    async def invoke(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


async def test_unknown_tool_lists_available_names() -> None:
    llm = _StubLLM.__new__(_StubLLM)
    tool_map = {
        name: _StubTool(name)
        for name in ("mcp_server_150__doctor_availability_elicitation", "search_doctors")
    }
    call = ToolCall(
        tool_id="t1", tool_name="doctor_availability_elicitation", tool_input={}
    )

    result = await llm.invoke_tool(tool_map, call, {})

    assert result.tool_id == "t1"
    assert "'doctor_availability_elicitation'" in result.result
    assert "mcp_server_150__doctor_availability_elicitation" in result.result
    assert "search_doctors" in result.result
    assert "don't use this tool" not in result.result
    assert "Retry" not in result.result
