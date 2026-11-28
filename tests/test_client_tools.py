"""Client tools: the page runs the tool; the host passes the transport."""

from typing import Any, Dict

import orjson

from echo.tools.client import BaseClientTool, ClientTool
from echo.tools.client import base_client_tool
from echo.tools.core.schemas import Observability

DECL = {
    "name": "read_note",
    "description": "Read the note.",
    "inputSchema": {"type": "object", "properties": {"section": {"type": "string"}}},
}


class Page:
    def __init__(self, reply: Dict[str, Any]):
        self.reply = reply
        self.calls = []

    async def __call__(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((name, arguments))
        return self.reply


async def _run(reply: Dict[str, Any], **kwargs) -> str:
    return await ClientTool(DECL).run(tool_context={"call_client_tool": Page(reply)}, **kwargs)


def test_declaration_becomes_the_tool():
    tool = ClientTool({**DECL, "outputSchema": {"type": "object"}})
    assert tool.name == "read_note"
    assert tool.input_schema == DECL["inputSchema"]
    assert tool.description.endswith('Returns JSON matching this schema: {"type":"object"}')
    assert tool.observability == Observability.SILENT
    assert tool.meta == {"tool_class": "client_tool"}


async def test_call_goes_to_the_page_without_tool_context():
    page = Page({"result": {"content": [{"type": "text", "text": "hi"}]}})
    result = await ClientTool(DECL).run(section="rx", tool_context={"call_client_tool": page, "user_id": "u1"})
    assert result == "hi"
    assert page.calls == [("read_note", {"section": "rx"})]


async def test_structured_content_wins():
    reply = {"result": {"structuredContent": {"ok": 1}, "isError": True, "content": []}}
    assert orjson.loads(await _run(reply)) == {"isError": True, "ok": 1}


async def test_text_error_and_jsonrpc_error():
    assert await _run({"result": {"isError": True, "content": [{"type": "text", "text": "bad"}]}}) == "[tool error] bad"
    reply = {"error": {"code": "timeout", "message": "late"}}
    assert orjson.loads(await _run(reply)) == {"isError": True, "code": "timeout", "message": "late"}


async def test_no_transport_is_unreachable():
    for ctx in (None, {}):
        assert orjson.loads(await ClientTool(DECL).run(tool_context=ctx))["code"] == "client_unreachable"


async def test_large_result_is_refused(monkeypatch):
    monkeypatch.setattr(base_client_tool, "MAX_RESULT_BYTES", 4)
    reply = {"result": {"content": [{"type": "text", "text": "too long"}]}}
    assert orjson.loads(await _run(reply))["code"] == "result_too_large"


async def test_code_defined_client_tool():
    class EditNote(BaseClientTool):
        name = "edit_note"
        description = "Edit the note."

        @property
        def input_schema(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}}

    page = Page({"result": {"content": [{"type": "text", "text": "done"}]}})
    assert await EditNote().run(tool_context={"call_client_tool": page}) == "done"
    assert page.calls == [("edit_note", {})]
