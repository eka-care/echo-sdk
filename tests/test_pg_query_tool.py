"""Tests for PgQueryTool — mocks the PostgresClient."""

from unittest.mock import AsyncMock

import pytest

from echo.tools.databases import PgQueryTool


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.fetch_all = AsyncMock(return_value=[{"doctor_id": "doc_1"}])
    client.fetch_one = AsyncMock(return_value={"doctor_id": "doc_1"})
    return client


async def test_run_fetch_all_passes_params(mock_client):
    tool = PgQueryTool(
        client=mock_client,
        name="search_doctors",
        description="search",
        sql="SELECT * FROM datasets_doctor WHERE workspace_id = %(workspace_id)s",
        input_schema={"type": "object", "properties": {}, "required": []},
        fetch_mode="all",
    )
    rows = await tool.run(tool_context={"workspace_id": "ws_abc"})
    # default transform_params passes kwargs through; tool_context isn't injected
    # by the base class — that's the subclass's job. Here kwargs is empty.
    mock_client.fetch_all.assert_awaited_once()
    call_args, call_kwargs = mock_client.fetch_all.call_args
    assert call_args[0] == tool.sql
    assert call_kwargs["params"] == {}
    assert rows == [{"doctor_id": "doc_1"}]


async def test_fetch_one_routes_to_fetch_one(mock_client):
    tool = PgQueryTool(
        client=mock_client,
        name="get_doctor",
        description="lookup",
        sql="SELECT * FROM datasets_doctor WHERE doctor_id = %(doctor_id)s",
        fetch_mode="one",
    )
    result = await tool.run(doctor_id="doc_1")
    mock_client.fetch_one.assert_awaited_once()
    mock_client.fetch_all.assert_not_called()
    assert result == {"doctor_id": "doc_1"}


async def test_subclass_transform_params_is_honored(mock_client):
    class MyTool(PgQueryTool):
        name = "my_tool"
        description = "..."
        sql = "SELECT * FROM t WHERE workspace_id = %(workspace_id)s AND q = %(q)s"
        fetch_mode = "all"

        def transform_params(self, tool_context, **kwargs):
            return {
                "workspace_id": (tool_context or {}).get("workspace_id"),
                "q": kwargs.get("query_text"),
            }

    tool = MyTool(client=mock_client)
    await tool.run(query_text="hi", tool_context={"workspace_id": "ws_xyz"})
    _, call_kwargs = mock_client.fetch_all.call_args
    assert call_kwargs["params"] == {"workspace_id": "ws_xyz", "q": "hi"}


def test_input_schema_override():
    schema = {
        "type": "object",
        "properties": {"query_text": {"type": "string"}},
        "required": ["query_text"],
    }
    tool = PgQueryTool(
        client=AsyncMock(),
        name="t",
        description="d",
        sql="SELECT 1",
        input_schema=schema,
    )
    assert tool.input_schema == schema


def test_input_schema_default_is_empty_object():
    tool = PgQueryTool(client=AsyncMock(), name="t", description="d", sql="SELECT 1")
    assert tool.input_schema == {"type": "object", "properties": {}, "required": []}
