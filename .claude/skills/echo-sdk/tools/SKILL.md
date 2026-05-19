---
name: echo-sdk-tools
description: Building tools — BaseTool contract, MCP tool wrapping, elicitation tools, schema conversion, tool_context injection. Use when adding a tool, debugging tool calls, or wiring an MCP server.
---

# Tools

## What you're working with

- `BaseTool` in `base_tool.py` — abstract; subclasses set `name`, `description`, `input_schema` (JSON Schema dict), implement `async run(**kwargs)`.
- `BaseElicitationTool` in `base_elicitation.py` — `is_elicitation == True`; signals the agent to suspend for user input.
- `MCPConnectionManager` + `MCPTool` — manage and wrap MCP server tools.
- `PgQueryTool` from `databases/` — postgres query tool (see `[[echo-sdk-databases]]`).
- Schemas in `schemas.py`: `MCPServerConfig`, `MCPTransport`, `MCPError` + subclasses, `ElicitationDetails`, `ElicitationStatus`, `ElicitationResponse`.

## Rules for a new tool

1. **Subclass `BaseTool`**, set `name` (unique across the agent's registry) and `description`.
2. **Define `input_schema`** as JSON Schema dict — `type: object`, `properties`, `required`. Keep it lean; Gemini strips unsupported fields automatically but simpler is better.
3. **`async def run(self, **kwargs)`** — never sync. Use the existing async clients (`asyncpg`, `httpx.AsyncClient`, MCP session).
4. **Don't put hidden params in `input_schema`** — they come via `system_context.tool_context` and are injected as kwargs to `run()`. Examples: `user_id`, `workspace_id`, `auth_token`.
5. **Return a JSON-serializable value** (dict / list / str). The LLM layer serializes via `orjson`.
6. **Errors**: raise; the agentic loop converts to a tool error result. For MCP, use `MCPError` subclasses.
7. **Schema conversion is provided**: never override `to_anthropic_schema` / `to_openai_schema` / `to_gemini_schema` / `to_bedrock_schema` — `BaseTool` handles all four.

## MCP wrapping

```python
async with MCPConnectionManager(configs=[MCPServerConfig(...)]) as mgr:
    tools = await mgr.list_tools()   # List[MCPTool]
    agent = GenericAgent(..., tools=tools)
    await agent.run(context, out_msg_id)
```

- Use `async with` or call `await mgr.close_all()` in `finally`. Leaked stdio subprocesses are a real resource leak.
- One manager per agent run; tools share its connections.
- Transports: `STDIO`, `SSE`, `HTTP` (see `MCPTransport`).

## Elicitation

When a tool needs user input mid-turn:
- Subclass `BaseElicitationTool` (sets `is_elicitation = True`).
- Return an `ElicitationResponse` with `status` and `details`.
- The agent suspends and surfaces to the host; host resumes with user-provided values.

## Common mistakes

- **Sync `run()`** → must be async, even if the work is sync (wrap blocking I/O in `asyncio.to_thread`).
- **`input_schema` with `$ref` / `$defs`** → works for most providers but Gemini strips them; prefer inline schemas.
- **Reusing tool `name` across the registry** → silently dropped with warning. Choose unique names.
- **Using `load_skill` / `unload_skill` as a tool name** → reserved, dropped.
- **Holding the MCP manager open for the whole process** → fine for long-lived hosts, but always `close_all()` on shutdown.

## See also

- Diagram: `.claude/diagrams/tools-mcp-connection.md`
- Diagram: `.claude/diagrams/conversation-context-er.md` (for `tool_context` shape)
- `[[echo-sdk-models]]`, `[[echo-sdk-databases]]`, `[[python-async-discipline]]`
