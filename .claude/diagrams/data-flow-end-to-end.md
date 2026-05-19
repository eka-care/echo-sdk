# End-to-End Data Flow

One full turn from user input to assistant response, showing every layer involved.

```mermaid
sequenceDiagram
    participant U as User / Host
    participant A as GenericAgent
    participant L as BaseLLM (provider)
    participant T as Tool (BaseTool / MCPTool / PgQueryTool)
    participant E as External (provider API / MCP / Postgres)

    U->>A: run(context, out_msg_id)
    A->>A: _build_system_prompt() + _build_active_tools()
    A->>L: invoke(context, tools, system_prompt)
    L->>E: provider API call (with tool schemas)
    E-->>L: response with tool_calls
    loop agentic loop
        L->>L: parse tool_call
        L->>L: merge system_context.tool_context into args
        L->>T: await tool.run(**args)
        T->>E: side effect (HTTP / SQL / MCP)
        E-->>T: result
        T-->>L: output
        L->>L: append ToolCall + ToolResult to context
        L->>E: next provider call
        E-->>L: next response
    end
    L-->>A: (LLMResponse, updated_context)
    A-->>U: AgentResult(llm_response, context, agent_name)
```

## Streaming variant

Replace `invoke` with `invoke_stream` and `AgentResult` with `AsyncGenerator[StreamEvent, None]`. Events: `TEXT`, `TOOL_CALL_START`, `TOOL_CALL_END`, `DONE`, `ERROR`. The terminal `DONE` event carries the complete `LLMResponse` and final `ConversationContext`.

## Where things can go wrong

| Layer | Failure mode | Surface |
|-------|--------------|---------|
| Agent | Skill tool_names unresolved | dropped at __init__ with warning |
| LLM | Provider API error | exception propagates; agent returns `AgentResult(error=...)` |
| Tool | Schema mismatch | Provider rejects call before reaching tool.run |
| MCP tool | Connection drop | `MCPConnectionError`; wrap in try/except in tool.run |
| Postgres | Pool exhausted | asyncpg error; caller's responsibility to size pool |

See `src/echo/agents/base.py:_run_agent` and `_run_agent_stream`.
