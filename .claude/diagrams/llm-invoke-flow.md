# LLM Invoke Flow

How `BaseLLM.invoke()` and `invoke_stream()` execute the agentic loop with tool calls.

## Non-streaming

```mermaid
flowchart TD
    A[invoke context, tools, system_prompt] --> B[Convert tools to provider schema]
    B --> C[Send to provider API]
    C --> D{Response contains<br/>tool_calls?}
    D -- no --> E[Append text Message to context]
    E --> Z[Return LLMResponse, updated_context]
    D -- yes --> F[For each tool_call]
    F --> G[Inject tool_context<br/>from system_context]
    G --> H[await tool.run **args]
    H --> I[Append ToolCall + ToolResult to context]
    I --> C
```

## Streaming

```mermaid
flowchart TD
    A[invoke_stream] --> B[Open provider stream]
    B --> C{event type}
    C -- text delta --> D[yield StreamEvent TEXT]
    C -- tool_call start --> E[yield TOOL_CALL_START]
    E --> F[Buffer tool_call args]
    F --> G[yield TOOL_CALL_END<br/>after args complete]
    G --> H[Execute tool async]
    H --> I[Append ToolResult, continue loop]
    I --> B
    C -- done --> J[yield DONE with full LLMResponse + context]
    C -- error --> K[yield ERROR]
```

## Provider-specific notes

- **Anthropic / Gemini**: support thinking config via `ThinkingConfig`. Bedrock-Anthropic also via inference profile.
- **Bedrock**: uses Converse API; `toolSpec` envelope.
- **Gemini**: JSON schema is flattened (no `$ref`, `$defs`, `examples`, `default`, `title`, `additionalProperties`) — see `BaseTool._flatten_schema`.
- **OpenAI**: tools wrapped in `{type: function, function: {...}}`.

## Tool context injection

`ConversationContext.system_context.tool_context` is a dict of hidden params (e.g., `user_id`, `workspace_id`). Each provider's loop merges this into the tool-call args before invoking `tool.run()`. **The LLM never sees these params** — they don't appear in `input_schema`.

See `src/echo/llm/{anthropic,openai,bedrock,gemini}.py` for per-provider loops.
