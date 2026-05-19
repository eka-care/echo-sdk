# Conversation Context — Entity Relationships

The data model for in-flight conversation state. **Immutable from the caller's perspective**: `invoke()` returns a new context; never mutate the input.

```mermaid
erDiagram
    ConversationContext ||--|{ Message : "messages[]"
    ConversationContext ||--|| SystemContext : "system_context"
    SystemContext ||--o{ ToolContext : "tool_context (dict)"

    Message ||--|| MessageRole : role
    Message ||--o{ Content : "content[] (polymorphic)"

    Content }|--|| TextMessage : variant
    Content }|--|| ImageContent : variant
    Content }|--|| DocumentContent : variant
    Content }|--|| ToolCall : variant
    Content }|--|| ToolResult : variant

    ToolCall ||--|| ToolResult : "paired by tool_call_id"

    ConversationContext {
        list messages
        SystemContext system_context
    }
    Message {
        MessageRole role
        list content
    }
    ToolCall {
        string tool_call_id
        string name
        dict args
    }
    ToolResult {
        string tool_call_id
        any output
        bool is_error
    }
```

## Rules

- **Roles**: `user`, `assistant`, `system`, `tool` (see `MessageRole`).
- **Content is polymorphic**: one Message can hold text + image + document + tool_call/result in `content[]`.
- **`tool_call_id` pairs `ToolCall` ↔ `ToolResult`**. Providers rely on this for tool-use turns.
- **`SystemContext.tool_context`** carries hidden params (user_id, workspace_id, auth tokens) — merged into tool args at execution time; **not visible to the LLM**.
- **Never serialize with stdlib json** — use `orjson`. Pydantic models: `.model_dump()` then `orjson.dumps()`.

See `src/echo/models/user_conversation.py`.
