---
name: echo-sdk-models
description: Conversation data model — ConversationContext, Message, ToolCall/ToolResult, polymorphic content. Use when reading/writing context, adding a new content type, or debugging tool-call pairing.
---

# Models

## What you're working with

- `ConversationContext` — root container; has `messages: list[Message]` and `system_context`.
- `Message` — has `role: MessageRole` and `content: list` (polymorphic).
- Content variants: `TextMessage`, `ImageContent`, `DocumentContent`, `ToolCall`, `ToolResult`.
- `MessageRole`: `user`, `assistant`, `system`, `tool`.
- `ContentSourceType` — for image/document provenance.
- `Provider` enum in `providers.py`.

Public API: `src/echo/models/__init__.py`. Implementation: `user_conversation.py`.

## Rules

- **Immutability discipline.** `invoke()` and `invoke_stream()` return a **new** `ConversationContext` — never mutate the input. Tools must not touch `context.messages`.
- **Tool-call pairing.** Each `ToolCall` has a `tool_call_id`; the matching `ToolResult` must reuse the same `tool_call_id`. Providers rely on this.
- **`system_context.tool_context`** is a `dict` of hidden params injected into tool kwargs by the LLM layer. **Not visible to the LLM.** Add new hidden params here, not to tool `input_schema`.
- **Polymorphic `content`**: a single message can carry text + image + document + tool calls/results. Order matters for some providers (Anthropic in particular).
- **Pydantic v2.** Use `model_validate`, `model_dump`. For serialization to JSON, `orjson.dumps(ctx.model_dump())` — never stdlib `json`.

## Adding a new content type

1. Define a new Pydantic model in `user_conversation.py` (or a new file imported there).
2. Add to the `Message.content` union (and any discriminator).
3. Update each provider's converter in `src/echo/llm/<provider>.py` to translate the new type to the provider's native shape — or skip it cleanly if the provider doesn't support it.
4. Add a test in `tests/`.

## Common mistakes

- **Appending directly to `context.messages` from a tool** → tools return values; the LLM layer appends.
- **Sharing one `ConversationContext` between two concurrent agent runs** → make a copy or use separate contexts.
- **Putting auth tokens in `Message.content`** → they go in `system_context.tool_context` (hidden) or LLM config.
- **Forgetting `tool_call_id` matching** → provider rejects the next turn.

## See also

- Diagram: `.claude/diagrams/conversation-context-er.md`
- `[[echo-sdk-tools]]`, `[[echo-sdk-llm]]`
