---
name: echo-sdk-llm
description: LLM provider abstraction — get_llm factory, BaseLLM contract, agentic tool-calling loop, streaming, thinking config. Use when editing src/echo/llm/ or debugging provider behavior.
---

# LLM

## What you're working with

- `get_llm(LLMConfig) -> BaseLLM` in `factory.py` — single entry point.
- `BaseLLM` in `base.py` — declares `invoke()` and `invoke_stream()`.
- Providers: `anthropic.py`, `openai.py`, `bedrock.py`, `gemini.py` — each implements its own agentic loop and tool-calling semantics.
- `LLMConfig`, `ThinkingConfig`, `ReasoningEffort`, `GeminiThinkingLevel` in `config.py`.
- `LLMResponse`, `StreamEvent`, `StreamEventType`, `VerboseResponseItem` in `schemas.py`.

## Rules

- **Always go through `get_llm()`.** Never instantiate `AnthropicLLM` etc. directly from callers — the factory handles optional deps and config validation.
- **`invoke()` returns `(LLMResponse, updated_context)`.** The context is a **new** object — callers must use the returned one for the next turn.
- **`invoke_stream()` yields `StreamEvent`** of types `TEXT`, `TOOL_CALL_START`, `TOOL_CALL_END`, `DONE`, `ERROR`. The `DONE` event carries the complete response and final context.
- **Tool context injection happens here.** Each provider's loop reads `context.system_context.tool_context` and merges it into tool-call kwargs before invoking `tool.run(**args)`. Keep this consistent across providers.
- **Thinking config**: supported by Anthropic and Gemini (and Bedrock-Anthropic via inference profile). Surface it via `LLMConfig.thinking_config`; providers that don't support it must silently ignore.
- **Schema conversion**: use `tool.to_<provider>_schema()` — never roll your own. Gemini in particular requires the flattened schema (see `BaseTool._flatten_schema`).

## Provider quirks (must-know)

- **Bedrock**: Converse API. Tools wrapped as `{toolSpec: {name, description, inputSchema: {json: ...}}}`.
- **OpenAI**: tools wrapped as `{type: "function", function: {...}}`.
- **Anthropic**: native `tool_use` / `tool_result` blocks.
- **Gemini**: no `$ref` / `$defs` / `examples` / `default` / `title` / `additionalProperties` in schemas — `BaseTool` flattens before sending.
- **Streaming tool calls**: args arrive incrementally. Each provider's stream loop buffers until the tool_call is complete, then executes.

## Adding a new provider

→ Use `[[echo-sdk-adding-a-provider]]` skill. Summary: new file `providers.py`-style, subclass `BaseLLM`, register in `factory.py`, add optional-deps extra in `pyproject.toml`, never hard-import.

## Common mistakes

- **Returning the input context unchanged** → always append new messages to a copy.
- **Calling `tool.run(**llm_args)` directly without merging `tool_context`** → use the existing helper in the provider's loop.
- **Hard-coding model IDs** → use `LLMConfig`.
- **Catching `Exception` and swallowing** → let it propagate to the agent layer, which converts it to `AgentResult(error=...)` / `StreamEvent(ERROR)`.

## See also

- Diagram: `.claude/diagrams/llm-invoke-flow.md`
- Sibling: `[[echo-sdk-tools]]` (for schema conversion), `[[echo-sdk-models]]` (for context shape)
- `[[echo-sdk-adding-a-provider]]` when adding Cohere / Mistral / etc.
