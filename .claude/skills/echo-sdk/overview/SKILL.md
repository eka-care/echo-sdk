---
name: echo-sdk-overview
description: Router for any non-trivial echo-sdk task. Maps user intent to the right module skill and diagram. Invoke FIRST for any change to src/echo/ before reaching for a specific module skill.
---

# Echo SDK — Overview & Router

Echo is a framework-agnostic Python SDK for LLM agents. Multi-provider, async-first, Pydantic v2. Each submodule has a tight responsibility — pick the right one before editing.

## Module map

| Module | Public entry point | Skill | Diagram |
|--------|-------------------|-------|---------|
| `agents/` | `from echo.agents import BaseAgent, GenericAgent, Skill, AgentConfig` | `echo-sdk/agents` | `agent-skill-lifecycle.md` |
| `llm/` | `from echo.llm import get_llm, LLMConfig, LLMResponse, StreamEvent` | `echo-sdk/llm` | `llm-invoke-flow.md` |
| `tools/` | `from echo.tools import BaseTool, MCPConnectionManager, MCPTool, PgQueryTool` | `echo-sdk/tools` | `tools-mcp-connection.md` |
| `models/` | `from echo.models import ConversationContext, Message, ToolCall, ToolResult` | `echo-sdk/models` | `conversation-context-er.md` |
| `prompts/` | `from echo.prompts import get_prompt_provider, FetchedPrompt` | `echo-sdk/prompts` | — |
| `evals/` | `from echo.evals import get_eval_provider` | `echo-sdk/evals` | — |
| `audio/` | `from echo.audio import get_transcriber, AudioInput` | `echo-sdk/audio` | — |
| `databases/` | `from echo.tools import PgQueryTool` (or `echo.databases.postgres.*`) | `echo-sdk/databases` | — |
| `utils/` | helpers; minimal | — | — |

For the big picture: `.claude/diagrams/architecture-overview.md` and `.claude/diagrams/data-flow-end-to-end.md`.

## Decision tree

| Task | Route to |
|------|----------|
| Add a new LLM provider | `echo-sdk/adding-a-provider` → then `echo-sdk/llm` |
| Add a new transcriber / eval / prompt provider | `echo-sdk/adding-a-provider` → then module skill |
| Build a new agent | `echo-sdk/agents` |
| Attach a skill to an agent | `echo-sdk/adding-a-skill-to-an-agent` |
| Write a new tool | `echo-sdk/tools` (+ `echo-sdk/models` for tool_context) |
| Wrap an MCP server | `echo-sdk/tools` + diagram `tools-mcp-connection.md` |
| Modify conversation state shape | `echo-sdk/models` — careful, immutability is load-bearing |
| Fetch prompts from Langfuse | `echo-sdk/prompts` |
| Run an experiment / eval | `echo-sdk/evals` |
| Query Postgres from a tool | `echo-sdk/databases` |
| Transcribe audio | `echo-sdk/audio` |

## Always also apply

These python skills apply to **every** task in this repo — load them when relevant:

- `python/async-discipline` — no blocking calls in async code
- `python/orjson-only` — never stdlib `json`
- `python/optional-deps` — guard provider imports
- `python/pydantic-v2` — config/schemas
- `python/memory-safety` — close async resources (MCP managers, asyncpg pools, httpx clients)

## Hard invariants (also in CLAUDE.md, do not violate)

1. `ConversationContext` is immutable from the caller — return updated, don't mutate.
2. New providers go behind the existing `get_*()` factory; never wire concrete classes into callers.
3. Optional deps stay behind `try/except ImportError` with a clear install hint.
4. `orjson` not `json`.
5. Async-first; streaming via `AsyncGenerator[StreamEvent]`.
6. Tool context (`system_context.tool_context`) is injected by the LLM layer — tools receive these as kwargs; the LLM never sees them in `input_schema`.

## Anti-patterns specific to this SDK

- Pasting concrete provider classes into `BaseAgent` subclasses → use `get_llm(LLMConfig(...))`.
- Adding `import boto3` at module top → wrap in optional-deps guard.
- Mutating `context.messages` inside a tool → tools return values; the LLM layer appends.
- Re-implementing tool schema conversion → use `BaseTool.to_<provider>_schema()`.
- Hardcoding `user_id` in tool args → put it in `system_context.tool_context`, it's injected.
