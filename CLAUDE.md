# Echo SDK — Claude Code Guide

Framework-agnostic Python SDK for building LLM agents. Multi-provider (Bedrock / OpenAI / Anthropic / Gemini), async-first, Pydantic v2, optional-deps per provider.

## How to work in this repo

For any non-trivial task, **invoke skills in this order** — narrowest wins:

1. **`echo-sdk/overview`** — router. Start here. It points you to the right module skill and diagram.
2. **`echo-sdk/<module>`** — agents / llm / tools / models / prompts / evals / audio / databases / adding-a-provider / adding-a-skill-to-an-agent.
3. **`python/<topic>`** — language-level rules (async, Pydantic v2, orjson, optional-deps, typing, memory-safety, testing, packaging).
4. **`generic/<topic>`** — small-diffs, no-premature-abstraction, error-handling-at-boundaries, comments-only-when-why.

Skills live under `.claude/skills/`. Diagrams (Mermaid) under `.claude/diagrams/` — skills link to them; don't load diagrams unless a skill points you there.

## Hard invariants (non-negotiable)

- **`orjson`, never stdlib `json`.** It's a dep, used everywhere.
- **Async-first.** All `invoke` / `run` are `async`. Streaming via `AsyncGenerator[StreamEvent]`.
- **Optional deps are guarded.** New provider code goes behind `try: import ... except ImportError` with a clear install hint. Never add a hard import for an optional dep.
- **Pydantic v2 everywhere** — `ConfigDict`, `model_validate`, `model_dump`. No v1 patterns.
- **`ConversationContext` is immutable.** `invoke()` returns `(LLMResponse, updated_context)`; never mutate the input.
- **Factory pattern for providers.** Add new LLM/transcriber/eval/prompt providers via the existing `get_*()` factory — never wire concrete classes into callers.
- **Never `git commit` without explicit user confirmation** for the specific commit, even mid-plan.

## Layout

```
src/echo/
  agents/    llm/     tools/    models/     prompts/
  evals/     audio/   databases/ knowledge/  utils/
```

Public API per module is in its `__init__.py` — read that first when extending.
