---
name: generic-error-handling-at-boundaries
description: Validate at system boundaries, trust internally. No defensive fallbacks for impossible cases. Use whenever adding try/except, validation, or null-checks.
---

# Error Handling at Boundaries

## Rules

- **Validate at the boundary** (user input, external API responses, file/network I/O, MCP results, DB results). Inside, trust your own code.
- **Don't write `if x is None: ...` for values your own code just constructed** — that's noise.
- **Don't add fallbacks for scenarios that can't happen.** If a Pydantic model just validated, its fields are present; don't re-check.
- **Don't swallow exceptions** with `except Exception: pass`. Either handle a specific exception type with a meaningful recovery, or let it propagate.
- **Wrap external errors in domain errors** at the boundary — e.g., MCP raises → `MCPExecutionError` (already wrapped); HTTP 401 from a provider → bubble as the provider's documented exception type.
- **Re-raise `asyncio.CancelledError`** after cleanup; never absorb it.
- **Log at the boundary that recovered**, not at every layer that passed through. Logging "error" 5 times for one failure is noise.

## Why

- Defensive code for impossible cases reads like "I don't trust this codebase," which makes the next reader trust it less.
- Catching everything hides bugs that should crash loudly in dev.
- The right place to handle "the LLM API was unreachable" is the agent layer's `AgentResult(error=...)`, not five layers deep.

## Echo-specific

- Agent layer's `_run_agent` / `_run_agent_stream` are the boundary that converts exceptions to `AgentResult(error=...)` / `StreamEvent(ERROR)`. Tools, LLM providers, and skills should **raise** — don't catch.
- MCP tools wrap external errors in `MCPError` subclasses at the boundary with the MCP server.
- Postgres tools: bubble asyncpg errors; the agent layer catches.

## Common mistakes

- `try: result = await tool.run() except Exception as e: return None` → swallows real errors; tool failures become silent successes.
- `if config and config.foo and config.foo.bar` for Pydantic configs → if construction succeeded, the chain is fine.
- Catching `Exception` to log + re-raise → just let it propagate; the boundary logs.

## See also

- `[[generic-small-diffs]]`, `[[python-async-discipline]]`
