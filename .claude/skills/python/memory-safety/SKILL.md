---
name: python-memory-safety
description: Avoiding memory leaks and resource leaks in echo-sdk — close async resources, generators over lists, avoid retain cycles. Use when touching anything that opens connections, spawns subprocesses, or holds references long-term.
---

# Memory & Resource Safety

The leaks that actually bite in this SDK are **resource leaks**, not Python heap leaks. The async runtime keeps stdio subprocesses, HTTP connections, and DB pools alive long after you "forgot" about them.

## Rules

### Always close async resources

- `httpx.AsyncClient` → `async with httpx.AsyncClient() as client:`.
- `asyncpg` pool → `async with pool.acquire()`; pool itself: `await pool.close()` at shutdown.
- `MCPConnectionManager` → `async with MCPConnectionManager(...) as mgr:` or `await mgr.close_all()` in `finally`.
- File handles → `async with aiofiles.open(...)`.

A resource opened inside an exception path that isn't in `finally` will leak on error.

### Prefer streaming over buffering

- For large LLM outputs / audio / DB results, use `invoke_stream` and `async for` rather than collecting into a list.
- Generators release memory between yields; lists hold everything.
- `AsyncGenerator` callers that break out early **must** `aclose()` — use `async with aclosing(gen)` from `contextlib`.

### Watch for retain cycles

- Storing callbacks that close over `self` on long-lived registries (a tool registry holding lambdas that capture the agent) → consider `weakref` if the registry outlives the agent.
- `ConversationContext` holds references to every message ever; for long sessions, persist + prune on the host side.

### Don't leak `asyncio.Task`

- `asyncio.create_task(coro)` without keeping a reference → the task can be GC'd mid-run; even worse, if you do keep references in a set, you must remove them on done. Use `TaskGroup` (Python 3.11+) when possible.

### Pool/connection sizing

- One asyncpg pool per process, not per request.
- One MCP manager per agent run, not per tool call.
- One `httpx.AsyncClient` per service, not per call.

## Common mistakes

- `await client.connect(); ... return result` without `close()` in `finally` → leak on exception.
- Building `list(async_gen)` for a huge LLM stream → buffers full response in memory.
- `create_task(...)` and dropping the reference → "Task was destroyed but it is pending!" warning + possible silent loss.
- New `httpx.AsyncClient()` per request → connection-pool churn, slow.

## See also

- `[[python-async-discipline]]`, `[[echo-sdk-tools]]`, `[[echo-sdk-databases]]`
