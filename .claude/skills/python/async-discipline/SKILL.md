---
name: python-async-discipline
description: Async/await rules for echo-sdk — never block the loop, cancellation safety, asyncio.gather over sequential awaits. Use any time you write or modify async code in src/echo/.
---

# Async Discipline

Echo is async-first. Sync code in an async path is a bug — it stalls every other in-flight request on the same loop.

## Rules

- **Never call blocking I/O in an async function.** Use the async client (`httpx.AsyncClient`, `asyncpg`, `aiofiles`). If you must run sync code, wrap with `await asyncio.to_thread(fn, *args)`.
- **`time.sleep` is forbidden** in async paths. Use `await asyncio.sleep(...)`.
- **Parallelize independent awaits with `asyncio.gather`** — don't sequentially `await` two independent calls.
- **Always close async resources.** `async with` for `httpx.AsyncClient`, `asyncpg.create_pool`, `MCPConnectionManager`. If you can't use `async with`, put `await x.close()` in `finally`.
- **Cancellation propagates.** Don't swallow `asyncio.CancelledError` — re-raise after cleanup. Wrap teardown in `try/finally`, not `try/except`.
- **No `asyncio.run` inside library code.** It belongs in entrypoints (examples/, tests/, host apps).
- **Async generators**: callers must `aclose()` if they stop iterating early. Prefer `async for` with `async with` of a contextmanager that owns the generator.
- **Don't mix threads and asyncio carelessly.** Use `asyncio.to_thread` (preferred) or `loop.run_in_executor` — don't spin up bare `threading.Thread`.

## Red flags in PR review

- `requests.get(...)` in an async function → use `httpx.AsyncClient`.
- `time.sleep` → `await asyncio.sleep`.
- `for x in items: result = await foo(x)` where each call is independent → `asyncio.gather(*(foo(x) for x in items))`.
- Bare `try/except Exception: pass` around an async call → swallowing cancellation.
- `asyncio.run(...)` inside `src/echo/` → only in entrypoints.

## See also

- `[[python-memory-safety]]` (resource leaks), `[[echo-sdk-tools]]` (tool.run is async).
