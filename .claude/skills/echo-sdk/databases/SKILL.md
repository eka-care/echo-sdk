---
name: echo-sdk-databases
description: Postgres support — asyncpg client, schema-aware binder, PgQueryTool surface. Use when querying postgres from a tool, modifying the binder, or adding a new DB engine.
---

# Databases

## What you're working with

- `src/echo/databases/postgres/`
  - `client.py` — async client around `asyncpg`.
  - `binder.py` — schema → Python type mapping; converts query results to typed records.
  - `config.py` — `PostgresConfig` (DSN, pool size, timeouts).
- `PgQueryTool` exposed via `echo.tools` — wraps the client as a `BaseTool`.

## Rules

- **`asyncpg` is async-native.** Always `await`. Never use `psycopg2` in this SDK.
- **Pool sizing is the caller's responsibility.** Default pool is conservative; configure via `PostgresConfig` for high-throughput hosts.
- **Close the client.** Use `async with` or call `await client.close()` in `finally` — leaked connections accumulate.
- **Binder is schema-driven.** When the table schema changes, regenerate the binder mapping; don't hand-map types in callers.
- **Parameterize all queries.** Never string-format user input into SQL — `asyncpg` uses `$1`, `$2`, ...
- **Optional dep.** `asyncpg` is an extra (`echo[postgres]`). Guard imports.

## `PgQueryTool` for agents

- Wraps a `PostgresClient` and exposes a JSON-schema query interface to the LLM.
- The query parameters and any tenant scoping should come via `system_context.tool_context` (e.g., `workspace_id`) — not in the LLM-visible schema.
- See `examples/` and `tests/test_postgres_binder.py`.

## Adding a new DB engine (future)

- Add a sibling package under `databases/` (e.g., `mysql/`).
- Mirror the `client / binder / config` triad.
- Wrap as a `BaseTool` only if agent-callable.
- New optional-deps extra in `pyproject.toml`.

## Common mistakes

- **Opening one pool per request** → expensive; reuse one pool per process.
- **Building dynamic SQL with f-strings** → SQL injection; use bound parameters.
- **Returning raw asyncpg `Record`** to the LLM → use the binder, return plain dicts.
- **Blocking I/O inside a tool that wraps a query** → keep the whole path async.

## See also

- `[[echo-sdk-tools]]`, `[[python-async-discipline]]`, `[[python-memory-safety]]`
