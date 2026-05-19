---
name: python-testing-pytest-asyncio
description: Conventions for tests in echo-sdk — pytest + pytest-asyncio, async fixtures, dotenv loading, mocking provider SDKs. Use when adding or modifying tests under tests/.
---

# Testing

Stack: `pytest`, `pytest-asyncio`, `python-dotenv`. Setup lives in `tests/conftest.py`.

## Rules

- **Mark async tests** with `@pytest.mark.asyncio` (or set the asyncio mode in conftest if it's already configured — check there first).
- **Async fixtures**: `@pytest.fixture` + `async def` returning the value, or `@pytest_asyncio.fixture` for explicit async fixtures.
- **Don't hit real providers in CI.** Mock provider SDKs (`unittest.mock.patch`, or pass an injected client to the constructor).
- **`.env` loaded in `conftest.py`**; tests assume env is available, but secrets should not be required for unit tests — use mocks.
- **Test the agentic loop end-to-end** when touching providers — `invoke()` with a tool that returns deterministically, assert the final context has the expected `ToolCall` + `ToolResult` + assistant message.
- **Test schemas with `model_validate`**, not by constructing fields manually — catches Pydantic v2 regressions.
- **Async resource cleanup in tests**: use `async with` or `try/finally` so a failing test doesn't leak subprocesses (MCP stdio especially).

## Existing patterns

- `tests/test_llm_response.py` — provider response shape.
- `tests/test_prompts.py` — prompt provider behavior.
- `tests/test_postgres_binder.py` — binder type mapping.

## Common mistakes

- Forgetting `@pytest.mark.asyncio` → test silently doesn't await.
- Hitting `bedrock.client(...)` in CI → mock it.
- Leaving MCP stdio managers open after a test → flakes from subprocess leaks.
- Asserting on full LLM response strings → brittle; assert on structure (tool calls, message roles).

## See also

- `[[python-async-discipline]]`, `[[echo-sdk-llm]]`
