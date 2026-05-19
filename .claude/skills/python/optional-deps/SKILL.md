---
name: python-optional-deps
description: How to add and guard optional dependencies (provider SDKs, asyncpg, langfuse, mcp, etc.) so the core package stays installable without extras. Use any time you import a provider SDK or DB driver.
---

# Optional Dependencies

Every provider/integration is an extra in echo-sdk: `echo[bedrock]`, `echo[openai]`, `echo[anthropic]`, `echo[gemini]`, `echo[postgres]`, `echo[mcp]`, `echo[langfuse]`, `echo[all]`. The core package must be importable without any extra installed.

## Rules

- **Never hard-import optional deps at module top.** Wrap in `try/except ImportError`:
  ```python
  try:
      import asyncpg
  except ImportError as e:
      raise ImportError(
          "asyncpg is required for PostgresClient. "
          "Install with: pip install 'echo[postgres]'"
      ) from e
  ```
- **Place the guard at the import site that needs it** — usually inside the class `__init__` or method, NOT at module top of any file that the factory needs to import for discovery.
- **The factory must be importable without the extra.** Provider-specific code lives in its own module (`llm/anthropic.py`); the factory imports it lazily when its branch is chosen.
- **Add the extra to `pyproject.toml`**:
  ```toml
  [project.optional-dependencies]
  cohere = ["cohere>=X.Y"]
  all = [..., "cohere>=X.Y"]   # always update the umbrella too
  ```
- **Install hint message** must be precise: name the extra (`'echo[cohere]'`) and the missing package.

## Why

- Users installing only `echo[anthropic]` shouldn't need `boto3` on disk.
- Tests for unrelated providers shouldn't fail because an unrelated dep isn't installed.
- CI matrices stay manageable.

## Common mistakes

- Hard-import at module top of `llm/cohere.py` → the file itself is fine since the factory imports it lazily; but **never** hard-import in `llm/__init__.py` or `llm/factory.py`.
- Forgetting to update the `all` extra → `pip install 'echo[all]'` becomes inconsistent.
- Bare `except ImportError: pass` → silently broken at runtime; **always** re-raise with an install hint.
- Pinning extras too narrowly → use `>=X.Y` not `==X.Y`.

## See also

- `[[echo-sdk-adding-a-provider]]`, `[[python-packaging-uv-pyproject]]`
