---
name: python-packaging-uv-pyproject
description: uv + pyproject.toml conventions for echo-sdk — adding deps, extras, scripts; Python 3.11+. Use when editing pyproject.toml or running dep commands.
---

# Packaging: uv + pyproject

Build tool: `uv`. Python `>=3.11`. Lock file: `uv.lock` (committed).

## Rules

- **Add deps with `uv add <pkg>`** (modifies `pyproject.toml` + `uv.lock`). For dev/test: `uv add --dev <pkg>`.
- **Add an extra** by editing `[project.optional-dependencies]` directly:
  ```toml
  [project.optional-dependencies]
  cohere = ["cohere>=X.Y"]
  all = [..., "cohere>=X.Y"]   # always update umbrella
  ```
  Then `uv lock` to refresh.
- **Run tests / scripts** via `uv run pytest`, `uv run python examples/basic_usage.py` — `uv run` ensures the project env.
- **Version bumps**: edit `[project].version` in `pyproject.toml`. Tag the commit.
- **Use `>=`** for runtime deps unless a known incompat forces a cap. Avoid `==` outside lockfile.

## Repo specifics

- Core deps (always installed): `pydantic>=2.12.3`, `orjson`, `pyyaml`.
- Extras: `bedrock`, `openai`, `anthropic`, `gemini`, `postgres`, `mcp`, `langfuse`, `all`.
- Dev deps: `pytest`, `pytest-cov`, `pytest-asyncio`, `python-dotenv`.

## Common mistakes

- Editing `uv.lock` by hand → regenerate via `uv lock`.
- Adding to `[dependency-groups]` when it should be `[project.optional-dependencies]` (or vice versa) → check existing pattern in `pyproject.toml` first.
- Forgetting to update the `all` extra when adding a new provider extra.
- `pip install -e .` for dev → use `uv sync --all-extras` instead.

## See also

- `[[python-optional-deps]]`, `[[echo-sdk-adding-a-provider]]`
