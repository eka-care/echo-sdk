---
name: echo-sdk-adding-a-provider
description: Recipe for adding a new provider (LLM, transcriber, eval, prompt). Use whenever adding Cohere, Mistral, Deepgram, Phoenix, Promptlayer, etc. — anything that plugs into an existing get_*() factory.
---

# Adding a Provider

Every provider family in echo follows the same shape: `BaseX` abstract, concrete `XProvider` classes, `get_x()` factory, optional-deps extra. Don't deviate — the consistency is load-bearing for callers.

## Steps

1. **Pick the family.** LLM (`src/echo/llm/`), prompts (`src/echo/prompts/`), evals (`src/echo/evals/`), audio transcription (`src/echo/audio/transcription/`).
2. **Create the provider file**, e.g., `src/echo/llm/cohere.py`. Subclass the family's `BaseX`.
3. **Guard the optional import**:
   ```python
   try:
       import cohere
   except ImportError as e:
       raise ImportError(
           "cohere is required for CohereLLM. "
           "Install with: pip install 'echo[cohere]'"
       ) from e
   ```
   Do this **inside the class `__init__` or method**, not at module top. The factory must be importable without the optional dep installed.
4. **Implement the abstract methods.** For LLM: `invoke`, `invoke_stream`. Both async. Both must return/yield in the canonical shape (see `[[echo-sdk-llm]]`).
5. **Tool schema conversion** (LLM only): use `tool.to_<provider>_schema()` where applicable, or add a new converter on `BaseTool` if the provider needs a unique shape.
6. **Tool-context injection** (LLM only): merge `context.system_context.tool_context` into tool-call args before invoking `tool.run(**args)`. Match the existing providers exactly.
7. **Register in `factory.py`** — add a branch on the config field (e.g., `LLMConfig.provider`).
8. **Add the extra in `pyproject.toml`**:
   ```toml
   [project.optional-dependencies]
   cohere = ["cohere>=X.Y"]
   all = [..., "cohere>=X.Y"]
   ```
9. **Add an example** under `examples/`.
10. **Add a test** under `tests/`. Mock the external API; don't hit the wire in CI.
11. **Update `README.md`** install matrix.

## Rules

- **Never hard-import the optional dep.** The package must be importable without it.
- **Match the existing provider's error semantics** — exceptions propagate; the agent layer catches.
- **Don't change `BaseX`** to fit your provider. If the abstract is wrong, that's a separate discussion.
- **Streaming providers must yield the same `StreamEvent` types** — don't invent new ones without updating all consumers.
- **Test the agentic loop end-to-end**, not just `invoke()` — tool-call roundtrip is where providers diverge.

## Common mistakes

- Importing the optional dep at module top → breaks `get_x()` discovery for everyone.
- Returning a tuple shape that diverges from `(LLMResponse, updated_context)`.
- Forgetting to dedup `tool_call_id` pairing in streaming.
- Adding the extra under `[tool.uv]` instead of `[project.optional-dependencies]`.

## See also

- `[[echo-sdk-llm]]`, `[[echo-sdk-prompts]]`, `[[echo-sdk-evals]]`, `[[echo-sdk-audio]]`
- `[[python-optional-deps]]`, `[[python-packaging-uv-pyproject]]`
- Diagram: `.claude/diagrams/llm-invoke-flow.md`
