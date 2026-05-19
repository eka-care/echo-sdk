---
name: echo-sdk-prompts
description: External prompt-provider abstraction (Langfuse today). Use when fetching prompts at runtime, adding a new prompt provider, or wiring agent config from a managed prompt.
---

# Prompts

## What you're working with

- `BasePromptProvider` in `prompts/base.py` — abstract; declares `fetch(name, label=..., version=..., variables=...) -> FetchedPrompt`.
- `LangfuseProvider` in `prompts/langfuse_provider.py` — concrete implementation.
- `get_prompt_provider()` in `prompts/factory.py` — singleton factory with reset.
- `FetchedPrompt` — has compiled prompt string + ready-to-use `AgentConfig`.
- Local templates in `prompts/templates/` — `load_template(name)` used internally (e.g., `skill_mechanism`).

## Rules

- **Always go through the factory.** `get_prompt_provider()` returns the configured singleton; use `reset()` only in tests.
- **`FetchedPrompt.agent_config` is pre-compiled** — pass it straight into `GenericAgent(agent_config=...)`. Don't rebuild it.
- **Variable substitution** happens server-side or in the provider; pass `variables=` dict at fetch time.
- **Optional dep.** `langfuse` is an extra (`echo[langfuse]`). Guard imports with `try/except ImportError`.
- **Versions / labels.** Prefer `label` (e.g., `"production"`) over pinning to a version for deploys; pin versions for evals.

## Adding a new prompt provider

→ Use `[[echo-sdk-adding-a-provider]]`. Implement `BasePromptProvider`, register in `factory.py`, add optional-deps extra.

## Common mistakes

- **Re-fetching the same prompt every turn** → providers may rate-limit. Cache at the host layer if you fetch per request.
- **Mutating `FetchedPrompt`** → it's frozen-by-convention; treat as immutable.
- **Importing `langfuse` at module top** → optional dep, guard it.

## See also

- `[[echo-sdk-agents]]` for how `AgentConfig` is consumed.
- `[[python-optional-deps]]`
