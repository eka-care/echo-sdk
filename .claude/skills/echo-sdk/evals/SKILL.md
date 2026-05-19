---
name: echo-sdk-evals
description: Evaluation provider abstraction for dataset experiments (Langfuse today). Use when running an experiment, adding a new eval backend, or wiring evals into CI.
---

# Evals

## What you're working with

- `BaseEvalProvider` in `evals/base.py` — abstract; declares `run_experiment(name, dataset_name, run_func)`.
- `LangfuseProvider` in `evals/langfuse_provider.py` — concrete implementation.
- `get_eval_provider()` in `evals/factory.py` — singleton factory.

## Rules

- **`run_func` signature**: `async def run_func(item) -> result` — receives a dataset item, returns the model output (and optionally metadata). Provider records the result against the experiment.
- **Provider is optional.** `langfuse` is an extra; guard imports.
- **One experiment per call** — don't multiplex multiple experiments through one `run_experiment`.
- **Pin prompt versions** when evaluating (see `[[echo-sdk-prompts]]`) — `label="production"` drifts under your feet.

## Adding a new eval provider

→ `[[echo-sdk-adding-a-provider]]`. Implement `BaseEvalProvider`, register in `factory.py`.

## Common mistakes

- **Mutating shared state across `run_func` invocations** — items may run concurrently; keep `run_func` pure.
- **Logging large blobs** synchronously inside `run_func` — wrap in `asyncio.to_thread` or skip.

## See also

- `examples/prompt_eval_usage.py`
- `[[echo-sdk-prompts]]`, `[[python-optional-deps]]`
