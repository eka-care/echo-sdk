---
name: python-orjson-only
description: Use orjson, never stdlib json. orjson is a hard dep of echo-sdk and used throughout. Fires any time you serialize/deserialize JSON.
---

# orjson Only

`orjson` is a project dependency and the standard across echo-sdk. **Never** use stdlib `json`.

## Rules

- **`orjson.dumps(obj) -> bytes`** — returns bytes, not str. If you need str, `.decode()`.
- **`orjson.loads(data)`** — accepts bytes or str.
- **Pydantic models**: `orjson.dumps(model.model_dump())`. Pydantic's own `model_dump_json()` uses stdlib semantics and is slower — only use it if you specifically need stdlib JSON semantics (rare).
- **Datetime / UUID / numpy** — orjson handles these natively with the right options: `orjson.dumps(obj, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)`.
- **Sorted keys** for deterministic output (e.g., cache keys): `option=orjson.OPT_SORT_KEYS`.

## Why

- 5–10× faster than stdlib `json` on real workloads.
- Handles `datetime`, `UUID`, `numpy` without custom encoders.
- Already a hard dep — using stdlib `json` is just inconsistency.

## Common mistakes

- `import json` anywhere in `src/echo/` → use `import orjson`.
- `json.dumps(x).encode()` → `orjson.dumps(x)` (already bytes).
- `json.loads(x.decode())` → `orjson.loads(x)` (accepts bytes).

## Exception

- Reading `pyproject.toml` / config files where a library expects stdlib `json` — that's fine, but in our code, `orjson`.

## See also

- `[[python-pydantic-v2]]`
