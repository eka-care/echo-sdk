---
name: python-pydantic-v2
description: Pydantic v2 patterns used across echo-sdk configs and schemas — ConfigDict, model_validate, field_validator, no v1 patterns. Use when adding/editing config or schema models.
---

# Pydantic v2

The project is Pydantic ≥2.12. Don't write v1 code.

## Rules

- **`ConfigDict` not inner `class Config`.** `model_config = ConfigDict(frozen=True, ...)`.
- **`model_validate(data)` not `parse_obj(data)`.**
- **`model_dump()` not `dict()`.** For JSON: `orjson.dumps(model.model_dump())` — never `model.model_dump_json()` if you want orjson speed; use it if you want stdlib semantics.
- **`field_validator` not `validator`.** `@field_validator("name")` with `@classmethod`.
- **`model_validator(mode="after")` not `root_validator`.**
- **`Annotated[T, Field(...)]` is preferred** for field metadata over positional `Field(...)`.
- **Defaults**: `Optional[T] = None` is fine; `list[T] = Field(default_factory=list)` for mutable defaults (Pydantic 2 enforces this).
- **Frozen models for invariants.** `model_config = ConfigDict(frozen=True)` on configs that should not be mutated after construction.
- **Discriminated unions** for polymorphic content (`Message.content` variants): `Annotated[Union[...], Field(discriminator="type")]`.

## Patterns in this repo

- `LLMConfig`, `AgentConfig`, `PersonaConfig`, `TaskConfig`, `MCPServerConfig`, `PostgresConfig` — all Pydantic v2.
- `ConversationContext`, `Message`, `ToolCall`, `ToolResult` — Pydantic v2 with polymorphic content.

## Common mistakes

- `class Config: arbitrary_types_allowed = True` → `model_config = ConfigDict(arbitrary_types_allowed=True)`.
- `from pydantic import validator` → `field_validator`.
- `.dict()` / `.json()` → `.model_dump()` / `orjson.dumps(.model_dump())`.
- Mutating a frozen model → it'll raise; create a new instance with `model_copy(update={...})`.

## See also

- `[[python-orjson-only]]`, `[[python-typing-discipline]]`
