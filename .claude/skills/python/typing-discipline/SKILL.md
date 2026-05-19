---
name: python-typing-discipline
description: Type-hint rules for echo-sdk — no Any unless boundary, Protocol for structural typing, narrow Unions, PEP 604 syntax. Use when writing public API signatures.
---

# Typing Discipline

Python 3.11+. Use modern syntax and narrow types.

## Rules

- **PEP 604 unions**: `str | None`, not `Optional[str]` (in new code). Existing `Optional` is fine — don't churn.
- **Built-in generics**: `list[str]`, `dict[str, int]` — not `List`/`Dict` from `typing` (in new code).
- **`Any` is a smell.** Use it only at true boundaries (provider SDKs, JSON blobs from external sources). Internally: name the type.
- **`Protocol` for structural typing** — when you want to accept "anything that has `.run()`" without inheritance, define a `Protocol`.
- **Narrow `Union`s**: `str | int` is fine; `str | int | bytes | list | dict | None` is a smell — usually means the function is doing too much.
- **`TYPE_CHECKING` for circular-import-only imports**:
  ```python
  if TYPE_CHECKING:
      from echo.models.user_conversation import ConversationContext
  ```
  Then use `"ConversationContext"` (string) in signatures.
- **`Literal` for finite string enums** at API surfaces (`SkillActivation = Literal["llm", "manual"]`) — Pydantic models with enums are preferred for configs.
- **`AsyncGenerator[T, None]`** for `async def` generators; `Iterator[T]` for sync.
- **`Final`** for module-level constants that must not be reassigned (`_RESERVED_META_TOOL_NAMES: Final = ("load_skill", ...)`)).

## Common mistakes

- `def foo(x) -> Any` → name the type; if you don't know, refactor.
- Using `typing.List` in new code → `list[T]`.
- Stringified types everywhere → only needed for forward refs / circular imports.
- `Union[X, None]` → `X | None`.

## See also

- `[[python-pydantic-v2]]`, `[[generic-no-premature-abstraction]]`
