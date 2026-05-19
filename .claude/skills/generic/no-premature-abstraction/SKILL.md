---
name: generic-no-premature-abstraction
description: Don't design for hypothetical future needs. Three similar lines beats a premature abstraction. Use whenever you're tempted to add a helper, base class, or config flag "in case".
---

# No Premature Abstraction

## Rules

- **Wait for the third (often fourth) occurrence** before extracting a helper. Two cases is duplication you can read; one helper is duplication hidden behind misnaming.
- **Don't add config flags for behavior nobody asked for.** Hardcode; flag when a second caller actually needs the other branch.
- **Don't add a base class for a single concrete subclass.** Inline. Refactor when the second subclass arrives.
- **Don't add `**kwargs` "for flexibility."** Name the params; add more when callers need them.
- **Don't add a registry / plugin system** unless there are already 3+ implementations or a real external extension point.
- **YAGNI on metadata fields.** "I might want this later" → leave it out; add it when you actually want it.

## Why

- Abstractions encode assumptions; the wrong abstraction is worse than copy-paste because it's harder to back out.
- Code with one caller and one implementation reads as concrete; the same code wrapped in `Strategy` + `Factory` + `Registry` becomes unreadable.
- Premature abstraction makes future, real abstractions harder — you have to undo the wrong one first.

## When abstraction IS justified

- Echo SDK's provider factories (`get_llm`, `get_prompt_provider`, etc.) — multiple implementations, real plugin surface, optional deps. **This is the bar.** Match that level of justification before adding similar patterns.

## Common mistakes

- Creating `BaseFoo` with one concrete `Foo` to "leave room" → delete the base; inline.
- Wrapping a one-line function in a class for "testability" → if the one-line function is pure, just import it in tests.
- Generics where a concrete type would do → `def foo(x: T) -> T` for a function only ever called with `str`.

## See also

- `[[generic-small-diffs]]`, `[[python-typing-discipline]]`
