---
name: generic-comments-only-when-why
description: Default to no comments. Only write one when the WHY is non-obvious. Never comments that restate the code. Use any time you're about to write a comment.
---

# Comments Only When WHY

## Rules

- **Default: no comment.** Well-named identifiers explain WHAT. Code structure explains HOW. Comments are reserved for WHY.
- **Write a comment when**:
  - There's a non-obvious constraint (e.g., "API requires sorted keys; do not change order").
  - There's a subtle invariant (e.g., "dict-as-ordered-set; preserves insertion order").
  - There's a workaround for a specific bug or quirk (link to the issue/provider docs).
  - Behavior would surprise a reader (e.g., "silently dropped; see _validate_skill_registry").
- **Never write a comment that**:
  - Restates the next line (`# increment counter` above `counter += 1`).
  - References the current task or PR (`# fix for ticket ABC-123`) — that's PR metadata, not code.
  - Names the caller ("used by GenericAgent") — caller relationships change; the comment rots.
  - Describes WHAT the code does in prose ("This function takes a list and returns the sum").
- **Docstrings**: short. One-line summary; a few lines of args/returns only if non-obvious. No multi-paragraph essays.
- **Module-level docstring**: state the purpose in 2–3 sentences. That's it.

## Why

- Comments rot. Code is the source of truth.
- Restate-the-code comments train readers to skim past comments, missing the real ones.
- Caller-referencing comments are wrong the moment refactoring happens.

## Examples (good)

```python
# Gemini rejects $ref/$defs — flatten before sending.
flattened = self._flatten_schema(schema, defs)
```

```python
# Idempotent: re-activating a live skill is a no-op (hooks do not re-fire).
if name in self._active_skill_names:
    return
```

## Examples (bad)

```python
# Loop through tools
for tool in tools:   # ← restates the code
```

```python
# Added for the new skill activation PR
self._meta_tool_names = []   # ← rots the moment the PR merges
```

## See also

- `[[generic-small-diffs]]`, `[[generic-no-premature-abstraction]]`
