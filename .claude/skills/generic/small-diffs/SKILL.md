---
name: generic-small-diffs
description: Keep diffs surgical — no drive-by refactors, no opportunistic cleanups. Use whenever you're about to edit code.
---

# Small Diffs

The change should be exactly the change the task asks for. Nothing more.

## Rules

- **No drive-by refactors.** If you notice unrelated code that's ugly, leave it. Open a separate task if it matters.
- **No reformatting** of files you didn't otherwise need to touch. Don't fight the formatter; don't bring it.
- **No rename-everything passes.** If a single rename is requested, rename only what's needed for that rename to compile.
- **No opportunistic dependency bumps** while fixing an unrelated bug.
- **Three similar lines beats a premature abstraction.** Wait for the fourth occurrence before extracting.
- **Touch the fewest files possible** that still correctly solves the task. Fewer files = easier review = lower regression risk.

## Why

- Reviewers can actually review a small diff.
- Bisecting a regression to a 12-line diff is feasible; to a 1200-line "cleanup + fix" PR, it's not.
- Unrelated changes accumulate review fatigue and get rubber-stamped — that's where bugs hide.

## Common rationalizations to reject

- "It's right there, I'll just fix it" → no, separate task.
- "The test was wrong anyway" → fix the test in the test PR, not in the feature PR.
- "Let me bump the lockfile while I'm here" → no.

## See also

- `[[generic-no-premature-abstraction]]`, `[[generic-comments-only-when-why]]`
