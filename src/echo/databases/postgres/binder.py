"""Convert psycopg-style named placeholders to asyncpg positional placeholders.

asyncpg only supports `$1, $2, ...` positional binding, but hand-written SQL
files often use psycopg's `%(name)s` for readability and re-use. This binder
rewrites placeholders and returns the values list in the order asyncpg expects.
A repeated named param maps to the same positional slot (deduped).
"""

import re
from typing import Any, Mapping

_NAMED_RE = re.compile(r"%\((\w+)\)s")


def bind_named(sql: str, params: Mapping[str, Any]) -> tuple[str, list[Any]]:
    """Rewrite `%(name)s` to `$1..$N` and return ordered values.

    Repeats are deduped — the same name binds to the same slot.
    Missing names resolve to `None` (asyncpg will send NULL).
    """
    order: list[str] = []

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in order:
            order.append(name)
        return f"${order.index(name) + 1}"

    rewritten = _NAMED_RE.sub(repl, sql)
    values = [params.get(name) for name in order]
    return rewritten, values
