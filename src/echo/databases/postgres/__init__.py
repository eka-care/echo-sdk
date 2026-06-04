"""Postgres client + query tool for Echo SDK.

Requires the `postgres` extra: `pip install echo[postgres]`.

Holds both the Postgres infrastructure (client/config/binder) and the
domain tool that exposes it to an LLM (`PgQueryTool`). The tool lives with
its resource rather than under `tools/`, and imports the tool framework
from `echo.tools` — keeping the dependency direction one-way
(databases → tools).
"""

from .binder import bind_named
from .client import PostgresClient
from .config import PostgresConfig
from .pg_query_tool import PgQueryTool
from .registry import get_default_client, set_default_client

__all__ = [
    "PostgresClient",
    "PostgresConfig",
    "PgQueryTool",
    "bind_named",
    "set_default_client",
    "get_default_client",
]
