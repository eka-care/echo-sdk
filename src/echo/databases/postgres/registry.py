"""Process-wide default `PostgresClient` registry.

Lives in its own module (not the package `__init__`) so that
`pg_query_tool` can import `get_default_client` without creating an
import cycle with the package that re-exports `PgQueryTool`.
"""

from typing import Optional

from .client import PostgresClient

_default_client: Optional[PostgresClient] = None


def set_default_client(client: PostgresClient) -> None:
    """Register a process-wide default `PostgresClient`.

    Tools constructed without an explicit client (e.g. via `tool_class()`
    at dynamic-loader time) resolve to this instance at runtime.
    """
    global _default_client
    _default_client = client


def get_default_client() -> PostgresClient:
    """Return the registered default client; raise if none was set."""
    if _default_client is None:
        raise RuntimeError(
            "No default PostgresClient registered. "
            "Call echo.databases.postgres.set_default_client(...) at startup."
        )
    return _default_client
