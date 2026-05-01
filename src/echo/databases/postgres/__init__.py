"""Postgres client for Echo SDK.

Requires the `postgres` extra: `pip install echo[postgres]`.
"""

from typing import Optional

from .binder import bind_named
from .client import PostgresClient
from .config import PostgresConfig

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


__all__ = [
    "PostgresClient",
    "PostgresConfig",
    "bind_named",
    "set_default_client",
    "get_default_client",
]
