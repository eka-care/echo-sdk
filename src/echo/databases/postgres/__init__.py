"""Postgres client for Echo SDK.

Requires the `postgres` extra: `pip install echo[postgres]`.
"""

from .binder import bind_named
from .client import PostgresClient
from .config import PostgresConfig

__all__ = ["PostgresClient", "PostgresConfig", "bind_named"]
