"""Postgres connection configuration."""

import os
from typing import Optional

from pydantic import BaseModel


class PostgresConfig(BaseModel):
    """Postgres connection settings.

    Defaults are read from `ECHO_PG_*` environment variables at class-definition
    time, matching the `LLMConfig` pattern. Override via constructor kwargs.
    """

    host: str = os.getenv("ECHO_PG_HOST", "localhost")
    database: str = os.getenv("ECHO_PG_DATABASE", "")
    user: str = os.getenv("ECHO_PG_USER", "")
    password: str = os.getenv("ECHO_PG_PASSWORD", "")
    port: int = int(os.getenv("ECHO_PG_PORT", "5432"))
    min_size: int = int(os.getenv("ECHO_PG_POOL_MIN", "2"))
    max_size: int = int(os.getenv("ECHO_PG_POOL_MAX", "10"))
    command_timeout: int = int(os.getenv("ECHO_PG_COMMAND_TIMEOUT", "10"))
    ssl: Optional[str] = os.getenv("ECHO_PG_SSL") or None
