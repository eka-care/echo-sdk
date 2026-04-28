"""Async Postgres client backed by an asyncpg connection pool."""

import logging
from typing import Any, Optional

from .binder import bind_named
from .config import PostgresConfig

logger = logging.getLogger(__name__)


class PostgresClient:
    """Async Postgres client with lazy pool initialization.

    Each query method accepts either positional `*args` (asyncpg native) or a
    `params: dict` kwarg. When `params` is given, `%(name)s` placeholders in
    the SQL are rewritten to `$N` via `bind_named` before execution.
    """

    def __init__(self, config: Optional[PostgresConfig] = None):
        self._config = config or PostgresConfig()
        self._pool: Any = None  # asyncpg.Pool, lazy

    async def _get_pool(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as e:
                raise ImportError(
                    "asyncpg is required for PostgresClient. "
                    "Install with: pip install echo[postgres]"
                ) from e

            cfg = self._config
            self._pool = await asyncpg.create_pool(
                host=cfg.host,
                port=cfg.port,
                database=cfg.database,
                user=cfg.user,
                password=cfg.password,
                min_size=cfg.min_size,
                max_size=cfg.max_size,
                command_timeout=cfg.command_timeout,
                ssl=cfg.ssl,
            )
            logger.info("Postgres pool established host=%s db=%s", cfg.host, cfg.database)
        return self._pool

    def _resolve(
        self, sql: str, args: tuple, params: Optional[dict]
    ) -> tuple[str, list[Any]]:
        if params is not None:
            if args:
                raise ValueError("Pass either positional *args OR params=dict, not both.")
            return bind_named(sql, params)
        return sql, list(args)

    async def fetch_one(
        self, sql: str, *args: Any, params: Optional[dict] = None
    ) -> Optional[dict]:
        pool = await self._get_pool()
        rewritten, values = self._resolve(sql, args, params)
        row = await pool.fetchrow(rewritten, *values)
        return dict(row) if row else None

    async def fetch_all(
        self, sql: str, *args: Any, params: Optional[dict] = None
    ) -> list[dict]:
        pool = await self._get_pool()
        rewritten, values = self._resolve(sql, args, params)
        rows = await pool.fetch(rewritten, *values)
        return [dict(r) for r in rows]

    async def execute(
        self, sql: str, *args: Any, params: Optional[dict] = None
    ) -> str:
        pool = await self._get_pool()
        rewritten, values = self._resolve(sql, args, params)
        return await pool.execute(rewritten, *values)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Postgres pool closed")
