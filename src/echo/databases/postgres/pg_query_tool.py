"""BaseTool subclass that runs a parametric SQL query against Postgres."""

import logging

from typing import Any, Literal, Optional

from echo.tools.core import BaseTool

from .client import PostgresClient
from .registry import get_default_client

log = logging.getLogger(__name__)


class PgQueryTool(BaseTool):
    """Run a parametric SQL query and return rows.

    Subclass and set `name`, `description`, `sql`, `input_schema`, and
    `fetch_mode`, or pass them at construction. Override `transform_params`
    to derive SQL bind values from LLM input + tool_context (e.g. inject
    workspace_id from session, coalesce defaults).

    The SQL uses psycopg-style `%(name)s` placeholders; the underlying
    `PostgresClient` rewrites them to asyncpg `$N` via `bind_named`.

    Constructing with `client=None` is supported so dynamic loaders that call
    `tool_class()` work; in that case the client is resolved from
    `echo.databases.postgres.get_default_client()` at first run.
    """

    name: str = ""
    description: str = ""
    sql: str = ""
    fetch_mode: Literal["one", "all"] = "all"

    def __init__(
        self,
        client: Optional[PostgresClient] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        sql: Optional[str] = None,
        input_schema: Optional[dict] = None,
        fetch_mode: Optional[Literal["one", "all"]] = None,
    ):
        self._client = client
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if sql is not None:
            self.sql = sql
        if fetch_mode is not None:
            self.fetch_mode = fetch_mode
        self._input_schema_override = input_schema

    @property
    def client(self) -> PostgresClient:
        """Resolve to the explicitly-injected client or the registered default."""
        return self._client if self._client is not None else get_default_client()

    @property
    def input_schema(self) -> dict:
        if self._input_schema_override is not None:
            return self._input_schema_override
        return {"type": "object", "properties": {}, "required": []}

    def transform_params(self, tool_context: Optional[dict], **kwargs: Any) -> dict:
        """Map LLM args + tool_context to the dict bound into the SQL.

        Default implementation passes kwargs through. Subclasses typically
        inject session-scoped values (workspace_id, today) and coalesce defaults.
        """
        return dict(kwargs)

    async def run(self, *, tool_context: Optional[dict] = None, **kwargs: Any) -> Any:
        params = self.transform_params(tool_context, **kwargs)
        if self.fetch_mode == "one":
            return await self.client.fetch_one(self.sql, params=params)
        data = await self.client.fetch_all(self.sql, params=params)
        log.info("PGSQL params: %s" % params)
        # log.info("PGSQL kwargs: %s :: fetched %s rows", kwargs, data)
        return data
