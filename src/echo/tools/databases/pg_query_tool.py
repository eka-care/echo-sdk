"""BaseTool subclass that runs a parametric SQL query against Postgres."""

from typing import Any, Literal, Optional

from echo.databases.postgres import PostgresClient
from echo.tools.base_tool import BaseTool


class PgQueryTool(BaseTool):
    """Run a parametric SQL query and return rows.

    Subclass and set `name`, `description`, `sql`, `input_schema`, and
    `fetch_mode`, or pass them at construction. Override `transform_params`
    to derive SQL bind values from LLM input + tool_context (e.g. inject
    workspace_id from session, coalesce defaults).

    The SQL uses psycopg-style `%(name)s` placeholders; the underlying
    `PostgresClient` rewrites them to asyncpg `$N` via `bind_named`.
    """

    name: str = ""
    description: str = ""
    sql: str = ""
    fetch_mode: Literal["one", "all"] = "all"

    def __init__(
        self,
        client: PostgresClient,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        sql: Optional[str] = None,
        input_schema: Optional[dict] = None,
        fetch_mode: Optional[Literal["one", "all"]] = None,
    ):
        self.client = client
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
        breakpoint()
        params = self.transform_params(tool_context, **kwargs)
        breakpoint()
        if self.fetch_mode == "one":
            return await self.client.fetch_one(self.sql, params=params)
        return await self.client.fetch_all(self.sql, params=params)
