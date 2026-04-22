"""
MCP Connection Manager for Echo SDK.

Two independent caches:

1. Tool-schema cache: always on, keyed by (transport, url, hash of the
   subset of headers listed in config.tool_cache_key_headers). Populated
   by a throwaway session on miss. No eviction.

2. Session cache: off by default. Turned on per-call when the caller
   passes user_session_id to execute_tool. Keyed by user_session_id.
   Idle-TTL'd and LRU-bounded.

When user_session_id is None (the default), execute_tool opens a fresh
ClientSession per call and closes it — this is the parallel-friendly
path, recommended for telephony / agent fan-out workloads.
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import orjson
from mcp import ClientSession

from .mcp_tool import MCPTool
from .schemas import (
    MCPConfigError,
    MCPConnectionError,
    MCPExecutionError,
    MCPServerConfig,
    MCPTransport,
)

logger = logging.getLogger(__name__)


@dataclass
class MCPConnection:
    """
    A cached session entry.

    Held in MCPConnectionManager._sessions only when the caller opted in
    via user_session_id. Fresh-session calls never create one of these.
    """

    session: ClientSession
    transport_context: Any
    http_client: Any  # httpx.AsyncClient for HTTP transports; None for stdio
    connected_at: float
    last_used: float
    idle_ttl: int
    absolute_ttl: int
    active_count: int = 0


class MCPConnectionManager:
    """
    Manages MCP tool discovery and tool execution.

    - get_tools(): uses the tool-schema cache; opens a throwaway session
      on cache miss.
    - execute_tool(..., user_session_id=None): fresh session per call.
    - execute_tool(..., user_session_id="..."): cached session reused
      across calls sharing that id.
    - forget_session(id): evict and close a cached session.
    """

    MAX_CACHED_SESSIONS: ClassVar[int] = int(
        os.environ.get("ECHO_MCP_MAX_CONNECTIONS", "1000")
    )

    _sessions: ClassVar[Dict[str, MCPConnection]] = {}
    _session_locks: ClassVar[Dict[str, asyncio.Lock]] = {}
    _tools_cache: ClassVar[Dict[str, List[MCPTool]]] = {}
    _tool_discovery_locks: ClassVar[Dict[str, asyncio.Lock]] = {}
    _cleanup_task: ClassVar[Optional[asyncio.Task]] = None

    def __init__(self, config: MCPServerConfig):
        config.validate()
        self._config = config
        self._tool_cache_key = self._make_tool_cache_key(config)

    # ---- Public API ----

    async def get_tools(
        self,
        filter_fn: Optional[Callable] = None,
        tool_names: Optional[List[str]] = None,
    ) -> List[MCPTool]:
        """
        Discover tools. Uses the tool-schema cache keyed by
        (transport, url, tool_cache_key_headers). On miss, opens a
        throwaway session to list tools, closes it, caches the result.

        Concurrent get_tools calls for the same cache key will serialize
        through a per-key lock so only one throwaway session is opened.
        """
        if self._tool_cache_key not in self._tools_cache:
            lock = self._tool_discovery_locks.get(self._tool_cache_key)
            if lock is None:
                lock = asyncio.Lock()
                self._tool_discovery_locks[self._tool_cache_key] = lock
            async with lock:
                # Re-check under the lock — another caller may have just filled it
                if self._tool_cache_key not in self._tools_cache:
                    session, transport_ctx, http_client = await self._open_session()
                    try:
                        tools_response = await session.list_tools()
                    finally:
                        await self._close_parts(session, transport_ctx, http_client)

                    all_tools = [
                        MCPTool(
                            manager=self,
                            server_id=self._tool_cache_key,
                            tool_name=t.name,
                            tool_description=t.description or "",
                            input_schema=getattr(t, "inputSchema", None),
                        )
                        for t in tools_response.tools
                    ]
                    self._tools_cache[self._tool_cache_key] = all_tools
                    logger.info(
                        "Discovered and cached %d tools for %s",
                        len(all_tools), self._tool_cache_key,
                    )

        tools = self._tools_cache[self._tool_cache_key]
        return self._apply_filters(tools, filter_fn, tool_names)

    async def refresh_tools_cache(self) -> None:
        """Force refresh of the tool cache entry for this config."""
        self._tools_cache.pop(self._tool_cache_key, None)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        user_session_id: Optional[str] = None,
    ) -> Any:
        """
        Execute a tool.

        If user_session_id is None (default): fresh session per call.
          Safe for parallel fan-out. No caching.
        If user_session_id is not None: reuse (or create) a cached session
          for that id. Saves one initialize RTT per call after the first.
          Serializes when the server does; use None for parallel workloads.

        STDIO transport: user_session_id is accepted for API uniformity but
        ignored — stdio always uses the fresh-session path for now.
        """
        use_cache = (
            user_session_id is not None
            and self._config.transport != MCPTransport.STDIO
        )
        if not use_cache:
            return await self._execute_fresh(tool_name, arguments, meta)
        return await self._execute_cached(tool_name, arguments, meta, user_session_id)

    async def forget_session(self, user_session_id: str) -> None:
        """
        Evict and close the cached session for this id. No-op if not cached.

        Contract: caller should invoke this after their last execute_tool
        for this id has returned. If calls are in flight when forget_session
        runs, those calls will raise as the underlying session is torn down.
        """
        lock = self._get_session_lock(user_session_id)
        async with lock:
            conn = self._sessions.pop(user_session_id, None)
            self._session_locks.pop(user_session_id, None)
            if conn is None:
                return
            await self._close_connection_static(conn)
        logger.info("Forgot session: %s", user_session_id)

    @classmethod
    async def cleanup_all(cls):
        """Cleanup all sessions and the cleanup task. Call at shutdown."""
        if cls._cleanup_task and not cls._cleanup_task.done():
            cls._cleanup_task.cancel()
            try:
                await cls._cleanup_task
            except asyncio.CancelledError:
                pass
        cls._cleanup_task = None

        for user_session_id, conn in list(cls._sessions.items()):
            try:
                await cls._close_connection_static(conn)
            except Exception as e:
                logger.debug("Error closing session %s: %s", user_session_id, e)

        cls._sessions.clear()
        cls._session_locks.clear()
        cls._tools_cache.clear()
        cls._tool_discovery_locks.clear()
        logger.info("Cleaned up all MCP manager state")

    # ---- Internal: fresh-session path ----

    async def _execute_fresh(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        meta: Optional[Dict[str, Any]],
    ) -> Any:
        session, transport_ctx, http_client = await self._open_session()
        try:
            result = await session.call_tool(tool_name, arguments=arguments, meta=meta)
            logger.debug("Executed %s (fresh session)", tool_name)
            return result
        except Exception as e:
            raise MCPExecutionError(
                f"Failed to execute '{tool_name}' (fresh session): {e}"
            ) from e
        finally:
            await self._close_parts(session, transport_ctx, http_client)

    # ---- Internal: cached-session path ----

    async def _execute_cached(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        meta: Optional[Dict[str, Any]],
        user_session_id: str,
    ) -> Any:
        conn = await self._get_or_create_cached_session(user_session_id)
        conn.active_count += 1
        try:
            result = await conn.session.call_tool(
                tool_name, arguments=arguments, meta=meta
            )
            conn.last_used = time.time()
            logger.debug("Executed %s on cached session %s", tool_name, user_session_id)
            return result
        except Exception as e:
            logger.warning(
                "Cached session %s failed on %s: %s. Evicting.",
                user_session_id, tool_name, e,
            )
            await self.forget_session(user_session_id)
            raise MCPExecutionError(
                f"Failed to execute '{tool_name}' on cached session: {e}"
            ) from e
        finally:
            conn.active_count = max(0, conn.active_count - 1)

    async def _get_or_create_cached_session(
        self, user_session_id: str
    ) -> MCPConnection:
        lock = self._get_session_lock(user_session_id)
        async with lock:
            conn = self._sessions.get(user_session_id)
            if conn is not None:
                return conn

            if len(self._sessions) >= self.MAX_CACHED_SESSIONS:
                self._evict_lru_unlocked()

            session, transport_ctx, http_client = await self._open_session()
            now = time.time()
            conn = MCPConnection(
                session=session,
                transport_context=transport_ctx,
                http_client=http_client,
                connected_at=now,
                last_used=now,
                idle_ttl=self._config.session_idle_ttl,
                absolute_ttl=self._config.session_absolute_ttl,
            )
            self._sessions[user_session_id] = conn
            logger.info("Created cached session: %s", user_session_id)

            self._ensure_cleanup_task()
            return conn

    def _evict_lru_unlocked(self) -> None:
        """
        Evict least-recently-used idle (active_count == 0) cached session.
        Called while holding a per-id lock, so safe to mutate _sessions.
        Does NOT close the session synchronously — schedules close.
        Raises MCPConnectionError if no evictable candidate exists.
        """
        lru_id = None
        lru_time = float("inf")
        for sid, c in self._sessions.items():
            if c.active_count == 0 and c.last_used < lru_time:
                lru_time = c.last_used
                lru_id = sid
        if lru_id is None:
            raise MCPConnectionError(
                "Session cache full and all entries have in-flight calls"
            )
        conn = self._sessions.pop(lru_id)
        self._session_locks.pop(lru_id, None)
        asyncio.create_task(self._close_connection_static(conn))
        logger.info("Evicted LRU cached session: %s", lru_id)

    def _get_session_lock(self, user_session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(user_session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[user_session_id] = lock
        return lock

    # ---- Internal: session primitives ----

    async def _open_session(self) -> Tuple[ClientSession, Any, Any]:
        """
        Open a fresh MCP ClientSession. Returns (session, transport_context, http_client).
        Caller is responsible for closing via _close_parts.

        On partial failure (transport opened but session init failed), this
        method cleans up the transport and http_client itself so no resource
        leaks escape.
        """
        transport_ctx: Any = None
        http_client: Any = None
        try:
            if self._config.transport == MCPTransport.SSE:
                transport_ctx = self._connect_sse()
            elif self._config.transport == MCPTransport.STDIO:
                transport_ctx = self._connect_stdio()
            elif self._config.transport == MCPTransport.STREAMABLE_HTTP:
                transport_ctx, http_client = self._connect_streamable_http()
            else:
                raise MCPConfigError(
                    f"Unknown transport type: {self._config.transport}"
                )

            transport_result = await transport_ctx.__aenter__()
            if self._config.transport == MCPTransport.STREAMABLE_HTTP:
                read_stream, write_stream, _ = transport_result
            else:
                read_stream, write_stream = transport_result

            session = ClientSession(read_stream, write_stream)
            try:
                await session.__aenter__()
                await session.initialize()
            except Exception:
                # Session failed to initialize — tear down session, transport, http_client
                await self._close_parts(session, transport_ctx, http_client)
                raise
            return session, transport_ctx, http_client
        except MCPConfigError:
            raise
        except Exception as e:
            # Transport opened (or partially opened) and then failed before session was built.
            # Close the transport and http_client if we got them.
            if transport_ctx is not None:
                try:
                    await transport_ctx.__aexit__(None, None, None)
                except Exception as close_err:
                    logger.debug("Transport close after failed open: %s", close_err)
            if http_client is not None:
                try:
                    await http_client.aclose()
                except Exception as close_err:
                    logger.debug("httpx close after failed open: %s", close_err)
            raise MCPConnectionError(f"Failed to open MCP session: {e}") from e

    @staticmethod
    async def _close_parts(
        session: Optional[ClientSession], transport_ctx: Any, http_client: Any
    ) -> None:
        """Close session, transport, and http_client in order, ignoring benign errors."""
        if session is not None:
            try:
                await session.__aexit__(None, None, None)
            except RuntimeError as e:
                if "cancel scope" not in str(e):
                    logger.debug("Session close error: %s", e)
            except Exception as e:
                logger.debug("Session close error: %s", e)

        if transport_ctx is not None:
            try:
                await transport_ctx.__aexit__(None, None, None)
            except RuntimeError as e:
                if "cancel scope" not in str(e):
                    logger.debug("Transport close error: %s", e)
            except Exception as e:
                logger.debug("Transport close error: %s", e)

        if http_client is not None:
            try:
                await http_client.aclose()
            except Exception as e:
                logger.debug("httpx close error: %s", e)

    @classmethod
    async def _close_connection_static(cls, conn: MCPConnection) -> None:
        """Close a cached MCPConnection without needing a manager instance."""
        await cls._close_parts(conn.session, conn.transport_context, conn.http_client)

    # ---- Internal: transport construction ----

    def _connect_sse(self):
        from mcp.client.sse import sse_client
        return sse_client(
            url=str(self._config.url),
            headers=self._config.headers,
            timeout=self._config.timeout,
            sse_read_timeout=self._config.sse_read_timeout,
        )

    def _connect_stdio(self):
        from mcp.client.stdio import StdioServerParameters, stdio_client
        server_params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args or [],
            env=self._config.env,
        )
        return stdio_client(server_params)

    def _connect_streamable_http(self):
        import httpx
        from mcp.client.streamable_http import streamable_http_client
        http_client = httpx.AsyncClient(
            headers=self._config.headers or {},
            timeout=httpx.Timeout(
                self._config.timeout, read=self._config.sse_read_timeout
            ),
        )
        ctx = streamable_http_client(url=str(self._config.url), http_client=http_client)
        return ctx, http_client

    # ---- Internal: keying and filtering ----

    @staticmethod
    def _header_subset(
        headers: Optional[Dict[str, str]], key_headers: Optional[List[str]]
    ) -> Dict[str, str]:
        if not key_headers or not headers:
            return {}
        wanted = {h.lower() for h in key_headers}
        return {k.lower(): v for k, v in headers.items() if k.lower() in wanted}

    def _make_tool_cache_key(self, config: MCPServerConfig) -> str:
        if config.transport in (MCPTransport.SSE, MCPTransport.STREAMABLE_HTTP):
            subset = self._header_subset(config.headers, config.tool_cache_key_headers)
            headers_hash = hashlib.md5(
                orjson.dumps(subset, option=orjson.OPT_SORT_KEYS)
            ).hexdigest()[:8]
            return f"{config.transport.value}:{config.url}:{headers_hash}"
        env_hash = hashlib.md5(
            orjson.dumps(config.env or {}, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()[:8]
        return (
            f"{config.transport.value}:{config.command}:"
            f"{':'.join(config.args or [])}:{env_hash}"
        )

    def _apply_filters(
        self,
        tools: List[MCPTool],
        filter_fn: Optional[Callable],
        tool_names: Optional[List[str]],
    ) -> List[MCPTool]:
        result = tools
        if self._config.tool_include:
            inc = set(self._config.tool_include)
            result = [t for t in result if t.name in inc]
        if self._config.tool_exclude:
            exc = set(self._config.tool_exclude)
            result = [t for t in result if t.name not in exc]
        if tool_names:
            wanted = set(tool_names)
            result = [t for t in result if t.name in wanted]
        if filter_fn:
            result = [t for t in result if filter_fn(t)]
        return result

    # ---- Internal: background cleanup ----

    def _ensure_cleanup_task(self) -> None:
        cls = type(self)
        if cls._cleanup_task is not None and not cls._cleanup_task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No event loop available for cleanup task")
            return
        cls._cleanup_task = asyncio.create_task(cls._cleanup_loop())

    @classmethod
    async def _cleanup_loop(cls) -> None:
        """Periodically evict sessions past idle_ttl or absolute_ttl."""
        while True:
            try:
                await asyncio.sleep(60)
                now = time.time()
                expired: List[str] = []
                for sid, conn in list(cls._sessions.items()):
                    if conn.active_count > 0:
                        continue
                    idle = now - conn.last_used
                    absolute = now - conn.connected_at
                    if idle > conn.idle_ttl or absolute > conn.absolute_ttl:
                        expired.append(sid)
                for sid in expired:
                    conn = cls._sessions.pop(sid, None)
                    cls._session_locks.pop(sid, None)
                    if conn is not None:
                        try:
                            await cls._close_connection_static(conn)
                        except Exception as e:
                            logger.debug("Cleanup close error for %s: %s", sid, e)
                        logger.info("TTL-evicted session: %s", sid)
            except asyncio.CancelledError:
                logger.debug("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error("Cleanup loop error: %s", e)
