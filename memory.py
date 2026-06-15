"""
hermes_lite/memory.py — Local Database Layer (SQLite Memory Plugin)

Provides async-safe SQLite storage for conversations, messages, and metadata
with schema versioning and migration support.

Tables:
  - schema_version: tracks applied migrations (singleton row)
  - sessions: long-lived conversation sessions (id, title, created_at, updated_at)
  - messages: individual messages within a session (id, session_id, role, content,
              metadata JSON, created_at, sequence)
  - metadata: key-value store keyed by (scope, key) for plugin data
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiosqlite

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

CREATE_SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  REAL    NOT NULL,
    PRIMARY KEY (version)
);
"""

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);
"""

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL DEFAULT '',
    metadata    TEXT    NOT NULL DEFAULT '{}',
    created_at  REAL    NOT NULL,
    sequence    INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
"""

CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS metadata (
    scope       TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL DEFAULT '',
    updated_at  REAL NOT NULL,
    PRIMARY KEY (scope, key)
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages(session_id, sequence);",
    "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);",
]

ALL_TABLE_DDL = [
    CREATE_SCHEMA_TABLES,
    CREATE_SESSIONS_TABLE,
    CREATE_MESSAGES_TABLE,
    CREATE_METADATA_TABLE,
    *CREATE_INDEXES,
]

# Migration functions: index = target version, function = migration logic
# Each migration receives (connection) and must apply its changes atomically.
MIGRATIONS: dict[int, str] = {
    # Version 1 is the base schema — no migrations to run from 0→1 is just DDL.
    # Future versions add entries like:
    #   2: "ALTER TABLE sessions ADD COLUMN tags TEXT NOT NULL DEFAULT '[]';",
}


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

@dataclass
class PooledConnection:
    """Wraps an aiosqlite connection with idle tracking."""
    conn: aiosqlite.Connection
    last_used: float = field(default_factory=time.time)
    in_use: bool = False


class AsyncSQLitePool:
    """Async-safe SQLite connection pool with configurable size and idle timeout.

    Designed for agent workloads: multiple coroutines share a pool of
    connections to the same database file. Connections are checked out
    on demand and returned after use.

    Thread-safe: all internal state mutations are protected by a lock since
    aiosqlite connections themselves are not thread-safe but the pool's
    bookkeeping must be.
    """

    def __init__(
        self,
        db_path: str | Path,
        min_size: int = 1,
        max_size: int = 4,
        idle_timeout: float = 60.0,
        journal_mode: str = "WAL",
    ) -> None:
        self.db_path = str(db_path)
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.journal_mode = journal_mode
        self._pool: list[PooledConnection] = []
        self._lock = threading.Lock()
        self._closed = False

    async def initialize(self) -> None:
        """Open the minimum number of connections and apply schema/migrations."""
        for _ in range(self.min_size):
            conn = await self._create_connection()
            self._pool.append(PooledConnection(conn=conn))

    async def _create_connection(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute(f"PRAGMA journal_mode={self.journal_mode};")
        await conn.execute("PRAGMA busy_timeout=5000;")
        await conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire a connection from the pool (context manager)."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        conn_wrapper = await self._checkout()
        try:
            yield conn_wrapper.conn
        finally:
            self._checkin(conn_wrapper)

    async def _checkout(self) -> PooledConnection:
        """Get the least-recently-used idle connection, or grow the pool."""
        now = time.time()
        with self._lock:
            # Prune stale idle connections
            self._pool = [
                pc
                for pc in self._pool
                if pc.in_use or (now - pc.last_used) < self.idle_timeout
            ]

            # Find an idle connection
            for pc in self._pool:
                if not pc.in_use:
                    pc.in_use = True
                    pc.last_used = now
                    return pc

        # No idle connection — grow if under max
        if len(self._pool) < self.max_size:
            conn = await self._create_connection()
            wrapper = PooledConnection(conn=conn, in_use=True)
            with self._lock:
                self._pool.append(wrapper)
            return wrapper

        # All connections busy — wait for one (spin with backoff)
        # In production you'd use an asyncio.Condition; for simplicity,
        # poll with exponential backoff capped at 100ms.
        delay = 0.01
        while True:
            await asyncio.sleep(delay)
            with self._lock:
                for pc in self._pool:
                    if not pc.in_use:
                        pc.in_use = True
                        pc.last_used = time.time()
                        return pc
            delay = min(delay * 2, 0.1)

    def _checkin(self, wrapper: PooledConnection) -> None:
        with self._lock:
            wrapper.in_use = False
            wrapper.last_used = time.time()

    async def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True
        with self._lock:
            conns = [pc.conn for pc in self._pool]
            self._pool.clear()
        for conn in conns:
            await conn.close()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._pool)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for pc in self._pool if pc.in_use)


# ---------------------------------------------------------------------------
# Schema management & migrations
# ---------------------------------------------------------------------------

async def ensure_schema(pool: AsyncSQLitePool) -> int:
    """Ensure the database schema exists and run pending migrations.

    Returns the current schema version after applying any migrations.
    """
    async with pool.acquire() as conn:
        # Create the schema_version table first (idempotent)
        for ddl in [CREATE_SCHEMA_TABLES, *[t for t in ALL_TABLE_DDL if t != CREATE_SCHEMA_TABLES]]:
            await conn.execute(ddl)

        # Read current version from the singleton row
        cursor = await conn.execute("SELECT MAX(version) AS v FROM schema_version;")
        row = await cursor.fetchone()
        current_version = row["v"] if row and row["v"] is not None else 0

        target = max(SCHEMA_VERSION, current_version)

        # Apply pending migrations
        for v in range(current_version + 1, target + 1):
            migration_sql = MIGRATIONS.get(v)
            if migration_sql:
                await conn.execute(migration_sql)
            await conn.execute(
                "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?);",
                (v, time.time()),
            )
            # Update the applied_at if the row already existed (re-migration)
            await conn.execute(
                "UPDATE schema_version SET applied_at = ? WHERE version = ?;",
                (time.time(), v),
            )

        await conn.commit()
        return current_version


# ---------------------------------------------------------------------------
# CRUD: Sessions
# ---------------------------------------------------------------------------

async def create_session(
    pool: AsyncSQLitePool,
    session_id: str,
    title: str = "",
) -> dict[str, Any]:
    """Create a new session. Returns the created session as a dict."""
    now = time.time()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
            (session_id, title, now, now),
        )
        await conn.commit()
    return {"id": session_id, "title": title, "created_at": now, "updated_at": now}


async def get_session(
    pool: AsyncSQLitePool,
    session_id: str,
) -> Optional[dict[str, Any]]:
    """Retrieve a session by id. Returns None if not found."""
    async with pool.acquire() as conn:
        cursor = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?;",
            (session_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return dict(row)


async def update_session(
    pool: AsyncSQLitePool,
    session_id: str,
    title: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Update a session's title and/or updated_at. Returns the updated session or None."""
    fields = []
    values: list[Any] = []
    now = time.time()

    if title is not None:
        fields.append("title = ?")
        values.append(title)
    fields.append("updated_at = ?")
    values.append(now)
    values.append(session_id)

    async with pool.acquire() as conn:
        cursor = await conn.execute(
            f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?;",
            values,
        )
        await conn.commit()
        if cursor.rowcount == 0:
            return None

    return await get_session(pool, session_id)


async def delete_session(
    pool: AsyncSQLitePool,
    session_id: str,
) -> bool:
    """Delete a session and all its messages (cascade). Returns True if deleted."""
    async with pool.acquire() as conn:
        cursor = await conn.execute(
            "DELETE FROM sessions WHERE id = ?;",
            (session_id,),
        )
        await conn.commit()
    return cursor.rowcount > 0


async def list_sessions(
    pool: AsyncSQLitePool,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List sessions ordered by most recently updated."""
    async with pool.acquire() as conn:
        cursor = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?;",
            (limit, offset),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# CRUD: Messages
# ---------------------------------------------------------------------------

async def insert_message(
    pool: AsyncSQLitePool,
    message_id: str,
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
    sequence: Optional[int] = None,
) -> dict[str, Any]:
    """Insert a message into a session. Returns the created message dict.

    If sequence is None, auto-increments from the last message's sequence + 1.
    """
    now = time.time()

    async with pool.acquire() as conn:
        # Auto-sequence if not specified
        if sequence is None:
            cursor = await conn.execute(
                "SELECT MAX(sequence) AS max_seq FROM messages WHERE session_id = ?;",
                (session_id,),
            )
            row = await cursor.fetchone()
            sequence = (row["max_seq"] + 1) if row["max_seq"] is not None else 0

        meta_json = json.dumps(metadata or {})

        await conn.execute(
            """INSERT INTO messages
               (id, session_id, role, content, metadata, created_at, sequence)
               VALUES (?, ?, ?, ?, ?, ?, ?);""",
            (message_id, session_id, role, content, meta_json, now, sequence),
        )

        # Touch the session's updated_at
        await conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?;",
            (now, session_id),
        )
        await conn.commit()

    return {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": now,
        "sequence": sequence,
    }


async def get_messages(
    pool: AsyncSQLitePool,
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    after_sequence: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Retrieve messages for a session, ordered by sequence.

    If after_sequence is provided, only messages with sequence > that value are returned.
    """
    where = "session_id = ?"
    values: list[Any] = [session_id]

    if after_sequence is not None:
        where += " AND sequence > ?"
        values.append(after_sequence)

    values.extend([limit, offset])

    async with pool.acquire() as conn:
        cursor = await conn.execute(
            f"SELECT id, session_id, role, content, metadata, created_at, sequence "
            f"FROM messages WHERE {where} ORDER BY sequence ASC LIMIT ? OFFSET ?;",
            values,
        )
        rows = await cursor.fetchall()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d["metadata"])
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
        result.append(d)
    return result


async def get_message_count(
    pool: AsyncSQLitePool,
    session_id: str,
) -> int:
    """Return the number of messages in a session."""
    async with pool.acquire() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM messages WHERE session_id = ?;",
            (session_id,),
        )
        row = await cursor.fetchone()
    return row["cnt"] if row else 0


async def delete_messages(
    pool: AsyncSQLitePool,
    session_id: str,
    before_sequence: Optional[int] = None,
) -> int:
    """Delete messages from a session. Returns the number deleted.

    If before_sequence is set, deletes all messages with sequence < that value.
    Otherwise deletes ALL messages in the session.
    """
    where = "session_id = ?"
    values: list[Any] = [session_id]

    if before_sequence is not None:
        where += " AND sequence < ?"
        values.append(before_sequence)

    async with pool.acquire() as conn:
        cursor = await conn.execute(
            f"DELETE FROM messages WHERE {where};",
            values,
        )
        await conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# CRUD: Metadata (key-value store)
# ---------------------------------------------------------------------------

async def set_metadata(
    pool: AsyncSQLitePool,
    scope: str,
    key: str,
    value: Any,
) -> None:
    """Store a metadata value (JSON-serialized)."""
    serialized = json.dumps(value, default=str)
    now = time.time()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO metadata (scope, key, value, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(scope, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at;""",
            (scope, key, serialized, now),
        )
        await conn.commit()


async def get_metadata(
    pool: AsyncSQLitePool,
    scope: str,
    key: str,
    default: Any = None,
) -> Any:
    """Retrieve a metadata value (deserialized from JSON)."""
    async with pool.acquire() as conn:
        cursor = await conn.execute(
            "SELECT value FROM metadata WHERE scope = ? AND key = ?;",
            (scope, key),
        )
        row = await cursor.fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return row["value"]


async def list_metadata(
    pool: AsyncSQLitePool,
    scope: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List metadata entries, optionally filtered by scope."""
    if scope:
        query = "SELECT scope, key, value, updated_at FROM metadata WHERE scope = ? ORDER BY scope, key;"
        params = (scope,)
    else:
        query = "SELECT scope, key, value, updated_at FROM metadata ORDER BY scope, key;"
        params = ()

    async with pool.acquire() as conn:
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d["value"] = json.loads(d["value"])
        except (json.JSONDecodeError, TypeError):
            pass
        result.append(d)
    return result


async def delete_metadata(
    pool: AsyncSQLitePool,
    scope: str,
    key: Optional[str] = None,
) -> int:
    """Delete metadata entries. If key is None, deletes all entries in scope."""
    if key:
        query = "DELETE FROM metadata WHERE scope = ? AND key = ?;"
        params = (scope, key)
    else:
        query = "DELETE FROM metadata WHERE scope = ?;"
        params = (scope,)

    async with pool.acquire() as conn:
        cursor = await conn.execute(query, params)
        await conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Convenience: session context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def session_context(
    pool: AsyncSQLitePool,
    session_id: str,
    title: str = "",
) -> AsyncIterator[dict[str, Any]]:
    """Ensure a session exists, yield it, and auto-update updated_at on exit.

    If the session already exists, it's returned as-is. If not, it's created
    with the given title.
    """
    existing = await get_session(pool, session_id)
    if existing is not None:
        yield existing
    else:
        created = await create_session(pool, session_id, title)
        yield created

    # Touch updated_at on exit so the session stays near the top of lists
    await update_session(pool, session_id)