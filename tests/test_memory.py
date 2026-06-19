"""
tests/test_memory.py — Unit tests for hermes_lite.memory

Tests cover:
- AsyncSQLitePool lifecycle (init, acquire, close, concurrent access)
- Schema migration (ensure_schema, version tracking)
- Session CRUD (create, read, update, delete, list)
- Message CRUD (insert, read, count, delete)
- Metadata CRUD (set, get, list, delete)
- Edge cases (missing sessions, large content, concurrent writes)
- session_context convenience manager
"""

import asyncio
import json
import os
import time
import tempfile
import uuid

import pytest

from hermes_lite.memory import (
    AsyncSQLitePool,
    SCHEMA_VERSION,
    ensure_schema,
    create_session,
    get_session,
    update_session,
    delete_session,
    list_sessions,
    insert_message,
    get_messages,
    get_message_count,
    delete_messages,
    set_metadata,
    get_metadata,
    list_metadata,
    delete_metadata,
    session_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def pool():
    """Create a fresh in-memory SQLite pool for each test."""
    p = AsyncSQLitePool(":memory:", min_size=1, max_size=2)
    await p.initialize()
    await ensure_schema(p)
    yield p
    await p.close()


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a file path for a temporary SQLite database (file-based, shared)."""
    return str(tmp_path / "test.db")


def new_id() -> str:
    """Generate a unique ID for a session or message."""
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Pool tests
# ---------------------------------------------------------------------------

class TestPool:
    async def test_initialize_creates_min_connections(self):
        p = AsyncSQLitePool(":memory:", min_size=2, max_size=4)
        assert p.size == 0
        await p.initialize()
        assert p.size == 2
        await p.close()

    async def test_acquire_returns_connection(self):
        p = AsyncSQLitePool(":memory:", min_size=1, max_size=1)
        await p.initialize()
        async with p.acquire() as conn:
            cursor = await conn.execute("SELECT 1 AS val;")
            row = await cursor.fetchone()
            assert row["val"] == 1
        assert p.active_count == 0
        await p.close()

    async def test_pool_grows_under_load(self):
        p = AsyncSQLitePool(":memory:", min_size=1, max_size=3, idle_timeout=999)
        await p.initialize()
        assert p.size == 1

        async def use_conn():
            async with p.acquire() as conn:
                cursor = await conn.execute("SELECT 1;")
                await cursor.fetchone()
                await asyncio.sleep(0.05)

        # Use connections concurrently — pool should grow
        await asyncio.gather(use_conn(), use_conn(), use_conn())
        # Pool should have grown (maybe not to max, but > 1)
        assert p.size >= 2
        await p.close()

    async def test_close_idempotent(self):
        p = AsyncSQLitePool(":memory:")
        await p.initialize()
        await p.close()
        await p.close()  # should not raise

    async def test_acquire_after_close_raises(self):
        p = AsyncSQLitePool(":memory:")
        await p.initialize()
        await p.close()
        with pytest.raises(RuntimeError, match="closed"):
            async with p.acquire():
                pass

    async def test_pool_prunes_stale_idle(self):
        p = AsyncSQLitePool(":memory:", min_size=2, max_size=4, idle_timeout=0.01)
        await p.initialize()
        await asyncio.sleep(0.05)  # let idle timeout pass
        # Acquire and release — this triggers prune
        async with p.acquire():
            pass
        await p.close()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    async def test_ensure_schema_creates_tables(self, pool):
        """Tables exist and have expected columns."""
        async with pool.acquire() as conn:
            # Check tables
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            tables = {row["name"] for row in await cursor.fetchall()}
            assert "sessions" in tables
            assert "messages" in tables
            assert "metadata" in tables
            assert "schema_version" in tables

    async def test_ensure_schema_tracks_version(self, pool):
        """Schema version is recorded correctly."""
        async with pool.acquire() as conn:
            cursor = await conn.execute("SELECT MAX(version) AS v FROM schema_version;")
            row = await cursor.fetchone()
            assert row["v"] == SCHEMA_VERSION

    async def test_ensure_schema_idempotent(self, pool):
        """Running ensure_schema twice doesn't fail."""
        await ensure_schema(pool)
        async with pool.acquire() as conn:
            cursor = await conn.execute("SELECT COUNT(*) AS cnt FROM schema_version;")
            row = await cursor.fetchone()
            assert row["cnt"] == SCHEMA_VERSION

    async def test_indexes_created(self, pool):
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';"
            )
            indexes = {row["name"] for row in await cursor.fetchall()}
            assert "idx_messages_session" in indexes
            assert "idx_messages_session_seq" in indexes
            assert "idx_messages_created" in indexes
            assert "idx_sessions_updated" in indexes

    async def test_foreign_key_enforced(self, pool):
        """Inserting a message with a non-existent session_id should fail."""
        async with pool.acquire() as conn:
            with pytest.raises(Exception):
                await conn.execute(
                    "INSERT INTO messages (id, session_id, role, content, metadata, created_at, sequence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?);",
                    (new_id(), "nonexistent", "user", "hello", "{}", time.time(), 0),
                )


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

class TestSessions:
    async def test_create_and_get(self, pool):
        sid = new_id()
        created = await create_session(pool, sid, title="test session")
        assert created["id"] == sid
        assert created["title"] == "test session"

        fetched = await get_session(pool, sid)
        assert fetched is not None
        assert fetched["id"] == sid
        assert fetched["title"] == "test session"

    async def test_get_nonexistent(self, pool):
        fetched = await get_session(pool, "nope")
        assert fetched is None

    async def test_update_title(self, pool):
        sid = new_id()
        await create_session(pool, sid, title="old")
        updated = await update_session(pool, sid, title="new")
        assert updated["title"] == "new"
        assert updated["updated_at"] >= updated["created_at"]

    async def test_update_nonexistent(self, pool):
        result = await update_session(pool, "nope", title="ignored")
        assert result is None

    async def test_delete(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        deleted = await delete_session(pool, sid)
        assert deleted is True
        assert await get_session(pool, sid) is None

    async def test_delete_nonexistent(self, pool):
        assert await delete_session(pool, "nope") is False

    async def test_delete_cascades_to_messages(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        await insert_message(pool, new_id(), sid, "user", "hi")
        await delete_session(pool, sid)
        assert await get_message_count(pool, sid) == 0

    async def test_list_sessions_ordered(self, pool):
        sids = [new_id() for _ in range(3)]
        for i, sid in enumerate(sids):
            await create_session(pool, sid, title=f"session-{i}")
            await asyncio.sleep(0.01)  # ensure distinct timestamps

        all_sessions = await list_sessions(pool, limit=10)
        assert len(all_sessions) >= 3
        # Most recently created should be last (since created time determines order)
        # Actually: list is ordered by updated_at DESC, and we didn't touch updated_at
        # after creation, so the last created has the latest updated_at
        assert all_sessions[0]["id"] == sids[-1]  # newest first

    async def test_list_pagination(self, pool):
        sids = []
        for i in range(5):
            sid = new_id()
            await create_session(pool, sid)
            sids.append(sid)
            await asyncio.sleep(0.005)

        page1 = await list_sessions(pool, limit=2, offset=0)
        assert len(page1) == 2
        page2 = await list_sessions(pool, limit=2, offset=2)
        assert len(page2) == 2
        # Ensure no overlap at the extremes
        assert page1[0]["id"] != page2[0]["id"]


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

class TestMessages:
    async def test_insert_and_retrieve(self, pool):
        sid = new_id()
        await create_session(pool, sid, title="msg-test")
        mid = new_id()

        msg = await insert_message(pool, mid, sid, "user", "Hello world")
        assert msg["id"] == mid
        assert msg["content"] == "Hello world"
        assert msg["role"] == "user"

        messages = await get_messages(pool, sid)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello world"

    async def test_messages_auto_sequence(self, pool):
        sid = new_id()
        await create_session(pool, sid)

        msg0 = await insert_message(pool, new_id(), sid, "user", "first")
        msg1 = await insert_message(pool, new_id(), sid, "assistant", "second")
        msg2 = await insert_message(pool, new_id(), sid, "user", "third")

        assert msg0["sequence"] == 0
        assert msg1["sequence"] == 1
        assert msg2["sequence"] == 2

    async def test_messages_ordered_by_sequence(self, pool):
        sid = new_id()
        await create_session(pool, sid)

        mid_a = new_id()
        mid_c = new_id()
        mid_b = new_id()

        await insert_message(pool, mid_a, sid, "user", "A", sequence=0)
        await insert_message(pool, mid_c, sid, "user", "C", sequence=2)
        await insert_message(pool, mid_b, sid, "user", "B", sequence=1)

        messages = await get_messages(pool, sid)
        contents = [m["content"] for m in messages]
        assert contents == ["A", "B", "C"]

    async def test_insert_touches_session_updated_at(self, pool):
        sid = new_id()
        sess = await create_session(pool, sid)
        orig_updated = sess["updated_at"]
        await asyncio.sleep(0.01)

        await insert_message(pool, new_id(), sid, "user", "ping")
        updated_sess = await get_session(pool, sid)
        assert updated_sess["updated_at"] > orig_updated

    async def test_get_messages_with_after_sequence(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        for i in range(5):
            await insert_message(pool, new_id(), sid, "user", f"msg-{i}")

        recent = await get_messages(pool, sid, after_sequence=2)
        assert len(recent) == 2  # sequences 3 and 4
        assert recent[0]["content"] == "msg-3"
        assert recent[1]["content"] == "msg-4"

    async def test_get_messages_in_empty_session(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        messages = await get_messages(pool, sid)
        assert messages == []

    async def test_message_count(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        assert await get_message_count(pool, sid) == 0
        for i in range(3):
            await insert_message(pool, new_id(), sid, "user", str(i))
        assert await get_message_count(pool, sid) == 3

    async def test_delete_messages_all(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        for i in range(3):
            await insert_message(pool, new_id(), sid, "user", str(i))
        deleted = await delete_messages(pool, sid)
        assert deleted == 3
        assert await get_message_count(pool, sid) == 0

    async def test_delete_messages_before_sequence(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        for i in range(5):
            await insert_message(pool, new_id(), sid, "user", str(i), sequence=i)

        deleted = await delete_messages(pool, sid, before_sequence=3)
        assert deleted == 3  # sequences 0, 1, 2
        remaining = await get_messages(pool, sid)
        assert len(remaining) == 2
        assert remaining[0]["content"] == "3"
        assert remaining[1]["content"] == "4"

    async def test_message_metadata_field(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        meta = {"tokens": 42, "model": "gpt-4"}
        msg = await insert_message(pool, new_id(), sid, "user", "hi", metadata=meta)
        assert msg["metadata"] == meta

        messages = await get_messages(pool, sid)
        assert messages[0]["metadata"] == meta


# ---------------------------------------------------------------------------
# Metadata CRUD
# ---------------------------------------------------------------------------

class TestMetadata:
    async def test_set_and_get(self, pool):
        await set_metadata(pool, "agent", "last_run", "2026-06-13")
        val = await get_metadata(pool, "agent", "last_run")
        assert val == "2026-06-13"

    async def test_get_default(self, pool):
        val = await get_metadata(pool, "missing", "key", default=42)
        assert val == 42

    async def test_set_overwrites(self, pool):
        await set_metadata(pool, "scope", "key", "v1")
        await set_metadata(pool, "scope", "key", "v2")
        assert await get_metadata(pool, "scope", "key") == "v2"

    async def test_json_serialization(self, pool):
        complex_val = {"a": [1, 2, 3], "b": None, "c": True}
        await set_metadata(pool, "test", "complex", complex_val)
        fetched = await get_metadata(pool, "test", "complex")
        assert fetched == complex_val

    async def test_list_all(self, pool):
        await set_metadata(pool, "scope1", "k1", 1)
        await set_metadata(pool, "scope2", "k2", 2)
        entries = await list_metadata(pool)
        assert len(entries) == 2

    async def test_list_filtered_by_scope(self, pool):
        await set_metadata(pool, "scope_a", "x", 1)
        await set_metadata(pool, "scope_a", "y", 2)
        await set_metadata(pool, "scope_b", "z", 3)

        entries = await list_metadata(pool, scope="scope_a")
        assert len(entries) == 2
        keys = {e["key"] for e in entries}
        assert keys == {"x", "y"}

    async def test_delete_single_key(self, pool):
        await set_metadata(pool, "scope", "k1", "v1")
        await set_metadata(pool, "scope", "k2", "v2")
        deleted = await delete_metadata(pool, "scope", key="k1")
        assert deleted == 1
        assert await get_metadata(pool, "scope", "k1") is None
        assert await get_metadata(pool, "scope", "k2") == "v2"

    async def test_delete_entire_scope(self, pool):
        await set_metadata(pool, "scope", "a", 1)
        await set_metadata(pool, "scope", "b", 2)
        deleted = await delete_metadata(pool, "scope")
        assert deleted == 2
        entries = await list_metadata(pool, scope="scope")
        assert entries == []


# ---------------------------------------------------------------------------
# Edge cases & error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    async def test_large_message_content(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        large = "x" * 100_000
        msg = await insert_message(pool, new_id(), sid, "user", large)
        assert len(msg["content"]) == 100_000

        fetched = await get_messages(pool, sid)
        assert len(fetched[0]["content"]) == 100_000

    async def test_special_characters_in_content(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        special = "Hello\nWorld\tTab\u00e9\u00fc\u2603"
        await insert_message(pool, new_id(), sid, "user", special)
        fetched = await get_messages(pool, sid)
        assert fetched[0]["content"] == special

    async def test_empty_session_title(self, pool):
        sid = new_id()
        sess = await create_session(pool, sid, title="")
        assert sess["title"] == ""

    async def test_concurrent_writes(self, tmp_db_path):
        """Multiple coroutines writing to the same session concurrently."""
        p = AsyncSQLitePool(tmp_db_path, min_size=2, max_size=4)
        await p.initialize()
        await ensure_schema(p)

        sid = new_id()
        await create_session(p, sid)

        async def write_msg(i: int):
            msg_id = new_id()
            await insert_message(p, msg_id, sid, "user", f"concurrent-{i}")

        await asyncio.gather(*[write_msg(i) for i in range(10)])

        count = await get_message_count(p, sid)
        assert count == 10
        await p.close()

    async def test_pool_reuses_connections(self, pool):
        """Acquire/release cycle should return connection to pool."""
        active_before = pool.active_count
        async with pool.acquire():
            assert pool.active_count == active_before + 1
        assert pool.active_count == active_before

    async def test_explicit_sequence_respected(self, pool):
        sid = new_id()
        await create_session(pool, sid)
        # Insert with explicit sequence that skips ahead
        msg = await insert_message(pool, new_id(), sid, "user", "skip", sequence=99)
        assert msg["sequence"] == 99
        # Next auto should be after the explicit
        msg2 = await insert_message(pool, new_id(), sid, "user", "after")
        assert msg2["sequence"] == 100


# ---------------------------------------------------------------------------
# session_context
# ---------------------------------------------------------------------------

class TestSessionContext:
    async def test_creates_new_session(self, pool):
        sid = new_id()
        async with session_context(pool, sid, title="ctx test") as s:
            assert s["id"] == sid
            assert s["title"] == "ctx test"
        # Session should persist after context exits
        fetched = await get_session(pool, sid)
        assert fetched is not None

    async def test_reuses_existing_session(self, pool):
        sid = new_id()
        await create_session(pool, sid, title="original")
        async with session_context(pool, sid) as s:
            assert s["title"] == "original"

    async def test_touches_updated_at_on_exit(self, pool):
        sid = new_id()
        sess = await create_session(pool, sid)
        orig = sess["updated_at"]
        await asyncio.sleep(0.02)
        async with session_context(pool, sid):
            pass
        updated = await get_session(pool, sid)
        assert updated["updated_at"] > orig