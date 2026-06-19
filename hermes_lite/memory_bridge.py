"""
hermes_lite/memory_bridge.py — Cross-session memory layer (T9)

A boring, atomic, file-backed SQLite layer for two concerns:

- ``memory``: agent notes that should survive a session restart but may be
  ephemeral (best practices, project context, recent decisions).
- ``user``: stable profile facts about the human user (preferences,
  environment, name, role). Higher bar for changes.

Schema is intentionally flat (no FTS, no embeddings) so the small-model
attention budget isn't blown out by accident. The ``load_into_prompt``
helper returns a formatted block sized for a system message slot, with
a footer when truncation was applied.

Public API
----------
MemoryBridge(db_path=...)
    add(target, content) -> int
    replace(target, old_text, content) -> int          # unique match required
    remove(target, old_text) -> int                    # unique match required
    load_into_prompt(max_chars=800) -> str
    list(target) -> list[MemoryEntry]
    close()

Errors
------
``MemoryBridgeError`` is raised on uniqueness violations and missing
uniqueness matches. The tools_builtins ``memory`` tool handler
translates these into the standard ``{"ok": False, "error": str}``
envelope, consistent with the other essentials.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 1

DEFAULT_DB_PATH = Path.home() / ".hermes_lite" / "memory.db"


CREATE_MEMORY_BRIDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    target      TEXT    NOT NULL CHECK (target IN ('memory', 'user')),
    content     TEXT    NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
"""

CREATE_MEMORY_BRIDGE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_target "
    "ON memory_entries(target);",
)

CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS memory_schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  INTEGER NOT NULL
);
"""


class MemoryBridgeError(Exception):
    """Non-fatal error raised by the bridge for uniqueness / missing-match cases.

    The tool handler catches this and returns ``{"ok": False, "error": str}``.
    Raise (don't return ``None``) so the contract is explicit.
    """


@dataclass
class MemoryEntry:
    """A single memory row."""

    id: int
    target: str
    content: str
    created_at: int
    updated_at: int


def _now() -> int:
    """Wall-clock seconds — matches the rest of hermes_lite."""
    return int(time.time())


class MemoryBridge:
    """SQLite-backed cross-session memory layer.

    Thread-safe via an internal lock. Connections are opened in
    ``isolation_level=None`` mode (autocommit) so we can issue explicit
    transactions where we need them (replace/remove uniqueness checks).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            # Defer to Path.home() each construction so test fixtures that
            # monkeypatch the classmethod take effect.
            resolved = Path.home() / ".hermes_lite" / "memory.db"
        else:
            resolved = Path(db_path).expanduser()
        self.db_path = resolved
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,  # autocommit; we BEGIN EXPLICITLY where needed
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "MemoryBridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- schema -----------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Apply the bridge schema (idempotent)."""
        with self._lock:
            self._conn.executescript(CREATE_MEMORY_BRIDGE_SCHEMA)
            self._conn.executescript(CREATE_SCHEMA_VERSION_TABLE)
            for stmt in CREATE_MEMORY_BRIDGE_INDEXES:
                self._conn.execute(stmt)
            row = self._conn.execute(
                "SELECT version FROM memory_schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO memory_schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, _now()),
                )

    # -- CRUD --------------------------------------------------------------

    def add(self, target: str, content: str) -> int:
        """Append a new entry. Returns the inserted row id."""
        if target not in ("memory", "user"):
            raise MemoryBridgeError(f"target must be 'memory' or 'user', got {target!r}")
        if not content:
            raise MemoryBridgeError("content must be non-empty for add")
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memory_entries (target, content, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (target, content, now, now),
            )
            return int(cur.lastrowid or 0)

    def replace(self, target: str, old_text: str, content: str) -> int:
        """Replace the unique entry whose ``content`` equals ``old_text``.

        Raises MemoryBridgeError if no entry matches or more than one does.
        Returns the id of the updated row.
        """
        if target not in ("memory", "user"):
            raise MemoryBridgeError(f"target must be 'memory' or 'user', got {target!r}")
        if not old_text:
            raise MemoryBridgeError("old_text is required for replace")
        now = _now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM memory_entries WHERE target = ? AND content = ? ORDER BY id",
                (target, old_text),
            ).fetchall()
            if not rows:
                raise MemoryBridgeError(
                    f"old_text not found in {target}: {old_text!r}"
                )
            if len(rows) > 1:
                raise MemoryBridgeError(
                    f"old_text matches {len(rows)} entries in {target}; narrow it first"
                )
            (row_id,) = rows[0]
            self._conn.execute(
                "UPDATE memory_entries SET content = ?, updated_at = ? WHERE id = ?",
                (content, now, row_id),
            )
            return int(row_id)

    def remove(self, target: str, old_text: str) -> int:
        """Remove the unique entry whose ``content`` equals ``old_text``.

        Raises MemoryBridgeError if no entry matches or more than one does.
        Returns the count removed (0 or 1, but only 1 on success path).
        """
        if target not in ("memory", "user"):
            raise MemoryBridgeError(f"target must be 'memory' or 'user', got {target!r}")
        if not old_text:
            raise MemoryBridgeError("old_text is required for remove")
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM memory_entries WHERE target = ? AND content = ? ORDER BY id",
                (target, old_text),
            ).fetchall()
            if not rows:
                raise MemoryBridgeError(
                    f"old_text not found in {target}: {old_text!r}"
                )
            if len(rows) > 1:
                raise MemoryBridgeError(
                    f"old_text matches {len(rows)} entries in {target}; narrow it first"
                )
            (row_id,) = rows[0]
            self._conn.execute(
                "DELETE FROM memory_entries WHERE id = ?",
                (row_id,),
            )
            return 1

    def list(self, target: str) -> list[MemoryEntry]:
        """Return all entries for a target, ordered by id."""
        if target not in ("memory", "user"):
            raise MemoryBridgeError(f"target must be 'memory' or 'user', got {target!r}")
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, target, content, created_at, updated_at "
                "FROM memory_entries WHERE target = ? ORDER BY id",
                (target,),
            ).fetchall()
        return [
            MemoryEntry(
                id=r[0], target=r[1], content=r[2], created_at=r[3], updated_at=r[4]
            )
            for r in rows
        ]

    # -- prompt formatting -------------------------------------------------

    # Default block format:
    #
    #   <hermes_lite_memory>
    #   # MEMORY (agent notes)
    #   - fact 1
    #   - fact 2
    #
    #   # USER (stable profile facts)
    #   - fact A
    #   - fact B
    #   ... and N more facts
    #   </hermes_lite_memory>
    #
    # Truncation strategy: render the full block; if over budget, drop
    # tail entries one at a time (favoring user facts over memory facts)
    # and insert a "... and N more facts" footer line just before the
    # closing tag (so the closing tag always terminates the block).

    FOOTER_TEMPLATE = "... and {n} more facts"

    def load_into_prompt(self, max_chars: int = 800) -> str:
        """Return a formatted memory block ≤ ``max_chars`` characters.

        Always returns a string (possibly empty) — never raises on an
        empty DB or missing file. Truncation appends a footer note when
        the full set of facts would overflow the budget.
        """
        if max_chars < 0:
            raise ValueError("max_chars must be >= 0")

        memory_entries = self.list("memory")
        user_entries = self.list("user")

        if not memory_entries and not user_entries:
            return ""

        full = self._render_block(memory_entries, user_entries)
        if len(full) <= max_chars:
            return full

        return self._truncate_block(memory_entries, user_entries, max_chars)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _render_block(
        memory_entries: list[MemoryEntry],
        user_entries: list[MemoryEntry],
        footer: str | None = None,
    ) -> str:
        """Render the prompt block. Optional footer line is placed just
        before the closing ``</hermes_lite_memory>`` tag, when present.
        """
        lines = ["<hermes_lite_memory>"]
        if memory_entries:
            lines.append("# MEMORY (agent notes)")
            for e in memory_entries:
                lines.append(f"- {e.content}")
        if user_entries:
            if memory_entries:
                lines.append("")
            lines.append("# USER (stable profile facts)")
            for e in user_entries:
                lines.append(f"- {e.content}")
        if footer:
            if memory_entries or user_entries:
                lines.append("")
            lines.append(footer)
        lines.append("</hermes_lite_memory>")
        return "\n".join(lines)

    @classmethod
    def _truncate_block(
        cls,
        memory_entries: list[MemoryEntry],
        user_entries: list[MemoryEntry],
        max_chars: int,
    ) -> str:
        """Drop entries from the tail until the block fits, then add footer.

        Drop order: memory entries first, then user entries (preserve
        user-profile facts longer). The footer is rendered as the last
        line before the closing tag so the closing tag always terminates
        the returned string.
        """
        kept_memory = list(memory_entries)
        kept_user = list(user_entries)
        dropped = 0

        # Worst-case footer width (6-digit estimate); if the full block
        # already exceeds max_chars even with all entries removed and the
        # tightest footer, we surface a footer-only block.
        worst_footer_len = len(cls.FOOTER_TEMPLATE.format(n=999999))
        # Try progressively drop and re-check using a footer count that
        # actually fits the remaining budget.
        while True:
            n_dropped = len(memory_entries) - len(kept_memory) + (
                len(user_entries) - len(kept_user)
            )
            # Footer count = entries dropped since first truncation round.
            # If we kept everything, dropped=0 and we render no footer.
            if n_dropped == 0:
                candidate = cls._render_block(kept_memory, kept_user)
            else:
                footer_text = cls.FOOTER_TEMPLATE.format(n=n_dropped)
                candidate = cls._render_block(kept_memory, kept_user, footer=footer_text)
            if len(candidate) <= max_chars:
                return candidate
            # Try a worst-case footer length to see if even that'd fit:
            wc_footer = cls.FOOTER_TEMPLATE.format(n=999999)
            wc_candidate = cls._render_block(kept_memory, kept_user, footer=wc_footer)
            if len(wc_candidate) <= max_chars:
                # With this kept set the block fits even if N=999999 —
                # so render with the actual drop count.
                return wc_candidate
            # Drop one more entry — prefer memory first.
            if kept_memory:
                kept_memory.pop()
            elif kept_user:
                kept_user.pop()
            else:
                # Nothing left — footer-only block.
                footer_only = cls.FOOTER_TEMPLATE.format(n=max(n_dropped, 1))
                return cls._render_block([], [], footer=footer_only)

        # Unreachable; loop above always returns. Silence pylint.

# ---------------------------------------------------------------------------
# Module-level convenience handle
# ---------------------------------------------------------------------------


def _default_db_path() -> Path:
    """Resolve the bridge DB path at call-time so HOME/PATH.home() mocking
    in tests takes effect."""
    return Path.home() / ".hermes_lite" / "memory.db"


_default_bridge: Optional[MemoryBridge] = None
_default_bridge_lock = threading.Lock()


def get_default_bridge() -> MemoryBridge:
    """Return a process-wide singleton bridge rooted at
    ``$HOME/.hermes_lite/memory.db`` (resolved lazily so the path tracks
    any HOME / ``Path.home()`` override in effect when this is called).

    The orchestrator and tool handler both share this; tests should
    instantiate ``MemoryBridge(db_path=...)`` directly with a
    ``tmp_path`` fixture instead of relying on this singleton.
    """
    global _default_bridge
    with _default_bridge_lock:
        if _default_bridge is None:
            _default_bridge = MemoryBridge(db_path=_default_db_path())
        return _default_bridge


def reset_default_bridge() -> None:
    """Drop the cached default bridge (used by tests when HOME changes)."""
    global _default_bridge
    with _default_bridge_lock:
        if _default_bridge is not None:
            _default_bridge.close()
            _default_bridge = None


__all__ = [
    "MemoryBridge",
    "MemoryBridgeError",
    "MemoryEntry",
    "get_default_bridge",
    "reset_default_bridge",
    "DEFAULT_DB_PATH",
    "SCHEMA_VERSION",
]
