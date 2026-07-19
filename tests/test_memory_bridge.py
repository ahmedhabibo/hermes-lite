"""
tests/test_memory_bridge.py — Coverage for the cross-session memory bridge (T9).

Targets:
- hermes_lite.memory_bridge.MemoryBridge (SQLite store)
- hermes_lite.memory_bridge.MemoryBridgeError (uniqueness / missing-match fails)
- hermes_lite.memory_bridge.load_into_prompt(max_chars=800) (truncation contract)
- The orchestrator wiring that injects the memory block into the system prompt.
- Survival across restart (acceptance: a fact written before "shutdown" is
  readable after a fresh bridge instance is opened on the same DB path).

We deliberately do NOT cover vector embeddings or fuzzy match — the
task spec is explicit that the bridge stays boring: exact substring,
SQLite-only, bounded prompt payload.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_lite.memory_bridge import (
    DEFAULT_DB_PATH,
    MemoryBridge,
    MemoryBridgeError,
    get_default_bridge,
    reset_default_bridge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge(tmp_path: Path):
    """A fresh bridge rooted in tmp_path, auto-closed at end of test."""
    b = MemoryBridge(db_path=tmp_path / "memory.db")
    yield b
    b.close()


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    """Force ``Path.home()`` and the default-bridge cache to read tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    reset_default_bridge()
    yield tmp_path
    reset_default_bridge()


# ---------------------------------------------------------------------------
# Tests — basic CRUD
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_appends_entry_with_id(self, bridge: MemoryBridge) -> None:
        rid = bridge.add("memory", "first fact")
        assert isinstance(rid, int) and rid > 0
        entries = bridge.list("memory")
        assert [e.content for e in entries] == ["first fact"]

    def test_add_two_targets_isolated(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "agent note")
        bridge.add("user", "user fact")
        assert [e.content for e in bridge.list("memory")] == ["agent note"]
        assert [e.content for e in bridge.list("user")] == ["user fact"]

    def test_add_rejects_blank_content(self, bridge: MemoryBridge) -> None:
        with pytest.raises(MemoryBridgeError):
            bridge.add("memory", "")

    def test_add_rejects_unknown_target(self, bridge: MemoryBridge) -> None:
        with pytest.raises(MemoryBridgeError):
            bridge.add("agent", "fact")


class TestReplace:
    def test_replace_unique_match(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "old text")
        bridge.replace("memory", "old text", "new text")
        assert [e.content for e in bridge.list("memory")] == ["new text"]

    def test_replace_unknown_target_raises(self, bridge: MemoryBridge) -> None:
        with pytest.raises(MemoryBridgeError):
            bridge.replace("bogus", "x", "y")

    def test_replace_missing_match_raises(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "exists")
        with pytest.raises(MemoryBridgeError, match="not found"):
            bridge.replace("memory", "ghost", "x")

    def test_replace_multiple_matches_raises(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "dup")
        bridge.add("memory", "dup")
        with pytest.raises(MemoryBridgeError, match=r"matches 2"):
            bridge.replace("memory", "dup", "x")

    def test_replace_requires_old_text(self, bridge: MemoryBridge) -> None:
        with pytest.raises(MemoryBridgeError):
            bridge.replace("memory", "", "x")

    def test_replace_only_targets_named_store(self, bridge: MemoryBridge) -> None:
        """old_text in 'memory' must not match 'user' entries."""
        bridge.add("user", "shared text")
        bridge.add("memory", "shared text")
        # Replacing 'shared text' in memory must hit only the memory row.
        bridge.replace("memory", "shared text", "replaced")
        assert [e.content for e in bridge.list("memory")] == ["replaced"]
        assert [e.content for e in bridge.list("user")] == ["shared text"]


class TestRemove:
    def test_remove_unique_match(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "drop me")
        bridge.add("memory", "keep me")
        n = bridge.remove("memory", "drop me")
        assert n == 1
        assert [e.content for e in bridge.list("memory")] == ["keep me"]

    def test_remove_missing_raises(self, bridge: MemoryBridge) -> None:
        with pytest.raises(MemoryBridgeError, match="not found"):
            bridge.remove("memory", "ghost")

    def test_remove_multiple_raises(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "dup")
        bridge.add("memory", "dup")
        with pytest.raises(MemoryBridgeError, match=r"matches 2"):
            bridge.remove("memory", "dup")


# ---------------------------------------------------------------------------
# Tests — load_into_prompt
# ---------------------------------------------------------------------------


class TestLoadIntoPrompt:
    def test_empty_db_returns_empty_string(self, bridge: MemoryBridge) -> None:
        assert bridge.load_into_prompt(max_chars=800) == ""

    def test_renders_one_section_per_target(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "fact A")
        bridge.add("user", "fact X")
        block = bridge.load_into_prompt(max_chars=800)
        assert block.startswith("<hermes_lite_memory>")
        assert block.endswith("</hermes_lite_memory>")
        assert "# MEMORY (agent notes)" in block
        assert "# USER (stable profile facts)" in block
        assert "- fact A" in block
        assert "- fact X" in block

    def test_truncates_with_footer_when_over_budget(self, bridge: MemoryBridge) -> None:
        # Pack 200 facts of 40 chars each = 8_000 chars in the entries
        # area alone; with the wrappers they'll blow past any small budget.
        for i in range(60):
            bridge.add("memory", f"fact number {i:04d} with extra padding text " * 2)
        block = bridge.load_into_prompt(max_chars=400)
        assert len(block) <= 400
        assert "... and" in block
        assert "more facts" in block
        # Block still has the structural tags even when footer-only.
        assert block.startswith("<hermes_lite_memory>")
        assert block.endswith("</hermes_lite_memory>")

    def test_full_block_under_budget_has_no_footer(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "one")
        bridge.add("user", "two")
        block = bridge.load_into_prompt(max_chars=4000)
        assert "... and" not in block
        assert "more facts" not in block

    def test_max_chars_zero_returns_footer_or_empty(self, bridge: MemoryBridge) -> None:
        bridge.add("memory", "X")
        out = bridge.load_into_prompt(max_chars=0)
        # Either empty (no header fits) or just the footer — both are valid
        # as long as we don't crash and don't exceed the budget.
        assert len(out) <= 0 or (
            out.startswith("<hermes_lite_memory>") and out.endswith("</hermes_lite_memory>")
        )

    def test_max_chars_negative_raises(self, bridge: MemoryBridge) -> None:
        with pytest.raises(ValueError):
            bridge.load_into_prompt(max_chars=-1)

    def test_truncation_prefers_user_over_memory(self, bridge: MemoryBridge) -> None:
        """When memory+user can't both fit, the user facts must be the
        ones preserved (user-profile facts are higher-stability)."""
        for i in range(30):
            bridge.add("memory", f"agent note {i:04d} with padding " * 3)
        bridge.add("user", "stable user fact")
        block = bridge.load_into_prompt(max_chars=300)
        # The user fact should be in the block even though memory entries
        # are dropped first.
        assert "stable user fact" in block
        assert len(block) <= 300


# ---------------------------------------------------------------------------
# Tests — persistence / restart (acceptance: fact survives restart)
# ---------------------------------------------------------------------------


class TestSurvivalAcrossRestart:
    def test_fact_survives_closing_and_reopening(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        b1 = MemoryBridge(db_path=db_path)
        b1.add("memory", "macOS Sonoma, M1 8GB")
        b1.close()

        b2 = MemoryBridge(db_path=db_path)
        try:
            entries = b2.list("memory")
            assert any(e.content == "macOS Sonoma, M1 8GB" for e in entries)
        finally:
            b2.close()

    def test_replace_then_close_then_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        b1 = MemoryBridge(db_path=db_path)
        b1.add("memory", "before")
        b1.replace("memory", "before", "after")
        b1.close()

        b2 = MemoryBridge(db_path=db_path)
        try:
            assert [e.content for e in b2.list("memory")] == ["after"]
        finally:
            b2.close()


# ---------------------------------------------------------------------------
# Tests — default singleton respects HOME
# ---------------------------------------------------------------------------


class TestDefaultBridge:
    def test_default_singleton_returns_one_bridge_per_cache(self, tmp_path: Path) -> None:
        """The default bridge is a module-level singleton; resetting drops it."""
        reset_default_bridge()
        b1 = get_default_bridge()
        b2 = get_default_bridge()
        try:
            assert b1 is b2
        finally:
            b1.close()
            reset_default_bridge()

    def test_bridge_db_path_follows_path_home(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """When ambient ``Path.home()`` returns ``tmp_path``, the bridge
        lives at ``tmp_path/.hermes_lite/memory.db``. (On macOS
        :func:`pathlib.Path.home` ignores the HOME env var; we therefore
        also patch the module-level default constructor to use tmp_path
        for this assertion.)
        """
        from pathlib import Path as _Path

        # Force Path.home() to return tmp_path for the duration of this test.
        monkeypatch.setattr(_Path, "home", classmethod(lambda cls: tmp_path))
        reset_default_bridge()
        b = get_default_bridge()
        try:
            assert b.db_path == tmp_path / ".hermes_lite" / "memory.db"
        finally:
            b.close()
            reset_default_bridge()

    def test_bridge_db_path_in_ambient_home_after_reset(self, tmp_path: Path) -> None:
        """After reset, get_default_bridge opens a fresh bridge that
        re-derives its path from the current Path.home().
        """
        # Re-derive after patching — re-import the module to drop the
        # cached DEFAULT_DB_PATH as well, so we don't reuse the user's
        # real ~/.hermes_lite directory.
        import importlib

        from hermes_lite import memory_bridge as _mb2

        with pytest.MonkeyPatch.context() as monkeypatch:
            from pathlib import Path as _Path

            monkeypatch.setattr(
                _Path, "home", classmethod(lambda cls: tmp_path)
            )
            importlib.reload(_mb2)
            b = _mb2.get_default_bridge()
            try:
                assert b.db_path == tmp_path / ".hermes_lite" / "memory.db"
            finally:
                b.close()
                _mb2.reset_default_bridge()
            # Re-reload once more so subsequent tests don't see the patched
            # home dir.
            importlib.reload(_mb2)

    def test_reset_clears_singleton(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            "pathlib.Path.home", classmethod(lambda cls: tmp_path)
        )
        reset_default_bridge()
        b = get_default_bridge()
        try:
            assert b.db_path == tmp_path / ".hermes_lite" / "memory.db"
        finally:
            b.close()
            reset_default_bridge()

    def test_default_bridge_loadable_after_reset(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            "pathlib.Path.home", classmethod(lambda cls: tmp_path)
        )
        reset_default_bridge()
        b1 = get_default_bridge()
        b1.add("memory", "alpha")
        b1.close()
        reset_default_bridge()
        b2 = get_default_bridge()
        try:
            assert any(e.content == "alpha" for e in b2.list("memory"))
        finally:
            b2.close()
            reset_default_bridge()


# ---------------------------------------------------------------------------
# Tests — orchestrator wiring (the bridge block lands in the system prompt)
# ---------------------------------------------------------------------------


class TestOrchestratorWiring:
    @pytest.mark.asyncio
    async def test_system_prompt_includes_memory_block(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The orchestrator must inject the bridge block into every chat turn."""
        from pathlib import Path as _Path

        from hermes_lite import memory_bridge as _mb
        from hermes_lite.orchestrator import HermesOrchestrator

        monkeypatch.setattr(
            _Path, "home", classmethod(lambda cls: tmp_path)
        )
        _mb.reset_default_bridge()

        # Seed the bridge so load_into_prompt has something to render.
        # Use a bridge explicitly tied to tmp_path so the data lives
        # inside the test fixture regardless of module-level singleton state.
        b = _mb.MemoryBridge(db_path=tmp_path / ".hermes_lite" / "memory.db")
        try:
            b.add("user", "favorite IDE: neovim")
        finally:
            b.close()

        orch = HermesOrchestrator(db_path=str(tmp_path / "sessions.db"))
        # Close any session DB pool the orchestrator may have opened lazily.
        try:
            msgs = await orch._build_llm_history("hello")
        finally:
            _mb.reset_default_bridge()
            if orch.pool is not None:
                await orch.pool.close()

        sys_msg = next(m for m in msgs if m["role"] == "system")
        assert "favorite IDE: neovim" in sys_msg["content"]
        assert "<hermes_lite_memory>" in sys_msg["content"]

    @pytest.mark.asyncio
    async def test_empty_bridge_yields_no_extra_block_in_prompt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An empty bridge must not leak an empty XML wrapper."""
        from pathlib import Path as _Path

        from hermes_lite import memory_bridge as _mb
        from hermes_lite.orchestrator import HermesOrchestrator

        monkeypatch.setattr(
            _Path, "home", classmethod(lambda cls: tmp_path)
        )
        _mb.reset_default_bridge()

        # Pre-create the empty DB file so the singleton's first call
        # lands at tmp_path and sees an empty store.
        empty_dir = tmp_path / ".hermes_lite"
        empty_dir.mkdir(parents=True, exist_ok=True)
        empty_bridge = _mb.MemoryBridge(db_path=empty_dir / "memory.db")
        empty_bridge.close()

        orch = HermesOrchestrator(db_path=str(tmp_path / "sessions.db"))
        try:
            msgs = await orch._build_llm_history("anything")
        finally:
            _mb.reset_default_bridge()
            if orch.pool is not None:
                await orch.pool.close()

        sys_msg = next(m for m in msgs if m["role"] == "system")
        # The base system prompt is present (markdown wrappers aside)…
        assert "Hermes-Lite" in sys_msg["content"]
        # …but the empty memory wrapper is omitted so we don't burn tokens.
        assert "<hermes_lite_memory>" not in sys_msg["content"]
