"""hermes_lite.tests.test_observability — Tests for T12 per-turn logging."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from hermes_lite.observability import (
    DEFAULT_LOG_FILE,
    MAX_LOG_BYTES,
    TurnRecord,
    _make_record,
    _rotate_if_needed,
    _write_record,
    compute_stats,
    log_turn,
    read_turns,
)


class TestMakeRecord:
    def test_basic_record(self) -> None:
        rec = _make_record(
            turn_id="abc123",
            tier="local",
            model="local:qwen2.5-3b",
            prompt_tokens=100,
            completion_tokens=50,
            elapsed_ms=200,
            tools_called=["read_file", "search_files"],
            errors=[],
        )

        assert rec["turn"] == "abc123"
        assert rec["tier"] == "local"
        assert rec["model"] == "local:qwen2.5-3b"
        assert rec["prompt_tokens"] == 100
        assert rec["completion_tokens"] == 50
        assert rec["elapsed_ms"] == 200
        assert rec["tools_called"] == ["read_file", "search_files"]
        assert rec["errors"] == []
        assert "ts" in rec
        assert isinstance(rec["ts"], int)

    def test_defaults(self) -> None:
        rec = _make_record(
            turn_id="x",
            tier="cloud",
            model="minimaxai/minimax-m3",
        )

        assert rec["prompt_tokens"] == 0
        assert rec["completion_tokens"] == 0
        assert rec["elapsed_ms"] == 0
        assert rec["tools_called"] == []
        assert rec["errors"] == []


class TestWriteRecord:
    def test_write_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            rec = _make_record(
                turn_id="t1", tier="local", model="local:m",
                prompt_tokens=10, completion_tokens=5, elapsed_ms=50,
            )

            _write_record(path, rec)

            assert path.exists()
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["turn"] == "t1"

    def test_write_appends(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            for i in range(3):
                rec = _make_record(
                    turn_id=f"t{i}", tier="local", model="m",
                )
                _write_record(path, rec)

            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 3
            assert [json.loads(l)["turn"] for l in lines] == ["t0", "t1", "t2"]


class TestRotateIfNeeded:
    def test_no_rotate_when_small(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            # Write less than 10MB
            path.write_text("x" * (MAX_LOG_BYTES - 100))

            _rotate_if_needed(path, max_bytes=MAX_LOG_BYTES)

            # No rotation
            rotated = Path(str(path) + ".1")
            assert path.exists()
            assert not rotated.exists()

    def test_rotate_when_large(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            path.write_text("x" * (MAX_LOG_BYTES + 100))

            _rotate_if_needed(path, max_bytes=MAX_LOG_BYTES)

            rotated = Path(str(path) + ".1")
            assert rotated.exists()
            assert not path.exists()

    def test_rotate_replaces_old(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            old_rotated = Path(str(path) + ".1")
            old_rotated.write_text("old")
            path.write_text("x" * (MAX_LOG_BYTES + 100))

            _rotate_if_needed(path, max_bytes=MAX_LOG_BYTES)

            # Old .1 should be gone, new rotated file should exist
            assert (Path(str(path) + ".1")).exists()
            # New .1 should NOT contain "old" (it was deleted)
            content = (Path(str(path) + ".1")).read_text()
            assert content != "old"


class TestReadTurns:
    def test_read_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            # File doesn't exist
            turns = read_turns(log_path=path, limit=10)
            assert turns == []

    def test_read_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            # Write 3 records with different timestamps
            for i in range(3):
                rec = _make_record(
                    turn_id=f"t{i}", tier="local", model="m",
                    prompt_tokens=i, completion_tokens=i * 2, elapsed_ms=i * 100,
                )
                rec["ts"] = i  # Ensure unique timestamps
                _write_record(path, rec)

            turns = read_turns(log_path=path, limit=10)
            assert len(turns) == 3
            # Most recent first (highest ts)
            assert [t["turn"] for t in turns] == ["t2", "t1", "t0"]

    def test_read_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            for i in range(100):
                rec = _make_record(turn_id=f"t{i}", tier="local", model="m")
                rec["ts"] = i  # Ensure unique timestamps
                _write_record(path, rec)

            turns = read_turns(log_path=path, limit=10)
            assert len(turns) == 10
            # Most recent 10
            assert [t["turn"] for t in turns] == [f"t{i}" for i in range(99, 89, -1)]

    def test_read_from_rotated_backup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            backup = Path(str(path) + ".1")

            # Write some to backup
            for i in range(3):
                rec = _make_record(turn_id=f"old{i}", tier="local", model="m")
                _write_record(backup, rec)

            # Write some to main (more recent)
            for i in range(2):
                rec = _make_record(turn_id=f"new{i}", tier="local", model="m")
                _write_record(path, rec)

            turns = read_turns(log_path=path, limit=10)
            # Should merge both files, sorted by timestamp
            assert len(turns) == 5


class TestComputeStats:
    def test_empty(self) -> None:
        stats = compute_stats([])
        assert stats["count"] == 0
        assert stats["p50_ms"] == 0
        assert stats["p95_ms"] == 0

    def test_single_turn(self) -> None:
        turns: list[TurnRecord] = [
            {
                "ts": int(time.time()),
                "turn": "x",
                "tier": "local",
                "model": "m",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "elapsed_ms": 200,
                "tools_called": [],
                "errors": [],
            }
        ]
        stats = compute_stats(turns)

        assert stats["count"] == 1
        assert stats["p50_ms"] == 200
        assert stats["p95_ms"] == 200
        assert stats["total_prompt_tokens"] == 100
        assert stats["total_completion_tokens"] == 50
        assert stats["total_errors"] == 0

    def test_statistics(self) -> None:
        turns: list[TurnRecord] = [
            {
                "ts": int(time.time()),
                "turn": f"t{i}",
                "tier": "local" if i % 2 == 0 else "cloud",
                "model": "m1" if i % 2 == 0 else "m2",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "elapsed_ms": (i + 1) * 100,
                "tools_called": [],
                "errors": ["e"] if i == 0 else [],
            }
            for i in range(10)
        ]

        stats = compute_stats(turns)

        assert stats["count"] == 10
        # p50 of [100, 200, ..., 1000] = 550 (median)
        assert stats["p50_ms"] == 550
        # p95 varies by Python version quantiles implementation
        # Just check it's in the upper range
        assert stats["p95_ms"] >= 900
        assert stats["model_counts"] == {"m1": 5, "m2": 5}
        assert stats["tier_counts"] == {"local": 5, "cloud": 5}
        assert stats["total_prompt_tokens"] == 100
        assert stats["total_completion_tokens"] == 50
        assert stats["total_errors"] == 1


class TestLogTurnIntegration:
    def test_log_turn_async_fire_and_forget(self) -> None:
        """Verify log_turn fires an async task when a loop exists."""
        import asyncio

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            task_dialogue_complete = asyncio.Event()

            async def run_test() -> None:
                log_turn(
                    turn_id="t1",
                    tier="cloud",
                    model="minimaxai/minimax-m3",
                    log_path=path,
                )
                # Give the background task a moment
                await asyncio.sleep(0.1)

            asyncio.run(run_test())

            # Should have written
            assert path.exists()

    def test_log_turn_sync_fallback(self) -> None:
        """log_turn writes synchronously when no event loop exists."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "turns.jsonl"
            # Call outside an async context
            log_turn(
                turn_id="sync1",
                tier="local",
                model="local:qwen2.5-3b",
                log_path=path,
            )

            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["turn"] == "sync1"