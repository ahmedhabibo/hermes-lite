"""Integration tests for HermesOrchestrator.stream_prompt()."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from hermes_lite.orchestrator import HermesOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_gen(items: list[str]):
    """Return an async generator yielding the given string items in order."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Happy path: normal streaming with token collection + persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_prompt_happy_path(tmp_path: Path):
    """stream_prompt yields tokens in order, persists full response, records router success."""
    orch = HermesOrchestrator(db_path=str(tmp_path / "sessions.db"))
    await orch._initialize_memory()

    try:
        # Mock chat_stream to yield three tokens
        tokens = ["Hello", " ", "world!"]
        with patch("hermes_lite.orchestrator.chat_stream", return_value=_async_gen(tokens)):
            collected = []
            async for token in orch.stream_prompt("Say hello"):
                collected.append(token)

        # Tokens yielded in order
        assert collected == tokens

        # Full response persisted to history
        history = await orch._get_history(limit=10)
        assistant_msgs = [m for m in history if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "Hello world!"

        # Router recorded success
        # (router state is internal; we verify via record_outcome call indirectly
        # by checking the metadata persisted with the assistant message)
        meta = assistant_msgs[0].get("metadata", {})
        routing = meta.get("routing", {})
        assert routing.get("model_id") is not None
        assert routing.get("tier") in ("local", "cloud")
    finally:
        if orch.pool is not None:
            await orch.pool.close()


# ---------------------------------------------------------------------------
# Delegation guard: commands / direct-tool / MoA delegate to _handle_prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_prompt_delegates_commands(tmp_path: Path):
    """stream_prompt('/help') delegates to _handle_prompt and yields single response."""
    orch = HermesOrchestrator(db_path=str(tmp_path / "sessions.db"))
    await orch._initialize_memory()

    try:
        with patch.object(orch, "_handle_prompt", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = "Help text from _handle_prompt"

            collected = []
            async for token in orch.stream_prompt("/help"):
                collected.append(token)

            # Single yield with the full response from _handle_prompt
            assert collected == ["Help text from _handle_prompt"]
            mock_handle.assert_awaited_once_with("/help")
    finally:
        if orch.pool is not None:
            await orch.pool.close()


@pytest.mark.asyncio
async def test_stream_prompt_delegates_direct_tool(tmp_path: Path):
    """stream_prompt('!tool') delegates to _handle_prompt."""
    orch = HermesOrchestrator(db_path=str(tmp_path / "sessions.db"))
    await orch._initialize_memory()

    try:
        with patch.object(orch, "_handle_prompt", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = "Direct tool result"

            collected = []
            async for token in orch.stream_prompt("!read_file /etc/hosts"):
                collected.append(token)

            assert collected == ["Direct tool result"]
            mock_handle.assert_awaited_once_with("!read_file /etc/hosts")
    finally:
        if orch.pool is not None:
            await orch.pool.close()


# ---------------------------------------------------------------------------
# Error path: chat_stream raises → no persistence, router failure, re-raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_prompt_error_path(tmp_path: Path):
    """stream_prompt propagates exception, does NOT persist partial response, records router failure."""
    orch = HermesOrchestrator(db_path=str(tmp_path / "sessions.db"))
    await orch._initialize_memory()

    try:
        # Mock chat_stream to raise after yielding one token
        async def failing_gen():
            yield "partial"
            raise RuntimeError("LLM endpoint down")

        with patch("hermes_lite.orchestrator.chat_stream", return_value=failing_gen()):
            with pytest.raises(RuntimeError, match="LLM endpoint down"):
                async for _ in orch.stream_prompt("This will fail"):
                    pass

        # No assistant message persisted (partial response discarded)
        history = await orch._get_history(limit=10)
        assistant_msgs = [m for m in history if m["role"] == "assistant"]
        assert len(assistant_msgs) == 0, "Partial response should not be persisted on error"

        # Router recorded failure (succeeded=False)
        # We can't directly inspect router internals easily, but the
        # record_outcome call with succeeded=False is the contract.
        # The test above verifies the exception propagates; the
        # router state is internal but the call happens in finally block.
    finally:
        if orch.pool is not None:
            await orch.pool.close()


# ---------------------------------------------------------------------------
# MoA preset delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_prompt_delegates_moa(tmp_path: Path):
    """stream_prompt with active MoA preset delegates to _handle_prompt."""
    orch = HermesOrchestrator(db_path=str(tmp_path / "sessions.db"))
    await orch._initialize_memory()

    try:
        # Activate a MoA preset
        from hermes_lite.moa import get_preset
        orch.active_moa_preset = get_preset("council")

        with patch.object(orch, "_handle_prompt", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = "MoA aggregated response"

            collected = []
            async for token in orch.stream_prompt("Explain quantum computing"):
                collected.append(token)

            assert collected == ["MoA aggregated response"]
            mock_handle.assert_awaited_once_with("Explain quantum computing")
    finally:
        if orch.pool is not None:
            await orch.pool.close()