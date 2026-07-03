"""Tests for chat_stream — LLM streaming support."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from hermes_lite.llm import ChatRequest, chat_stream


@dataclass
class FakeDelta:
    content: str

@dataclass
class FakeChoice:
    delta: FakeDelta

@dataclass
class FakeChunk:
    choices: list

@dataclass
class FakeStream:
    """Async iterator that yields chunks."""
    chunks: list

    def __aiter__(self):
        self._idx = 0
        return self

    async def __anext__(self):
        if self._idx >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self._idx]
        self._idx += 1
        return chunk


def _make_chunk(text):
    return FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=text))])


class TestChatStream:
    """Tests for the streaming chat function."""

    @pytest.mark.asyncio
    async def test_stream_yields_deltas(self):
        """Verify chat_stream yields text deltas in order."""
        chunks = [_make_chunk("Hello"), _make_chunk(" world"), _make_chunk("!")]
        fake_stream = FakeStream(chunks=chunks)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream)

        with patch("hermes_lite.llm._pick_client_and_model",
                   return_value=(lambda: mock_client, "test-model", "local")):
            req = ChatRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
            )
            collected = []
            async for delta in chat_stream(req):
                collected.append(delta)

        assert "".join(collected) == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_empty_response(self):
        """Handle empty streaming response gracefully."""
        fake_stream = FakeStream(chunks=[])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream)

        with patch("hermes_lite.llm._pick_client_and_model",
                   return_value=(lambda: mock_client, "test-model", "local")):
            req = ChatRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
            )
            collected = []
            async for delta in chat_stream(req):
                collected.append(delta)

        assert collected == []

    @pytest.mark.asyncio
    async def test_stream_skips_empty_deltas(self):
        """Chunks with no content should be skipped, not yield empty strings."""
        chunks = [
            _make_chunk("Hello"),
            FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=""))]),
            _make_chunk(" world"),
        ]
        fake_stream = FakeStream(chunks=chunks)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream)

        with patch("hermes_lite.llm._pick_client_and_model",
                   return_value=(lambda: mock_client, "test-model", "local")):
            req = ChatRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
            )
            collected = []
            async for delta in chat_stream(req):
                collected.append(delta)

        assert collected == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_passes_stream_flag(self):
        """Verify stream=True is passed to the API."""
        fake_stream = FakeStream(chunks=[_make_chunk("ok")])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream)

        with patch("hermes_lite.llm._pick_client_and_model",
                   return_value=(lambda: mock_client, "test-model", "local")):
            req = ChatRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
            )
            async for _ in chat_stream(req):
                pass

        # Verify stream=True was in the call kwargs
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("stream") is True
