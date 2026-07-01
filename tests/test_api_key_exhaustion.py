"""Tests for API key exhaustion handling (AllKeysExhausted)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from hermes_lite.llm import (
    APIKeyRotator,
    AllKeysExhausted,
    _chat_cloud_with_retry,
    chat,
    ChatRequest,
    Tier,
)
from hermes_lite.orchestrator import HermesOrchestrator


class TestAPIKeyRotatorExhaustion:
    """Tests for APIKeyRotator.is_exhausted() method."""

    def test_is_exhausted_no_keys(self):
        """Empty pool returns (True, 0.0)."""
        rotator = APIKeyRotator(keys=[])
        exhausted, cooldown = rotator.is_exhausted()
        assert exhausted is True
        assert cooldown == 0.0

    def test_is_exhausted_all_in_cooldown(self):
        """All keys marked failed → (True, remaining)."""
        rotator = APIKeyRotator(keys=["key1", "key2", "key3"])
        # Mark all keys as failed with future expiry times
        import time
        future = time.monotonic() + 1000.0
        rotator._cooldowns = {0: future, 1: future, 2: future}
        exhausted, cooldown = rotator.is_exhausted()
        assert exhausted is True
        assert cooldown > 0.0

    def test_is_exhausted_some_available(self):
        """One key not in cooldown → (False, 0.0)."""
        rotator = APIKeyRotator(keys=["key1", "key2", "key3"])
        # Only key1 in cooldown with future expiry
        import time
        future = time.monotonic() + 1000.0
        rotator._cooldowns = {0: future}
        exhausted, cooldown = rotator.is_exhausted()
        assert exhausted is False
        assert cooldown == 0.0

    def test_is_exhausted_cooldown_expired(self):
        """Key recovers after 60s → (False, 0.0)."""
        rotator = APIKeyRotator(keys=["key1"])
        # Mark as failed but with expired cooldown
        import time
        rotator._cooldowns = {0: time.monotonic() - 10.0}  # expired 10s ago
        exhausted, cooldown = rotator.is_exhausted()
        assert exhausted is False
        assert cooldown == 0.0


class TestAllKeysExhaustedException:
    """Tests for AllKeysExhausted exception."""

    def test_all_keys_exhausted_exception_attributes(self):
        """Verify keys_tried and cooldown_remaining attributes."""
        exc = AllKeysExhausted(keys_tried=3, cooldown_remaining=45.5)
        assert exc.keys_tried == 3
        assert exc.cooldown_remaining == 45.5
        assert "3 API keys exhausted" in str(exc)
        assert "46" in str(exc)  # 45.5 rounds to 46 with :.0f


class TestChatFallbackOnExhaustion:
    """Tests for chat() falling back to local on AllKeysExhausted."""

    @pytest.mark.asyncio
    async def test_chat_falls_back_to_local_on_exhaustion(self):
        """Mock is_exhausted() → True, verify local call."""
        with patch("hermes_lite.llm._key_rotator.is_exhausted", return_value=(True, 30.0)):
            with patch("hermes_lite.llm.AsyncOpenAI") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message = MagicMock()
                mock_resp.choices[0].message.content = "local response"
                mock_resp.choices[0].message.tool_calls = None
                mock_resp.choices[0].finish_reason = "stop"
                mock_resp.usage = MagicMock()
                mock_resp.usage.prompt_tokens = 10
                mock_resp.usage.completion_tokens = 20
                mock_resp.usage.total_tokens = 30
                mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

                req = ChatRequest(
                    messages=[{"role": "user", "content": "test"}],
                    model="minimaxai/minimax-m3",
                )
                resp = await chat(req)

                assert resp.content == "local response"
                assert resp.tier == "local"
                # Verify local client was called
                mock_client_class.assert_called_with(
                    base_url="http://127.0.0.1:8080/v1",
                    api_key="not-needed",
                    timeout=60.0,
                )


class TestOrchestratorExhaustionHandling:
    """Tests for orchestrator catching AllKeysExhausted."""

    @pytest.mark.asyncio
    async def test_orchestrator_moa_falls_back_on_exhaustion(self):
        """MoA path catches AllKeysExhausted and falls back to ToolLoop."""
        orch = HermesOrchestrator()
        orch.active_moa_preset = "council"

        # Mock the MoA engine to raise AllKeysExhausted
        with patch("hermes_lite.orchestrator.MoAEngine") as mock_moa_class:
            mock_moa = AsyncMock()
            mock_moa_class.return_value = mock_moa
            mock_moa.run = AsyncMock(side_effect=AllKeysExhausted(keys_tried=2, cooldown_remaining=30.0))

            # Mock tool_loop.run to succeed - patch _tool_loop instead of tool_loop property
            mock_tool_loop = AsyncMock()
            mock_loop_result = MagicMock()
            mock_loop_result.tool_names = []
            mock_loop_result.iterations = 1
            mock_loop_result.terminated_by = "complete"
            mock_loop_result.response = "fallback response"
            mock_loop_result.tool_calls_made = 0
            mock_loop_result.metadata = {"_obs": {}}
            mock_tool_loop.run = AsyncMock(return_value=mock_loop_result)
            orch._tool_loop = mock_tool_loop

            # Mock router
            with patch.object(orch, "router") as mock_router:
                mock_router.route = MagicMock(return_value=MagicMock(
                    tier="cloud",
                    model_id="minimaxai/minimax-m3",
                    reasoning="test"
                ))
                mock_router.record_outcome = MagicMock()

                # Mock memory methods
                orch._save_message = AsyncMock()
                orch._build_llm_history = AsyncMock(return_value=[])

                response = await orch._handle_prompt("test prompt")

                assert "fallback response" in response
                mock_tool_loop.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_toolloop_graceful_error_on_exhaustion(self):
        """ToolLoop path catches AllKeysExhausted and returns graceful error message."""
        orch = HermesOrchestrator()
        orch.active_moa_preset = None

        # Mock tool_loop.run to raise AllKeysExhausted - patch _tool_loop
        mock_tool_loop = AsyncMock()
        mock_tool_loop.run = AsyncMock(side_effect=AllKeysExhausted(keys_tried=2, cooldown_remaining=45.0))
        orch._tool_loop = mock_tool_loop

        # Mock router
        with patch.object(orch, "router") as mock_router:
            mock_router.route = MagicMock(return_value=MagicMock(
                tier="cloud",
                model_id="minimaxai/minimax-m3",
                reasoning="test"
            ))
            mock_router.record_outcome = MagicMock()

            # Mock memory methods
            orch._save_message = AsyncMock()
            orch._build_llm_history = AsyncMock(return_value=[])

            response = await orch._handle_prompt("test prompt")

            assert "All API keys exhausted" in response
            assert "45s" in response
            assert "HERMES_LITE_LOCAL_MODEL" in response
            mock_router.record_outcome.assert_not_called()  # early return


if __name__ == "__main__":
    pytest.main([__file__, "-v"])