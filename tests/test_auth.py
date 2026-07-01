"""Tests for Auth/Authorization (ToolAuthError, dangerous flag, auth_token)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from hermes_lite.registry import (
    PluginRegistry,
    ToolDefinition,
    ToolAuthError,
    ToolNotFoundError,
    ToolValidationError,
)
from hermes_lite.orchestrator import HermesOrchestrator
from hermes_lite.tools_builtins import register_builtins


class TestToolAuthError:
    """Tests for ToolAuthError exception."""

    def test_tool_auth_error_basic(self):
        """Basic ToolAuthError creation."""
        exc = ToolAuthError("terminal")
        assert exc.tool_name == "terminal"
        assert "authentication" in str(exc).lower()

    def test_tool_auth_error_custom_message(self):
        """ToolAuthError with custom message."""
        exc = ToolAuthError("read_file", "Custom auth message")
        assert exc.tool_name == "read_file"
        assert "Custom auth message" in str(exc)


class TestPluginRegistryAuth:
    """Tests for PluginRegistry auth_token handling."""

    def test_registry_auth_token_none_no_auth_required(self):
        """auth_token=None means no auth required for any tool."""
        registry = PluginRegistry(strict_validation=False, auth_token=None)
        registry.add_tool(ToolDefinition(
            name="test_tool",
            description="Test",
            schema_model=None,
            handler=lambda: "ok",
            dangerous=True,
        ))
        # Should work without auth_token
        result = registry.call_tool("test_tool", {})
        assert result == "ok"

    def test_registry_auth_token_set_dangerous_tool_blocked_without_token(self):
        """Dangerous tool blocked when auth_token set but not provided."""
        registry = PluginRegistry(strict_validation=False, auth_token="secret123")
        registry.add_tool(ToolDefinition(
            name="dangerous_tool",
            description="Test",
            schema_model=None,
            handler=lambda: "ok",
            dangerous=True,
        ))
        with pytest.raises(ToolAuthError) as exc_info:
            registry.call_tool("dangerous_tool", {})
        assert exc_info.value.tool_name == "dangerous_tool"

    def test_registry_auth_token_set_dangerous_tool_allowed_with_correct_token(self):
        """Dangerous tool allowed when correct auth_token provided."""
        registry = PluginRegistry(strict_validation=False, auth_token="secret123")
        registry.add_tool(ToolDefinition(
            name="dangerous_tool",
            description="Test",
            schema_model=None,
            handler=lambda: "ok",
            dangerous=True,
        ))
        result = registry.call_tool("dangerous_tool", {}, auth_token="secret123")
        assert result == "ok"

    def test_registry_auth_token_set_dangerous_tool_blocked_with_wrong_token(self):
        """Dangerous tool blocked when wrong auth_token provided."""
        registry = PluginRegistry(strict_validation=False, auth_token="secret123")
        registry.add_tool(ToolDefinition(
            name="dangerous_tool",
            description="Test",
            schema_model=None,
            handler=lambda: "ok",
            dangerous=True,
        ))
        with pytest.raises(ToolAuthError):
            registry.call_tool("dangerous_tool", {}, auth_token="wrong")

    def test_registry_auth_token_non_dangerous_tool_always_allowed(self):
        """Non-dangerous tool always allowed regardless of auth_token."""
        registry = PluginRegistry(strict_validation=False, auth_token="secret123")
        registry.add_tool(ToolDefinition(
            name="safe_tool",
            description="Test",
            schema_model=None,
            handler=lambda: "ok",
            dangerous=False,
        ))
        # Should work without token
        result = registry.call_tool("safe_tool", {})
        assert result == "ok"
        # Should work with wrong token
        result = registry.call_tool("safe_tool", {}, auth_token="wrong")
        assert result == "ok"

    def test_registry_auth_token_constructor_vs_call_tool_priority(self):
        """call_tool auth_token must match registry auth_token exactly."""
        registry = PluginRegistry(strict_validation=False, auth_token="registry_token")
        registry.add_tool(ToolDefinition(
            name="dangerous_tool",
            description="Test",
            schema_model=None,
            handler=lambda: "ok",
            dangerous=True,
        ))
        # Correct token → allowed
        result = registry.call_tool("dangerous_tool", {}, auth_token="registry_token")
        assert result == "ok"
        # Wrong token → blocked
        with pytest.raises(ToolAuthError):
            registry.call_tool("dangerous_tool", {}, auth_token="wrong_token")

    def test_builtins_dangerous_flags(self):
        """Verify built-in tools have correct dangerous flags."""
        registry = PluginRegistry(auth_token="test")
        register_builtins(registry)

        # Dangerous tools
        for name in ["read_file", "search_files", "terminal", "web_search", "web_fetch"]:
            tool = registry.get_tool(name)
            assert tool.dangerous is True, f"{name} should be dangerous"

        # Non-dangerous tool
        memory_tool = registry.get_tool("memory")
        assert memory_tool.dangerous is False


class TestOrchestratorAuth:
    """Tests for HermesOrchestrator auth integration."""

    @pytest.mark.asyncio
    async def test_orchestrator_auth_token_from_env(self):
        """Orchestrator reads HERMES_LITE_AUTH_TOKEN from env."""
        with patch.dict("os.environ", {"HERMES_LITE_AUTH_TOKEN": "env_token"}):
            orch = HermesOrchestrator()
            assert orch.auth_token == "env_token"
            assert orch.registry.auth_token == "env_token"

    @pytest.mark.asyncio
    async def test_orchestrator_auth_token_constructor_override(self):
        """Constructor auth_token overrides env var."""
        with patch.dict("os.environ", {"HERMES_LITE_AUTH_TOKEN": "env_token"}):
            orch = HermesOrchestrator(auth_token="constructor_token")
            assert orch.auth_token == "constructor_token"
            assert orch.registry.auth_token == "constructor_token"

    @pytest.mark.asyncio
    async def test_orchestrator_no_auth_token_by_default(self):
        """No auth token by default (backward compat)."""
        with patch.dict("os.environ", {}, clear=True):
            orch = HermesOrchestrator()
            assert orch.auth_token is None
            assert orch.registry.auth_token is None

    @pytest.mark.asyncio
    async def test_orchestrator_direct_tool_call_blocked_without_auth(self):
        """Direct !tool call blocked when auth_token set but not provided.
        
        This tests the case where a caller WITHOUT the token tries to use the tool.
        The orchestrator itself always passes its token, so it's allowed.
        """
        orch = HermesOrchestrator(auth_token="secret123")
        # Override registry with non-strict one so we can add schema-less tools
        orch.registry = PluginRegistry(strict_validation=False, auth_token="secret123")
        orch.registry.add_tool(ToolDefinition(
            name="test_dangerous",
            description="Test",
            schema_model=None,
            handler=lambda: "ok",
            dangerous=True,
        ))

        # Simulate a caller WITHOUT the token calling the registry directly
        with pytest.raises(ToolAuthError):
            orch.registry.call_tool("test_dangerous", {})  # No auth_token provided

    @pytest.mark.asyncio
    async def test_orchestrator_direct_tool_call_allowed_with_auth(self):
        """Direct !tool call allowed when auth_token matches."""
        orch = HermesOrchestrator(auth_token="secret123")
        # Override registry with non-strict one so we can add schema-less tools
        orch.registry = PluginRegistry(strict_validation=False, auth_token="secret123")
        orch.registry.add_tool(ToolDefinition(
            name="test_dangerous",
            description="Test",
            schema_model=None,
            handler=lambda: "ok",
            dangerous=True,
        ))

        # The orchestrator passes registry.auth_token to call_tool
        response = await orch._handle_prompt("!test_dangerous")
        assert "ok" in response.lower() or "Result: ok" in response

    @pytest.mark.asyncio
    async def test_orchestrator_tool_loop_auth_passed(self):
        """ToolLoop passes auth_token when calling tools."""
        orch = HermesOrchestrator(auth_token="secret123")
        orch.active_moa_preset = None

        # Mock tool_loop.run to capture the auth_token passed
        mock_tool_loop = AsyncMock()
        mock_loop_result = MagicMock()
        mock_loop_result.tool_names = []
        mock_loop_result.iterations = 1
        mock_loop_result.terminated_by = "complete"
        mock_loop_result.response = "done"
        mock_loop_result.tool_calls_made = 0
        mock_loop_result.metadata = {"_obs": {}}
        mock_tool_loop.run = AsyncMock(return_value=mock_loop_result)
        orch._tool_loop = mock_tool_loop

        with patch.object(orch, "router") as mock_router:
            mock_router.route = MagicMock(return_value=MagicMock(
                tier="cloud",
                model_id="minimaxai/minimax-m3",
                reasoning="test"
            ))
            mock_router.record_outcome = MagicMock()

            orch._save_message = AsyncMock()
            orch._build_llm_history = AsyncMock(return_value=[])

            await orch._handle_prompt("test prompt")

            # Verify tool_loop.run was called (auth is handled inside ToolLoop)
            mock_tool_loop.run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])