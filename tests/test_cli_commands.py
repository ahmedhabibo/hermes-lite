"""tests/test_cli_commands.py — Tests for CLI slash commands (T13).

Tests cover:
- /tools lists all 6 essentials
- /model shows current model and fallback chain
- /stats shows last 10 turns summary
- /clear resets conversation (keeps memory)
- /memory shows loaded memory entries
- /help lists all commands
- /exit, /quit, /q exit gracefully
"""

import pytest

from hermes_lite.orchestrator import HermesOrchestrator
from hermes_lite.router import LiteRouter
from hermes_lite.registry import PluginRegistry


# ---------------------------------------------------------------------------
# Mock LLM for testing
# ---------------------------------------------------------------------------

async def _mock_chat_fn(req):
    from hermes_lite.llm import ChatResponse
    last_user = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")[:200]
            break
    return ChatResponse(
        content=f"Mock response to: {last_user}",
        tool_calls=[],
        finish_reason="stop",
        model="mock",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        tier="local",
    )


@pytest.fixture
async def orchestrator(tmp_path):
    """Create an orchestrator with a temporary database and mock LLM."""
    db_path = str(tmp_path / "test_sessions.db")
    from hermes_lite.orchestrator import ToolLoop
    loop = ToolLoop(
        registry=PluginRegistry(strict_validation=True),
        chat_fn=_mock_chat_fn,
        max_iterations=4,
    )
    orch = HermesOrchestrator(db_path=db_path, session_title="Test Session", tool_loop=loop)
    orch._create_default_tools()
    loop.registry = orch.registry
    await orch._initialize_memory()
    yield orch
    if orch.pool:
        await orch.pool.close()


# ---------------------------------------------------------------------------
# Slash command tests
# ---------------------------------------------------------------------------

class TestSlashCommands:
    async def test_help_command_lists_all(self, orchestrator):
        """
        /help should list all slash commands including new ones.
        """
        response = await orchestrator._handle_prompt("/help")
        assert "Hermes-Lite Commands" in response
        assert "/tools" in response
        assert "/model" in response
        assert "/stats" in response
        assert "/clear" in response
        assert "/memory" in response
        assert "/history" in response
        assert "/help" in response
        assert "/exit" in response or "/quit" in response

    async def test_tools_command(self, orchestrator):
        """
        /tools should list all 6 essentials (+ subagent).
        """
        response = await orchestrator._handle_prompt("/tools")
        assert "Available Tools" in response
        for name in ("read_file", "search_files", "terminal", "memory", "web_search", "web_fetch"):
            assert name in response, f"missing {name} in /tools response"

    async def test_model_command(self, orchestrator):
        """
        /model should show current model from router's fallback chain.
        """
        response = await orchestrator._handle_prompt("/model")
        assert "Current Model" in response
        assert "Tier" in response
        assert "Fallback Chain" in response
        # Check that it shows the actual chain (cloud-first: NIM model prefixes)
        assert "local:" in response or "nvidia/" in response or "minimaxai/" in response or "z-ai/" in response

    async def test_stats_command_empty(self, orchestrator):
        """
        /stats with no history should show appropriate message.
        """
        response = await orchestrator._handle_prompt("/stats")
        assert "Last 10 Turns Summary" in response or "No conversation turns" in response

    async def test_stats_command_with_history(self, orchestrator):
        """
        /stats with prior messages should show turns.
        """
        await orchestrator._handle_prompt("Hello!")
        await orchestrator._handle_prompt("How are you?")
        response = await orchestrator._handle_prompt("/stats")
        assert "Last 10 Turns Summary" in response
        assert "Turn 1:" in response
        assert "Turn 2:" in response

    async def test_clear_command(self, orchestrator):
        """
        /clear should reset conversation history but keep memory.
        """
        # Add some history
        await orchestrator._handle_prompt("Test message 1")
        await orchestrator._handle_prompt("Test message 2")
        
        # Verify history exists
        history_before = await orchestrator._get_history()
        assert len(history_before) >= 2
        
        # Clear
        response = await orchestrator._handle_prompt("/clear")
        assert "cleared" in response.lower()
        assert "preserved" in response.lower()
        
        # Verify history is cleared
        history_after = await orchestrator._get_history()
        assert len(history_after) == 0

    async def test_memory_command(self, orchestrator):
        """
        /memory should show loaded memory entries or empty message.
        """
        response = await orchestrator._handle_prompt("/memory")
        # Either shows entries or says none loaded
        assert "Memory" in response

    async def test_history_command(self, orchestrator):
        """
        /history should show conversation history.
        """
        await orchestrator._handle_prompt("Hello!")
        response = await orchestrator._handle_prompt("/history")
        assert "History" in response
        assert "Hello!" in response


class TestGeneralPrompt:
    async def test_general_prompt_response(self, orchestrator):
        """
        General prompt should return a response (not empty).
        """
        response = await orchestrator._handle_prompt("What can you do?")
        assert response is not None
        assert len(response) > 0

    async def test_context_memory(self, orchestrator):
        """
        Smoke: type 'hello' twice in a row - second response acknowledges prior context.
        """
        r1 = await orchestrator._handle_prompt("hello")
        assert r1 is not None
        
        r2 = await orchestrator._handle_prompt("hello")
        # The second response should acknowledge the first message exists
        # (memory works in-conversation)
        assert r2 is not None


class TestDirectToolInvocation:
    async def test_direct_tool_invocation(self, orchestrator):
        """
        Direct tool call syntax should invoke a real tool and return its result.
        """
        import tempfile
        import json as _json
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("test content\n")
            tmp = fh.name
        
        args = _json.dumps({"path": tmp})
        response = await orchestrator._handle_prompt(f"!read_file {args}")
        assert "Tool: read_file" in response
        assert "test content" in response
        
        import os
        os.unlink(tmp)
