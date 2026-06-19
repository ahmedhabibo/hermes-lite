"""
tests/test_orchestrator.py — E2E tests for the Hermes-Lite orchestrator.

Tests cover:
- HermesOrchestrator initialization and tool registration
- PluginRegistry integration (tool dispatch via orchestrator)
- SQLite memory integration (history persistence, session management)
- Full integration: prompt → save → retrieve flow
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from hermes_lite.orchestrator import HermesOrchestrator, ToolLoop
from hermes_lite.registry import PluginRegistry, ToolDefinition
from hermes_lite.llm import ChatResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Mock LLM — always returns a plain-text response (no tool_calls).
# This keeps existing orchestrator tests isolated from any real LLM endpoint.
# ---------------------------------------------------------------------------


async def _mock_chat_fn(req) -> ChatResponse:
    """Stub chat function that mimics a simple text-only LLM response.

    Tests that need tool-calling behaviour should use
    tests/test_tool_loop.py instead.
    """
    # Peek at the last user message for a slightly contextual response.
    last_user = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")[:200]
            break
    return ChatResponse(
        content=f"I can help with file searches, web searches, and fetching content from URLs. "
                f"I can also add or modify entries in a persistent memory for future reference. "
                f"What specific task would you like to accomplish? "
                f"(echo of prompt: {last_user})",
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
    loop = ToolLoop(
        registry=PluginRegistry(strict_validation=True),
        chat_fn=_mock_chat_fn,
        max_iterations=4,
    )
    orch = HermesOrchestrator(db_path=db_path, session_title="Test Session", tool_loop=loop)
    orch._create_default_tools()
    # The ToolLoop was created with a bare registry — rewire it to
    # the orchestrator's populated one now that tools are registered.
    loop.registry = orch.registry
    await orch._initialize_memory()
    yield orch
    if orch.pool:
        await orch.pool.close()


@pytest.fixture
def fresh_orch():
    """Return a bare orchestrator without memory initialization (for init tests)."""
    return HermesOrchestrator(db_path=":memory:", session_title="Init Test")


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_creates_default_tools(self, fresh_orch):
        """Default tools should be the 6 essentials + subagent — no echo, calculator, save_note."""
        fresh_orch._create_default_tools()
        assert fresh_orch.registry.tool_count == 7
        names = {t.name for t in fresh_orch.registry.list_tools()}
        assert names == {"read_file", "search_files", "terminal", "memory", "web_search", "web_fetch", "subagent"}
        # Legacy tools must be gone.
        assert "echo" not in names
        assert "calculator" not in names
        assert "save_note" not in names

    def test_get_tool_descriptions(self, fresh_orch):
        """Tool descriptions should be properly formatted for all 7 default tools."""
        fresh_orch._create_default_tools()
        descs = fresh_orch.get_tool_descriptions()
        assert len(descs) == 7
        names = {d["name"] for d in descs}
        assert names == {"read_file", "search_files", "terminal", "memory", "web_search", "web_fetch", "subagent"}

        read_desc = next(d for d in descs if d["name"] == "read_file")
        assert "description" in read_desc
        assert "parameters" in read_desc
        # Schema must expose the path arg as required.
        required = read_desc["parameters"].get("required", [])
        assert "path" in required

    def test_db_directory_created(self, tmp_path):
        """DB parent directory should be created automatically."""
        db_path = str(tmp_path / "nested" / "dir" / "sessions.db")
        orch = HermesOrchestrator(db_path=db_path)
        assert Path(db_path).parent.exists()


# ---------------------------------------------------------------------------
# Memory integration tests
# ---------------------------------------------------------------------------


class TestMemoryIntegration:
    async def test_memory_initialization(self, orchestrator):
        """Orchestrator should create a valid session in the DB."""
        assert orchestrator.session_id != ""
        assert orchestrator.pool is not None
        assert orchestrator.pool.size >= 1

    async def test_save_and_retrieve_user_message(self, orchestrator):
        """Messages should be persisted and retrievable."""
        await orchestrator._save_message("user", "Hello, world!")
        await orchestrator._save_message("assistant", "Hi there!")

        history = await orchestrator._get_history(limit=10)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello, world!"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there!"

    async def test_message_order_preserved(self, orchestrator):
        """Messages should maintain insertion order."""
        for i in range(5):
            await orchestrator._save_message("user", f"msg-{i}")

        history = await orchestrator._get_history(limit=10)
        contents = [m["content"] for m in history]
        assert contents == [f"msg-{i}" for i in range(5)]

    async def test_history_limit(self, orchestrator):
        """History retrieval should respect the limit parameter."""
        for i in range(10):
            await orchestrator._save_message("user", f"msg-{i}")

        history = await orchestrator._get_history(limit=3)
        assert len(history) == 3

    async def test_message_metadata(self, orchestrator):
        """Message metadata should be persisted."""
        await orchestrator._save_message(
            "user", "test", metadata={"source": "cli", "tokens": 10}
        )
        history = await orchestrator._get_history()
        assert history[0]["metadata"]["source"] == "cli"
        assert history[0]["metadata"]["tokens"] == 10


# ---------------------------------------------------------------------------
# Tool integration tests
# ---------------------------------------------------------------------------


class TestToolIntegration:
    async def test_read_file_tool_registered(self, orchestrator):
        """read_file should be registered and callable on a small temp file."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("hello\nworld\n")
            tmp = fh.name
        assert orchestrator.registry.has_tool("read_file")
        out = orchestrator.registry.call_tool("read_file", {"path": tmp})
        assert out["ok"] is True
        assert "hello" in out["output"]
        assert "world" in out["output"]
        Path(tmp).unlink(missing_ok=True)

    async def test_terminal_tool_runs_in_sandbox(self, orchestrator):
        """terminal tool should execute a trivial command via the sandbox."""
        assert orchestrator.registry.has_tool("terminal")
        out = orchestrator.registry.call_tool("terminal", {"cmd": "/bin/echo hello"})
        assert out["ok"] is True
        payload = json.loads(out["output"])
        assert payload["exit_code"] == 0
        assert "hello" in payload["stdout"]

    async def test_tool_not_found(self, orchestrator):
        """Unknown tool should raise ToolNotFoundError."""
        from hermes_lite.registry import ToolNotFoundError
        with pytest.raises(ToolNotFoundError):
            orchestrator.registry.call_tool("nonexistent", {})

    async def test_tool_validation_error(self, orchestrator):
        """Invalid tool arguments should raise ToolValidationError."""
        from hermes_lite.registry import ToolValidationError
        # read_file requires "path" — missing should fail.
        with pytest.raises(ToolValidationError):
            orchestrator.registry.call_tool("read_file", {})

    async def test_extra_args_rejected(self, orchestrator):
        """Schema rejects unknown keys (extra='forbid')."""
        from hermes_lite.registry import ToolValidationError
        with pytest.raises(ToolValidationError):
            orchestrator.registry.call_tool("read_file", {"path": "/tmp/x", "weird": 1})


# ---------------------------------------------------------------------------
# Prompt handling integration tests
# ---------------------------------------------------------------------------


class TestPromptHandling:
    async def test_prompt_saves_user_message(self, orchestrator):
        """User prompt should be saved to memory."""
        response = await orchestrator._handle_prompt("Hello!")
        # Check message was saved
        history = await orchestrator._get_history(limit=10)
        user_msgs = [m for m in history if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert user_msgs[-1]["content"] == "Hello!"

    async def test_prompt_saves_response(self, orchestrator):
        """Assistant response should be saved to memory."""
        response = await orchestrator._handle_prompt("Hello!")
        history = await orchestrator._get_history(limit=10)
        assistant_msgs = [m for m in history if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[-1]["content"] == response

    async def test_tools_command_lists_tools(self, orchestrator):
        """/tools command should list the 6 registered essentials."""
        response = await orchestrator._handle_prompt("/tools")
        for name in ("read_file", "search_files", "terminal", "memory", "web_search", "web_fetch"):
            assert name in response, f"missing {name} in /tools response"
        # Legacy names must not appear.
        assert "echo:" not in response
        assert "calculator:" not in response

    async def test_help_command(self, orchestrator):
        """/help command should show help text."""
        response = await orchestrator._handle_prompt("/help")
        assert "Hermes-Lite" in response or "Commands" in response
        assert "!/" in response or "!tool" in response

    async def test_history_command(self, orchestrator):
        """Empty history should show appropriate message."""
        response = await orchestrator._handle_prompt("/history")
        assert "No conversation history" in response or "history" in response.lower()

    async def test_history_with_messages(self, orchestrator):
        """/history with prior messages should show them."""
        await orchestrator._save_message("user", "first message")
        response = await orchestrator._handle_prompt("/history")
        assert "first message" in response or "first" in response

    async def test_general_prompt_response(self, orchestrator):
        """General prompt should return a response (not empty)."""
        response = await orchestrator._handle_prompt("What can you do?")
        assert response is not None
        assert len(response) > 0

    async def test_direct_tool_invocation(self, orchestrator):
        """Direct tool call syntax should invoke a real tool and return its result."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("direct-call-content\n")
            tmp = fh.name
        import json as _json
        args = _json.dumps({"path": tmp})
        response = await orchestrator._handle_prompt(f"!read_file {args}")
        assert "Tool: read_file" in response
        assert "direct-call-content" in response
        Path(tmp).unlink(missing_ok=True)

    async def test_direct_tool_invocation_no_args(self, orchestrator):
        """Direct tool call without required args yields a tool error."""
        response = await orchestrator._handle_prompt("!read_file")
        assert "Tool error" in response or "error" in response.lower()

    async def test_direct_tool_not_found(self, orchestrator):
        """Direct tool call for unknown tool should show error."""
        response = await orchestrator._handle_prompt('!unknown_tool {"x": 1}')
        assert "Tool error" in response or "error" in response.lower() or "not found" in response.lower()


# ---------------------------------------------------------------------------
# Full integration flow
# ---------------------------------------------------------------------------


class TestFullIntegration:
    async def test_conversation_flow(self, orchestrator):
        """Simulate a full conversation with tool use."""
        # First message
        r1 = await orchestrator._handle_prompt("Hello!")
        assert r1 is not None
        assert len(r1) > 0

        # Check history
        h1 = await orchestrator._get_history()
        assert len(h1) == 2  # user + assistant

        # Second message
        r2 = await orchestrator._handle_prompt("What tools do you have?")
        # After T5 the general-response path lists the 6 essential tools,
        # not the retired echo/calculator defaults.
        assert "read_file" in r2 or "terminal" in r2 or "tool" in r2.lower()
        assert any(
            name in r2
            for name in ("read_file", "search_files", "terminal", "memory", "web_search", "web_fetch")
        )

        # Third message — use /tools command
        r3 = await orchestrator._handle_prompt("/tools")
        assert len(r3) > 0

        # Check all messages persisted
        h3 = await orchestrator._get_history()
        assert len(h3) == 6  # 3 exchanges = 6 messages

    async def test_new_session_privacy(self, tmp_path):
        """Different orchestrators with different DBs should have isolated sessions.市
        """
        loop = ToolLoop(
            registry=PluginRegistry(strict_validation=True),
            chat_fn=_mock_chat_fn,
            max_iterations=2,
        )
        orch1 = HermesOrchestrator(
            db_path=str(tmp_path / "db1.db"), tool_loop=loop
        )
        orch1._create_default_tools()
        loop.registry = orch1.registry
        await orch1._initialize_memory()
        await orch1._handle_prompt("Hello from orch1!")

        loop2 = ToolLoop(
            registry=PluginRegistry(strict_validation=True),
            chat_fn=_mock_chat_fn,
            max_iterations=2,
        )
        orch2 = HermesOrchestrator(
            db_path=str(tmp_path / "db2.db"), tool_loop=loop2
        )
        orch2._create_default_tools()
        loop2.registry = orch2.registry
        await orch2._initialize_memory()
        await orch2._handle_prompt("Hello from orch2!")

        # Each should have their own history
        h1 = await orch1._get_history()
        h2 = await orch2._get_history()

        # user + assistant (count tool calls via role filter)
        assert len(h1) >= 2
        assert len(h2) >= 2
        # First message in each is from its own user
        first_user = next(m for m in h1 if m["role"] == "user")
        assert first_user["content"] == "Hello from orch1!"
        first_user2 = next(m for m in h2 if m["role"] == "user")
        assert first_user2["content"] == "Hello from orch2!"

        await orch1.pool.close()
        await orch2.pool.close()

    async def test_custom_tool_registration(self, orchestrator):
        """Custom tools can be added to the orchestrator's registry."""
        class GreetArgs(BaseModel):
            name: str = Field(..., description="Name to greet")

        def greet_handler(args: GreetArgs) -> str:
            return f"Hello, {args.name}!"

        tool_def = ToolDefinition(
            name="greet",
            description="Greet someone by name.",
            schema_model=GreetArgs,
            handler=greet_handler,
        )
        orchestrator.registry.add_tool(tool_def)
        # 7 defaults (6 essentials + subagent) + 1 custom = 8
        assert orchestrator.registry.tool_count == 8

        # Test invocation
        result = orchestrator.registry.call_tool("greet", {"name": "World"})
        assert result == "Hello, World!"

        # Test via CLI prompt
        response = await orchestrator._handle_prompt('!greet {"name": "Orchestrator"}')
        assert "Hello, Orchestrator" in response


# ---------------------------------------------------------------------------
# Routing controller integration
# ---------------------------------------------------------------------------


class TestRoutingIntegration:
    """End-to-end checks that the orchestrator actually consults the
    LiteRouter before responding — covers the spec's step 5
    (Wire into orchestrator: handle_prompt calls router.route() before
    llm.chat()).
    """

    async def _last_routing(self, orchestrator) -> dict:
        history = await orchestrator._get_history(limit=5)
        # Newest assistant message carries the routing metadata blob.
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                meta = msg.get("metadata") or {}
                routing = meta.get("routing")
                if routing:
                    return routing
        return {}

    async def test_simple_prompt_records_local_tier(self, orchestrator):
        """A trivial 'find X' prompt must record tier=local."""
        await orchestrator._handle_prompt("find customer_id 42")
        routing = await self._last_routing(orchestrator)
        assert routing.get("tier") == "local"
        assert routing.get("model_id", "").startswith("local:")

    async def test_refactor_prompt_records_cloud_tier(self, orchestrator):
        """Acceptance: 'refactor this 200-line script' must record cloud."""
        await orchestrator._handle_prompt("refactor this 200-line script")
        routing = await self._last_routing(orchestrator)
        assert routing.get("tier") == "cloud"
        # Cloud model id should NOT start with local:
        assert not routing.get("model_id", "").startswith("local:")

    async def test_transparency_emoji_in_response(self, orchestrator):
        """The general-response path surfaces the chosen tier to users."""
        response = await orchestrator._handle_prompt("find invoice 7")
        # ⚡ for local, ☁️ for cloud — at least one must appear.
        assert ("⚡" in response) or ("☁️" in response)
        assert "local" in response or "cloud" in response

    async def test_success_resets_escalation_counter(self, orchestrator):
        """Each successful prompt trips record_outcome(); counter must
        stay at zero on a fresh default router."""
        # Force two failures on the controller
        decision = orchestrator.router.route("hi", 0, 0)
        orchestrator.router.record_outcome(decision, succeeded=False)
        orchestrator.router.record_outcome(decision, succeeded=False)

        # Then a successful prompt must reset the counter.
        await orchestrator._handle_prompt("find invoice 5")
        assert orchestrator.router.consecutive_local_failures == 0

    async def test_router_override(self, tmp_path):
        """A custom router passed at construction is honoured."""
        from hermes_lite.router import LiteRouter

        custom = LiteRouter(
            fallback_chain=["local:custom-only"],
            local_max_complexity=0.05,  # aggressively cloud
        )
        orch = HermesOrchestrator(
            db_path=str(tmp_path / "routing.db"),
            router=custom,
        )
        orch._create_default_tools()
        await orch._initialize_memory()
        try:
            response = await orch._handle_prompt("hello")
            # Even a trivial prompt should escalate under the aggressive
            # threshold (0.05) — there's tiny prompt contribution.
            history = await orch._get_history(limit=5)
            routing_meta = next(
                m.get("metadata", {}).get("routing", {})
                for m in reversed(history)
                if m.get("role") == "assistant" and m.get("metadata", {}).get("routing")
            )
            assert routing_meta["tier"] == "cloud"
            assert response  # smoke check — non-empty
        finally:
            if orch.pool:
                await orch.pool.close()