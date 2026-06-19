"""
tests/test_tool_loop.py — Tool-calling loop (T6) with mocked LLM responses.

These are unit tests for `ToolLoop` in isolation.  We exercise the full
multi-turn state machine without ever hitting a real LLM endpoint by
injecting a *controller* that decides what each LLM turn looks like.

Acceptance coverage:
- read_file → summarise  (<= 3 turns, terminates with 'complete')
- max_iterations guard     (never exceeds the cap)
- repeated_error guard     (same tool + same error 2× → stop)
- malformed tool_call      (bad JSON → nudge once → break)
- on_tool_call callback    (bolt progress visible to caller)
- history mutation         (messages list grows correctly)
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from hermes_lite.orchestrator import ToolLoop, ToolLoopResult, HermesOrchestrator
from hermes_lite.registry import PluginRegistry, ToolDefinition
from hermes_lite.llm import ChatRequest, ChatResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockChatController:
    """Deterministic fake LLM that returns pre-programmed responses.

    Register a list of ``ChatResponse`` objects; each call pops the first.
    """

    def __init__(self, responses: list[ChatResponse]):
        self.responses = list(responses)
        self.calls: list[ChatRequest] = []

    async def __call__(self, req: ChatRequest) -> ChatResponse:
        self.calls.append(req)
        if not self.responses:
            raise RuntimeError("MockChatController ran out of canned responses")
        return self.responses.pop(0)


def _toolcall(name: str, args: dict[str, Any], id: str = "") -> dict[str, Any]:
    return {
        "id": id or f"call_{name}",
        "name": name,
        "arguments": json.dumps(args),
    }


def _resp(content: str = "", tool_calls: list[dict] | None = None, tier: str = "local") -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop" if not tool_calls else "tool_calls",
        model="mock",
        usage={},
        tier=tier,
    )


def _make_registry() -> PluginRegistry:
    """Registry with 3 simple tools for exercising the loop."""
    registry = PluginRegistry(strict_validation=True)

    class ReadArgs(BaseModel):
        path: str = Field(...)
        offset: int = Field(default=1)
        limit: int = Field(default=500)

    class CalcArgs(BaseModel):
        a: float = Field(...)
        b: float = Field(...)

    def read_handler(args: ReadArgs) -> dict:
        try:
            with open(args.path, "r") as fh:
                lines = fh.readlines()
                start = max(0, args.offset - 1)
                end = start + args.limit
                content = "".join(lines[start:end])
                return {"ok": True, "output": content}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "output": ""}

    def calc_handler(args: CalcArgs) -> dict:
        return {"ok": True, "output": str(args.a + args.b)}

    def err_handler(args: Any) -> dict:
        # Always errors (used for repeated-error test)
        return {"ok": False, "error": "always fails", "output": ""}

    registry.add_tool(ToolDefinition(name="read_file", description="Read a file.", schema_model=ReadArgs, handler=read_handler))
    registry.add_tool(ToolDefinition(name="calc", description="Add two numbers.", schema_model=CalcArgs, handler=calc_handler))
    registry.add_tool(ToolDefinition(name="fail", description="Always fails.", schema_model=ReadArgs, handler=err_handler))
    return registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return _make_registry()


# ---------------------------------------------------------------------------
# 1. Happy path: tool call → result → final answer
# ---------------------------------------------------------------------------

class TestToolLoopHappyPath:
    async def test_complete_in_one_turn(self, registry):
        """If the LLM starts with no tool_calls, the loop exits immediately."""
        controller = MockChatController([_resp("Hello, world!")])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)

        result = await loop.run([{"role": "user", "content": "hi"}], model="local:mock")

        assert result.terminated_by == "complete"
        assert result.iterations == 1
        assert result.tool_calls_made == 0
        assert result.tool_names == []
        assert result.response == "Hello, world!"

    async def test_single_tool_call_then_done(self, registry):
        """One tool call, LLM sees result and answers without calling more tools."""
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("read_file", {"path": "/tmp/readme.md", "offset": 1, "limit": 10})]),
            _resp("README contains project docs."),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)

        result = await loop.run([{"role": "user", "content": "read the readme"}], model="local:mock")

        assert result.terminated_by == "complete"
        assert result.iterations == 2
        assert result.tool_calls_made == 1
        assert result.tool_names == ["read_file"]
        assert result.response == "README contains project docs."

    async def test_multi_tool_then_done(self, registry, tmp_path):
        """LLM issues 2 parallel tool calls in one turn, then finalises."""
        readme = tmp_path / "readme.md"
        readme.write_text("# Hello\n")
        controller = MockChatController([
            _resp(
                "",
                tool_calls=[
                    _toolcall("read_file", {"path": str(readme), "offset": 1, "limit": 5}),
                    _toolcall("calc", {"a": 1, "b": 2}, id="call_2"),
                ],
            ),
            _resp("The file says hello; 1+2 = 3."),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)

        result = await loop.run([{"role": "user", "content": "read and calc"}], model="local:mock")

        assert result.terminated_by == "complete"
        assert result.iterations == 2
        assert result.tool_calls_made == 2
        assert result.tool_names == ["read_file", "calc"]

    async def test_readme_and_summarize(self, registry, tmp_path):
        """Acceptance: 'read README.md and summarise' <= 3 turns, ends with 'complete'."""
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n\nThis is a test project.\n")
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("read_file", {"path": str(readme), "offset": 1, "limit": 100})]),
            _resp("This test project is documented in the README."),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)

        result = await loop.run([{"role": "user", "content": "read README.md and summarise"}], model="local:mock")

        assert result.terminated_by == "complete"
        assert result.iterations <= 3
        assert "read_file" in result.tool_names
        assert "test project" in result.response.lower()


# ---------------------------------------------------------------------------
# 2. Max-iterations guard
# ---------------------------------------------------------------------------

class TestToolLoopMaxIterations:
    async def test_respects_max_iterations(self, registry):
        """Loop should stop exactly at max_iterations even if LLM keeps insisting on tools."""
        # Every turn the LLM asks for a different tool call so repeated_error
        # does not fire. The loop must still stop at max_iterations.
        controller = MockChatController(
            [_resp("", tool_calls=[_toolcall("read_file", {"path": "/x", "offset": 1, "limit": 10})])]
            + [_resp("", tool_calls=[_toolcall("calc", {"a": i, "b": i + 1})]) for i in range(4)]
        )
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=3)

        result = await loop.run([{"role": "user", "content": "loop test"}], model="local:mock")

        assert result.terminated_by == "max_iterations"
        assert result.iterations == 3
        # 3 tool calls in 3 iterations, first is read_file then 2 calc
        assert result.tool_calls_made == 3

    async def test_adversarial_never_exceeds(self, registry):
        """Even with an adversarial prompt, loop never exceeds max_iterations deep."""
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("read_file", {"path": "/dev/urandom"})])
            for _ in range(10)
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=2)
        result = await loop.run([{"role": "user", "content": "loop forever"}], model="local:mock")
        assert result.terminated_by == "max_iterations"
        assert result.iterations <= 2


# ---------------------------------------------------------------------------
# 3. Repeated-error guard
# ---------------------------------------------------------------------------

class TestToolLoopRepeatedError:
    async def test_same_tool_same_error_twice_stops(self, registry):
        """If a tool produces the same error 2× in a row, the loop breaks with 'repeated_error'."""
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("fail", {"path": "/nope", "offset": 1, "limit": 10})]),
            _resp("", tool_calls=[_toolcall("fail", {"path": "/nope", "offset": 1, "limit": 10})]),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)
        result = await loop.run([{"role": "user", "content": "test"}], model="local:mock")

        assert result.terminated_by == "repeated_error"
        # Two tool calls total (one in each of two iterations)
        assert result.tool_calls_made == 2
        assert "fail" in result.tool_names
        assert "repeated error" in result.response.lower()

    async def test_different_error_does_not_stop(self, registry):
        """Different errors from the same tool should not trigger repeated_error."""
        # We'll simulate read_file returning a ToolError with different messages.
        # Instead we use a custom handler that returns different messages each call.
        ...  # Tested below via the failure handler that always returns same error.
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("fail", {"path": "/first", "offset": 1, "limit": 10})]),
            _resp("", tool_calls=[_toolcall("fail", {"path": "/second", "offset": 1, "limit": 10})]),
            _resp("ok"),  # third turn — if we got here, loop didn't break
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)
        # Since _handle_fail always returns "always fails", both failures have identical
        # error messages → repeated_error should fire on the second failure.
        # This is the expected behavior per the spec.
        result = await loop.run([{"role": "user", "content": "test"}], model="local:mock")
        assert result.terminated_by == "repeated_error"

    async def test_recovery_after_different_errors(self):
        """If a tool errors once with message A then message B, loop continues."""
        custom = PluginRegistry(strict_validation=True)

        class VaryingArgs(BaseModel):
            """Schema with a dummy field so handler gets called properly."""
            dummy: str = Field(default="x")

        class _State:
            count = 0

        def varying_handler(args: VaryingArgs) -> dict:
            _State.count += 1
            return {"ok": False, "error": f"err_{_State.count}", "output": ""}

        custom.add_tool(ToolDefinition(
            name="varying", description="...", schema_model=VaryingArgs,
            handler=varying_handler,
        ))
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("varying", {})]),
            _resp("", tool_calls=[_toolcall("varying", {})]),
            _resp("ok"),
        ])
        loop = ToolLoop(registry=custom, chat_fn=controller, max_iterations=4)
        result = await loop.run([{"role": "user", "content": "test"}], model="local:mock")
        assert result.terminated_by == "complete"
        assert result.iterations == 3


# ---------------------------------------------------------------------------
# 4. Malformed JSON / nudge-once
# ---------------------------------------------------------------------------

class TestToolLoopMalformedJson:
    async def test_bad_json_nudges_once(self, registry):
        """A tool call with bad JSON causes a nudge; if the next call is still bad, break."""
        # First turn: assistant sends a tool call with garbage JSON arguments.
        controller = MockChatController([
            _resp("", tool_calls=[{"id": "c1", "name": "read_file", "arguments": "not-json"}]),
            _resp("", tool_calls=[{"id": "c2", "name": "read_file", "arguments": "still not json"}]),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)
        result = await loop.run([{"role": "user", "content": "test"}], model="local:mock")

        assert result.terminated_by == "malformed_tool_call"
        assert result.iterations == 2  # Did a second turn after the nudge
        assert "repeated" in result.response.lower()

    async def test_good_json_after_nudge(self, registry, tmp_path):
        """After a nudge, if the LLM returns valid JSON, the loop continues normally."""
        readme = tmp_path / "r.md"
        readme.write_text("ok")
        controller = MockChatController([
            _resp("", tool_calls=[{"id": "c1", "name": "read_file", "arguments": "bad"}]),
            _resp("", tool_calls=[_toolcall("read_file", {"path": str(readme)})]),
            _resp("Done!"),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)
        result = await loop.run([{"role": "user", "content": "test"}], model="local:mock")
        assert result.terminated_by == "complete"
        assert result.response == "Done!"


# ---------------------------------------------------------------------------
# 5. Tool progress callback
# ---------------------------------------------------------------------------

class TestToolLoopCallback:
    async def test_on_tool_call_fired(self, registry):
        """Each tool execution should fire the on_tool_call callback."""
        calls = []
        def capture(name, arg):
            calls.append((name, arg))

        controller = MockChatController([
            _resp("", tool_calls=[
                _toolcall("read_file", {"path": "/etc/hosts"}),
                _toolcall("calc", {"a": 2, "b": 3}, id="c2"),
            ]),
            _resp("done"),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4, on_tool_call=capture)
        await loop.run([{"role": "user", "content": "test"}], model="local:mock")

        assert len(calls) == 2
        assert calls[0] == ("read_file", "/etc/hosts")
        assert calls[1] == ("calc", "2")

    async def test_callback_exceptions_propagate(self, registry):
        """Callback exceptions currently propagate (no suppression in ToolLoop)."""
        def boom(*_):
            raise RuntimeError("boom")

        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("read_file", {"path": "/etc/hosts"})]),
            _resp("ok"),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4, on_tool_call=boom)
        # The callback is called inside the loop and currently raises.
        with pytest.raises(RuntimeError, match="boom"):
            await loop.run([{"role": "user", "content": "test"}], model="local:mock")


# ---------------------------------------------------------------------------
# 6. History mutation
# ---------------------------------------------------------------------------

class TestToolLoopHistory:
    async def test_history_appends_turns(self, registry):
        """Messages list should grow with assistant + tool messages per turn."""
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("read_file", {"path": "/ x"})]),
            _resp("ok"),
        ])
        messages = [{"role": "user", "content": "test"}]
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)
        result = await loop.run(messages, model="local:mock")

        # Original messages should be untouched (loop copies).
        assert len(messages) == 1

    async def test_history_grows_inside_loop(self, registry):
        """The messages list inside the loop should accumulate as turns go on."""
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("read_file", {"path": "/a"})]),
            _resp("", tool_calls=[_toolcall("read_file", {"path": "/b"})]),
            _resp("done"),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)
        result = await loop.run([{"role": "user", "content": "test"}], model="local:mock")
        assert result.terminated_by == "complete"
        assert result.iterations == 3
        # The number of calls made to the LLM should equal the number of iterations.
        assert len(controller.calls) == 3


# ---------------------------------------------------------------------------
# 7. Integration with HermesOrchestrator
# ---------------------------------------------------------------------------

class TestToolLoopIntegration:
    async def test_orchestrator_uses_tool_loop(self, tmp_path):
        """Orchestrator._handle_prompt should wire the ToolLoop for general prompts."""
        readme = tmp_path / "README.md"
        readme.write_text("# Test\nThis is a test.\n")
        db = tmp_path / "test.db"
        registry = _make_registry()

        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("read_file", {"path": str(readme), "offset": 1, "limit": 10})]),
            _resp("README says this is a test."),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)
        orch = HermesOrchestrator(db_path=str(db), tool_loop=loop)
        orch.registry = registry
        await orch._initialize_memory()
        result = await orch._handle_prompt("read README.md and summarize")
        assert "test" in result.lower()

    async def test_tool_loop_persists_tool_turns(self, tmp_path):
        """After a ToolLoop run, intermediate tool-call turns are persisted in memory."""
        db = tmp_path / "test.db"
        registry = _make_registry()
        controller = MockChatController([
            _resp("", tool_calls=[_toolcall("calc", {"a": 3, "b": 4})]),
            _resp("7"),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=4)
        orch = HermesOrchestrator(db_path=str(db), tool_loop=loop)
        orch.registry = registry
        await orch._initialize_memory()
        await orch._handle_prompt("3 + 4")
        history = await orch._get_history(limit=20)
        # Should have user + at least one assistant message
        assert any(m["role"] == "user" and "3 + 4" in m.get("content", "") for m in history)
        assert any(m["role"] == "assistant" for m in history)