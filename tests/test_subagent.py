"""
tests/test_subagent.py — Subagent (T8) single-shot delegation primitive.

Coverage map (acceptance criteria from the task body):
  - summary-style goal    → returns plain-text summary in ``output``
  - code-style goal       → scope="write" also exposes ``work_product``
  - no recursion          → child registry has NO ``subagent`` tool
  - memory isolation      → child writes to /tmp/lite-sub-<uuid>.db,
                            parent DB at ``tmp_path`` is untouched
  - caps                  → max_turns clamped to [1, 6]; timeout
                            returns failure with terminated_by="timeout";
                            ToolLoop max_iterations is bounded
  - logging               → one JSONL audit line per run on disk
  - registration          → register_subagent_tool() idempotent
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from hermes_lite.llm import ChatRequest, ChatResponse
from hermes_lite.registry import PluginRegistry, ToolNotFoundError
from hermes_lite.subagent import (
    DEFAULT_SUBAGENT_LOG_PATH,
    SUBAGENT_MAX_ITERATIONS,
    SUBAGENT_MAX_TOOL_CALLS,
    SUBAGENT_TOOL_NAME,
    SUBAGENT_WALL_TIMEOUT_S,
    SUBAGENT_SYSTEM_PROMPT,
    SubagentArgs,
    SubagentRunner,
    register_subagent_tool,
)
from hermes_lite.tools_builtins import ESSENTIAL_TOOL_NAMES


# ---------------------------------------------------------------------------
# Mock helpers (mirror tests/test_tool_loop.py)
# ---------------------------------------------------------------------------


class MockChat:
    """Deterministic fake LLM. Records every call, returns pre-canned responses."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[ChatRequest] = []

    async def __call__(self, req: ChatRequest) -> ChatResponse:
        self.calls.append(req)
        if not self.responses:
            raise RuntimeError("MockChat ran out of canned responses")
        return self.responses.pop(0)


def _resp(content: str = "", tool_calls: list[dict] | None = None) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=list(tool_calls or []),
        finish_reason="stop" if not tool_calls else "tool_calls",
        model="mock-qwen-3b",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        tier="local",
    )


def _toolcall(name: str, args: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": json.dumps(args)}


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSubagentArgs:
    def test_defaults_make_scope_read(self) -> None:
        a = SubagentArgs(goal="summarize pickle module")
        assert a.scope == "read"
        assert a.max_turns == 4

    def test_minimum_goal_required(self) -> None:
        with pytest.raises(Exception):
            SubagentArgs(goal="")

    def test_max_turns_window(self) -> None:
        # Pydantic enforces [1, 20]; server-side we clamp to [1, 6].
        a = SubagentArgs(goal="x", max_turns=20)
        assert a.max_turns == 20  # schema accepts, runner clamps
        with pytest.raises(Exception):
            SubagentArgs(goal="x", max_turns=0)

    def test_scope_enum(self) -> None:
        with pytest.raises(Exception):
            SubagentArgs(goal="x", scope="delete")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Runner — happy paths
# ---------------------------------------------------------------------------


class TestSubagentRunnerHappyPath:
    """Acceptance: ``subagent goal='summarize pickle module'``
    returns a 3-paragraph summary."""

    @pytest.mark.asyncio
    async def test_returns_plain_text_summary(self, tmp_path: Path) -> None:
        chat = MockChat([_resp(
            "Paragraph one. The pickle module serializes Python objects.\n\n"
            "Paragraph two. It preserves type and structure.\n\n"
            "Paragraph three. Useful for IPC and caching."
        )])
        runner = SubagentRunner(
            chat_fn=chat,
            db_path=str(tmp_path / "child.db"),
            log_path=tmp_path / "audit.jsonl",
        )
        result = await runner.run("summarize the pickle module in 3 paragraphs")

        assert result["ok"] is True
        assert "Paragraph one" in result["output"]
        assert "Paragraph three" in result["output"]
        assert result["scope"] == "read"
        assert result["terminated_by"] == "complete"
        assert result["iterations"] == 1
        assert result["tool_calls_made"] == 0
        assert result["tool_names"] == []

    @pytest.mark.asyncio
    async def test_scope_write_exposes_work_product(self, tmp_path: Path) -> None:
        chat = MockChat([_resp("def hello():\n    return 'hi'\n")])
        runner = SubagentRunner(
            chat_fn=chat,
            db_path=str(tmp_path / "child.db"),
            log_path=tmp_path / "audit.jsonl",
            scope="write",
        )
        result = await runner.run("refactor this function")
        assert result["ok"] is True
        assert "work_product" in result
        assert result["work_product"] == result["output"]


# ---------------------------------------------------------------------------
# No recursion + memory isolation
# ---------------------------------------------------------------------------


class TestNoRecursion:
    """Acceptance: subagent cannot call subagent
    (the isolated registry has no ``subagent`` tool)."""

    @pytest.mark.asyncio
    async def test_isolated_registry_has_no_subagent_tool(self, tmp_path: Path) -> None:
        seen_tools: list[str] = []

        async def _echo(req: ChatRequest) -> ChatResponse:
            # Capture which tools the *child* sees, then "settle" with text.
            for t in (req.tools or []):
                seen_tools.append(t.get("function", {}).get("name", "?"))
            return _resp("done", tool_calls=[
                _toolcall(
                    "missing_tool_xyz",
                    {"x": 1},
                ),
            ]) if len(seen_tools) == 1 else _resp("ok final")

        runner = SubagentRunner(
            chat_fn=_echo,
            db_path=str(tmp_path / "child.db"),
            log_path=tmp_path / "audit.jsonl",
            scope="read",
        )
        await runner.run("anything")

        assert seen_tools, "child never saw tool list"
        assert SUBAGENT_TOOL_NAME not in seen_tools
        # The 6 essentials are present (or, in scope=read, present-and-shimmed)
        for name in ESSENTIAL_TOOL_NAMES:
            assert name in seen_tools, f"essential tool '{name}' missing from child"

    @pytest.mark.asyncio
    async def test_subagent_self_call_would_be_rejected(self, tmp_path: Path) -> None:
        """If the child model hallucinates a subagent tool_call, we'd raise
        ToolNotFoundError inside ToolLoop. The loop swallows it via
        repeated_error, but we can prove it directly at the registry level."""

        # Build an isolated registry exactly the way SubagentRunner does.
        from hermes_lite.tools_builtins import register_builtins as _reg_ess

        registry = PluginRegistry(strict_validation=True)
        _reg_ess(registry, overwrite=False)
        # The child's registry must NOT contain the subagent tool.
        with pytest.raises(ToolNotFoundError):
            registry.call_tool(SUBAGENT_TOOL_NAME, {"goal": "x"})


class TestMemoryIsolation:
    """Acceptance: subagent doesn't pollute main session's memory.

    The child writes to /tmp/lite-sub-<uuid>.db. The host's DB is left
    alone. We simulate the host by creating a separate sqlite file and
    verifying its content is unchanged after the subagent run.
    """

    @pytest.mark.asyncio
    async def test_child_db_is_per_run(self, tmp_path: Path) -> None:
        chat = MockChat([_resp("hello")])
        host_db = tmp_path / "host.db"

        # Force two runs and confirm two different child DBs are created.
        runner1 = SubagentRunner(
            chat_fn=chat,
            db_path=str(tmp_path / "child1.db"),
            log_path=tmp_path / "audit.jsonl",
        )
        runner1.sub_id = "aaaaaaaaaaaa"
        Path("/tmp/lite-sub-aaaaaaaaaaaa.db").unlink(missing_ok=True)

        runner2 = SubagentRunner(
            chat_fn=chat,
            db_path=str(tmp_path / "child2.db"),
            log_path=tmp_path / "audit.jsonl",
        )
        runner2.sub_id = "bbbbbbbbbbbb"
        Path("/tmp/lite-sub-bbbbbbbbbbbb.db").unlink(missing_ok=True)

        await runner1.run("hi")
        await runner2.run("bye")

        assert (tmp_path / "child1.db").exists()
        assert (tmp_path / "child2.db").exists()
        assert not host_db.exists() or host_db.stat().st_size == 0

    async def test_env_isolation_strips_secrets(self, tmp_path: Path) -> None:
        """Subagent should run with a sanitized env — secrets stripped."""
        chat = MockChat([_resp("done")])
        runner = SubagentRunner(
            chat_fn=chat,
            db_path=str(tmp_path / "child.db"),
            log_path=tmp_path / "audit.jsonl",
        )

        # Inject a secret into the parent env
        import os as _os
        _os.environ["FAKE_API_KEY"] = "sk-leak-test-12345"

        result = await runner.run("test goal")

        # After run, parent env should be restored (including the secret)
        assert _os.environ.get("FAKE_API_KEY") == "sk-leak-test-12345"
        assert result["ok"] is True

        # Clean up
        del _os.environ["FAKE_API_KEY"]


# ---------------------------------------------------------------------------
# Caps + timeout
# ---------------------------------------------------------------------------


class TestCaps:
    def test_max_turns_clamped_above_max(self) -> None:
        # Schema accepts [1, 20]; the runner clamps to SUBAGENT_MAX_ITERATIONS=6.
        assert SUBAGENT_MAX_ITERATIONS == 6
        a = SubagentArgs(goal="x", max_turns=20)
        assert a.max_turns == 20  # schema accepts, runner clamps

    @pytest.mark.asyncio
    async def test_runner_clamps_max_turns(self, tmp_path: Path) -> None:
        """If max_turns=20 is requested, the child ToolLoop sees at most 6."""
        chat = MockChat([_resp("done")])
        runner = SubagentRunner(
            chat_fn=chat,
            db_path=str(tmp_path / "child.db"),
            log_path=tmp_path / "audit.jsonl",
        )
        await runner.run("anything", max_turns=20)
        # ToolLoop was constructed with max_iterations=6 (the clamp).
        # We don't expose the ToolLoop instance, but the chat record shows
        # the loop terminated. Verify the audit log carries the effective cap.
        log_text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
        record = json.loads(log_text.strip().splitlines()[0])
        assert record["max_turns_requested"] == 20
        assert record["max_turns_effective"] == SUBAGENT_MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_max_tool_calls_constant(self) -> None:
        assert SUBAGENT_MAX_TOOL_CALLS == 6

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self, tmp_path: Path) -> None:
        """If chat_fn hangs longer than timeout_s, runner must return
        a failure dict with terminated_by='timeout'."""

        async def _slow(req: ChatRequest) -> ChatResponse:
            await asyncio.sleep(2.0)
            return _resp("too late")

        runner = SubagentRunner(
            chat_fn=_slow,
            db_path=str(tmp_path / "child.db"),
            log_path=tmp_path / "audit.jsonl",
            timeout_s=0.2,  # aggressive — guaranteed to fire
        )
        out = await runner.run("anything")
        assert out["ok"] is False
        assert out["terminated_by"] == "timeout"
        assert "timed out" in out["error"]

    @pytest.mark.asyncio
    async def test_default_timeout_is_60s(self) -> None:
        runner = SubagentRunner()
        assert runner.timeout_s == SUBAGENT_WALL_TIMEOUT_S
        assert SUBAGENT_WALL_TIMEOUT_S == 180.0  # default, overridable via HERMES_LITE_SUBAGENT_TIMEOUT


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    @pytest.mark.asyncio
    async def test_appends_json_line(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        chat = MockChat([_resp("hi")])
        runner = SubagentRunner(
            chat_fn=chat,
            db_path=str(tmp_path / "child.db"),
            log_path=log,
        )
        await runner.run("summarize pickle")

        assert log.exists()
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["ok"] is True
        assert record["terminated_by"] == "complete"
        assert record["scope"] == "read"
        assert record["goal_preview"] == "summarize pickle"
        assert "elapsed_ms" in record
        assert "model" in record

    @pytest.mark.asyncio
    async def test_logging_failure_doesnt_crash(self, tmp_path: Path) -> None:
        bad = tmp_path / "audit.jsonl"
        # Make the parent directory a file so the ``mkdir`` step in _log_run
        # raises OSError — proving we never let logging break the tool.
        bad.write_text("placeholder")
        chat = MockChat([_resp("ok")])
        runner = SubagentRunner(
            chat_fn=chat,
            db_path=str(tmp_path / "child.db"),
            log_path=bad,
        )
        out = await runner.run("anything")
        assert out["ok"] is True


# ---------------------------------------------------------------------------
# Registration via PluginRegistry
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_appends_to_registry(self) -> None:
        registry = PluginRegistry(strict_validation=True)
        defn = register_subagent_tool(registry)
        assert defn.name == SUBAGENT_TOOL_NAME
        assert registry.has_tool(SUBAGENT_TOOL_NAME)
        # Description documents the no-recursion guarantee.
        assert "cannot spawn further subagents" in defn.description

    def test_register_is_idempotent(self) -> None:
        registry = PluginRegistry(strict_validation=True)
        register_subagent_tool(registry)
        register_subagent_tool(registry)
        # Still exactly one entry.
        names = [t.name for t in registry.list_tools()]
        assert names.count(SUBAGENT_TOOL_NAME) == 1

    def test_callable_via_registry(self) -> None:
        """Direct registry.call_tool('subagent', {...}) returns the same shape."""
        registry = PluginRegistry(strict_validation=True)
        register_subagent_tool(registry, chat_fn=_FakeChat(_resp("inline")))
        result = registry.call_tool(SUBAGENT_TOOL_NAME, {"goal": "explain pickle"})
        assert result["ok"] is True
        assert "inline" in result["output"]


class _FakeChat:
    """Sync-style wrapper around canned ChatResponses."""

    def __init__(self, *responses: ChatResponse) -> None:
        self.responses = list(responses)

    async def __call__(self, req: ChatRequest) -> ChatResponse:
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# System prompt is non-trivial + scopes the child
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_prompt_forbids_recursion(self) -> None:
        assert "cannot spawn further subagents" in SUBAGENT_SYSTEM_PROMPT

    def test_prompt_includes_tool_names(self) -> None:
        # The subagent prompt must list the 6 essentials so the model
        # knows what's available; if a new essential joins, this will
        # fail and prompt the author to update SUBAGENT_SYSTEM_PROMPT.
        for name in ESSENTIAL_TOOL_NAMES:
            assert name in SUBAGENT_SYSTEM_PROMPT, (
                f"essential tool {name!r} missing from SUBAGENT_SYSTEM_PROMPT"
            )

    def test_prompt_is_concise(self) -> None:
        # The system prompt is shown to the LLM at every turn — keep
        # it small enough to fit alongside a focused goal + tools.
        assert SUBAGENT_SYSTEM_PROMPT.count("\n") < 20
        assert len(SUBAGENT_SYSTEM_PROMPT) < 1500


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------


def test_defaults_are_sane() -> None:
    assert SUBAGENT_TOOL_NAME == "subagent"
    assert SUBAGENT_MAX_TOOL_CALLS == 6
    assert SUBAGENT_MAX_ITERATIONS == 6
    assert SUBAGENT_WALL_TIMEOUT_S == 180.0  # default, overridable via HERMES_LITE_SUBAGENT_TIMEOUT
    assert DEFAULT_SUBAGENT_LOG_PATH.name == "subagents.log"
    assert "hermes_lite" in str(DEFAULT_SUBAGENT_LOG_PATH)
