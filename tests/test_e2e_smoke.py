"""
tests/test_e2e_smoke.py --- End-to-End Smoke Tests (T10)

Prove the loop works on realistic user prompts.
Five canonical scenarios, real tool execution through the canonical
ToolLoop → PluginRegistry path, plus a pure-router assertion for
cloud routing. All LLM calls use a deterministic MockChatController
— the tool handlers are the only real components being exercised.

Latency is captured per scenario into ``smoke/latency.json``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from hermes_lite.registry import PluginRegistry, ToolDefinition
from hermes_lite.orchestrator import ToolLoop
from hermes_lite.llm import ChatRequest, ChatResponse, Tier
from hermes_lite.tools_builtins import register_builtins
from hermes_lite.memory_bridge import MemoryBridge, reset_default_bridge
from hermes_lite.router import LiteRouter, RoutingDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toolcall(name: str, args: dict, id: str = "") -> dict:
    return {
        "id": id or f"call_{name}",
        "name": name,
        "arguments": json.dumps(args),
    }


def _resp(
    content: str = "",
    tool_calls: list[dict] | None = None,
    tier: Tier = "local",
) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop" if not tool_calls else "tool_calls",
        model="mock",
        usage={},
        tier=tier,
    )


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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def smoke_latency() -> dict[str, float]:
    """Shared latency collector over the whole session."""
    return {}


def _dump_latency_report(latency: dict[str, float]) -> None:
    """Write smoke/latency.json."""
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "")
    base = Path(workspace) if workspace else Path(__file__).resolve().parent.parent
    dest = base / "smoke" / "latency.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(latency, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# S1: Find Odoo modules via `search_files` (target=files, pattern=*.py)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFindOdooModules:
    """Smoke-test that search_files against the user's workspace returns
    non-empty results for a plausible glob. Uses the real registry + handler,
    mock LLM to drive the loop."""

    async def test_find_odoo_modules(self, smoke_latency):
        t0 = time.monotonic()
        registry = PluginRegistry(strict_validation=True)
        register_builtins(registry)

        search_path = str(Path.home() / "workspace")
        controller = MockChatController([
            _resp(
                "",
                tool_calls=[_toolcall(
                    "search_files",
                    {"pattern": r".*",
                     "target": "files",
                     "path": search_path,
                     "file_glob": "*.py"},
                )],
            ),
            _resp("Found Python files in the workspace."),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=2)

        result = await loop.run(
            [{"role": "user", "content": f"find all Python files in {search_path}"}],
            model="local:mock",
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        smoke_latency["find_odoo_modules"] = elapsed_ms

        assert result.terminated_by == "complete"
        assert result.tool_calls_made >= 1
        assert any("search_files" in tn for tn in result.tool_names)
        assert result.response != ""


# ---------------------------------------------------------------------------
# S2: Read README.md and summarise
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestReadAndSummarize:
    """Open README.md (real tool handler) and reply summarised."""

    async def test_read_and_summarize(self, smoke_latency, tmp_path):
        t0 = time.monotonic()
        readme = tmp_path / "README.md"
        readme.write_text(
            "# Hermes Lite\n\n"
            "A lightweight agent framework with tool registry, SQLite memory, and orchestration.\n\n"
            "## Features\n"
            "- Plugin registry with strict validation\n"
            "- Two-tier tool loop\n"
            "- Sandbox subprocess execution\n\n"
            "## Getting Started\n"
            "```bash\npip install hermes-lite\nhermes-lite\n```\n"
        )

        registry = PluginRegistry(strict_validation=True)
        register_builtins(registry)

        controller = MockChatController([
            _resp(
                "",
                tool_calls=[_toolcall(
                    "read_file",
                    {"path": str(readme), "offset": 1, "limit": 50},
                )],
            ),
            _resp("This is Hermes Lite --- a lightweight agent framework."),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=2)

        result = await loop.run(
            [{"role": "user", "content": f"open {readme} and tell me what this project does"}],
            model="local:mock",
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        smoke_latency["read_and_summarize"] = elapsed_ms

        assert result.terminated_by == "complete"
        assert result.tool_calls_made == 1
        assert "Hermes" in result.response
        assert "read_file" in result.tool_names


# ---------------------------------------------------------------------------
# S3: Terminal sandbox — list home directory size
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTerminalWithSandbox:
    """Uses the real sandbox via a mock LLM to exercise `terminal`.
    The sandbox log records the invocation."""

    async def test_list_home_dir_size(self, smoke_latency):
        t0 = time.monotonic()
        registry = PluginRegistry(strict_validation=True)
        register_builtins(registry)

        home = str(Path.home())
        controller = MockChatController([
            _resp(
                "",
                tool_calls=[_toolcall(
                    "terminal",
                    {"cmd": f"du -sh {home}", "timeout": 30},
                )],
            ),
            _resp("Your home directory size has been checked."),
        ])
        loop = ToolLoop(registry=registry, chat_fn=controller, max_iterations=2)

        result = await loop.run(
            [{"role": "user", "content": "list my home directory size"}],
            model="local:mock",
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        smoke_latency["terminal_with_sandbox"] = elapsed_ms

        assert result.terminated_by == "complete"
        assert result.tool_calls_made == 1
        assert "terminal" in result.tool_names

        # Verify sandbox log was written
        sandbox_log = Path.home() / ".hermes_lite" / "sandbox.log"
        # The log may or may not exist depending on prior runs;
        # we just confirm it exists after this test runs.
        assert sandbox_log.exists(), f"Expected sandbox log at {sandbox_log}"


# ---------------------------------------------------------------------------
# S4: Cloud routing for 'refactor' intent
# ---------------------------------------------------------------------------

class TestRouteToCloud:
    """A 'refactor this 200-line script' prompt MUST route to cloud.
    This is a pure router assertion --- no LLM API call."""

    def test_route_to_cloud(self, smoke_latency):
        t0 = time.monotonic()
        router = LiteRouter()

        prompt = "refactor this 200-line script to use async/await everywhere"
        decision = router.route(
            prompt=prompt,
            context_tokens=800,
            history_turns=1,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        smoke_latency["route_to_cloud"] = elapsed_ms

        assert isinstance(decision, RoutingDecision)
        assert decision.tier == "cloud", (
            f"Expected cloud, got {decision.tier!r}. "
            f"Reason: {decision.reason}"
        )
        assert decision.fell_back is True
        assert not decision.model_id.startswith("local:")


# ---------------------------------------------------------------------------
# S5: Memory persistence across sessions
# ---------------------------------------------------------------------------

class TestMemoryPersistence:
    """Save a fact via MemoryBridge, then open a fresh bridge and assert it's
    readable. This is a pure bridge test --- no LLM call."""

    def test_memory_persistence(self, smoke_latency, tmp_path, monkeypatch):
        t0 = time.monotonic()
        monkeypatch.setenv("HOME", str(tmp_path))
        reset_default_bridge()

        # Session 1: save facts.
        bridge1 = MemoryBridge(db_path=tmp_path / "memory.db")
        bridge1.add("user", "My machine is an M1 Mac Mini with 8GB RAM")
        bridge1.add("user", "I live in Egypt and work remotely")
        bridge1.close()

        # Session 2: fresh bridge, loaded facts must be present.
        bridge2 = MemoryBridge(db_path=tmp_path / "memory.db")
        entries = bridge2.list("user")
        contents = [e.content for e in entries]
        bridge2.close()

        elapsed_ms = (time.monotonic() - t0) * 1000
        smoke_latency["memory_persistence"] = elapsed_ms

        assert len(entries) >= 2
        assert any("M1 Mac Mini" in c for c in contents)
        assert any("Egypt" in c for c in contents)

        # Verify load_into_prompt includes the facts
        bridge3 = MemoryBridge(db_path=tmp_path / "memory.db")
        prompt_block = bridge3.load_into_prompt(max_chars=2000)
        bridge3.close()
        assert "M1 Mac Mini" in prompt_block
        assert "Egypt" in prompt_block


# ---------------------------------------------------------------------------
# Post-suite: dump latency report
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _dump_latency_after_all(
    smoke_latency: dict[str, float],
    request: pytest.FixtureRequest,
) -> object:
    """Write the latency JSON after ALL tests finish."""
    yield None
    # Only dump if this is the last worker (non-xdist or master)
    if hasattr(request.session.config, "workerinput"):
        # xdist worker — don't write; the controller fixture will.
        return None
    _dump_latency_report(smoke_latency)
    return None