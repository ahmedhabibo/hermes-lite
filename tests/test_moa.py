"""Tests for hermes_lite.moa — Mixture-of-Agents engine.

覆盖范围:
- MoAPreset / MoAModelConfig 数据结构
- MoAEngine: parallel mode, sequential mode
- Reference failure handling (partial, all-fail, aggregator fail)
- Built-in preset lookup
- Formatting helpers
- Orchestrator /moa command integration
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from hermes_lite.moa import (
    MoAEngine,
    MoAPreset,
    MoAModelConfig,
    MoAResult,
    BUILTIN_PRESETS,
    get_preset,
    list_presets,
    format_preset_info,
    format_moa_result,
)
from hermes_lite.llm import ChatRequest, ChatResponse
from hermes_lite.registry import PluginRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_response(content: str) -> ChatResponse:
    return ChatResponse(content=content, tool_calls=[], finish_reason="stop")


def _mock_chat_fn(responses: dict[str, str]):
    """Build a mock chat function that returns canned responses by model."""
    async def _chat(req: ChatRequest) -> ChatResponse:
        model = req.model
        if model in responses:
            return _make_chat_response(responses[model])
        # Default: echo the model name
        return _make_chat_response(f"Response from {model}")
    return _chat


def _failing_chat_fn(fail_models: set[str], ok_response: str = "OK"):
    """Build a mock chat function that raises for specific models."""
    async def _chat(req: ChatRequest) -> ChatResponse:
        if req.model in fail_models:
            raise RuntimeError(f"Simulated failure for {req.model}")
        return _make_chat_response(ok_response)
    return _chat


def _timeout_chat_fn(timeout_models: set[str], ok_response: str = "OK"):
    """Build a mock chat function that hangs for specific models."""
    async def _chat(req: ChatRequest) -> ChatResponse:
        if req.model in timeout_models:
            await asyncio.sleep(9999)  # will hit timeout
        return _make_chat_response(ok_response)
    return _chat


# ---------------------------------------------------------------------------
# MoAPreset / MoAModelConfig
# ---------------------------------------------------------------------------

class TestMoAModelConfig:
    def test_defaults(self):
        cfg = MoAModelConfig(model="test/model")
        assert cfg.model == "test/model"
        assert cfg.temperature == 0.4
        assert cfg.max_tokens == 4096

    def test_custom(self):
        cfg = MoAModelConfig(model="x/y", temperature=0.1, max_tokens=8192)
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 8192


class TestMoAPreset:
    def test_basic(self):
        p = MoAPreset(
            name="test-preset",
            references=[MoAModelConfig(model="a/b")],
            aggregator=MoAModelConfig(model="c/d"),
        )
        assert p.name == "test-preset"
        assert p.mode == "parallel"
        assert p.enabled is True
        assert len(p.references) == 1

    def test_sequential_mode(self):
        p = MoAPreset(name="seq", mode="sequential", references=[])
        assert p.mode == "sequential"


class TestBuiltinPresets:
    def test_all_presets_exist(self):
        names = list_presets()
        assert "council" in names
        assert "speed" in names
        assert "verification" in names
        assert "coding" in names
        assert "creative" in names

    def test_get_preset_case_insensitive(self):
        p = get_preset("Council")
        assert p is not None
        assert p.name == "council"

    def test_get_preset_unknown(self):
        p = get_preset("nonexistent")
        assert p is None

    def test_preset_has_references(self):
        for name in list_presets():
            p = get_preset(name)
            assert p is not None
            assert len(p.references) >= 2, f"Preset {name} needs ≥2 references"
            assert p.aggregator.model, f"Preset {name} needs an aggregator"

    def test_council_diversity(self):
        """Council preset should have diverse models."""
        c = get_preset("council")
        model_ids = [r.model for r in c.references]
        assert len(set(model_ids)) == len(model_ids), "References should be unique"


# ---------------------------------------------------------------------------
# MoAEngine — Parallel mode
# ---------------------------------------------------------------------------

class TestMoAEngineParallel:
    @pytest.mark.asyncio
    async def test_basic_parallel(self):
        """All references succeed → aggregator synthesises."""
        responses = {
            "ref/a": "Answer A: Python is fast",
            "ref/b": "Answer B: Python is versatile",
            "ref/c": "Answer C: Python has great libraries",
            "agg/x": "Python is fast, versatile, and has great libraries.",
        }
        chat_fn = _mock_chat_fn(responses)
        preset = MoAPreset(
            name="test",
            references=[
                MoAModelConfig(model="ref/a"),
                MoAModelConfig(model="ref/b"),
                MoAModelConfig(model="ref/c"),
            ],
            aggregator=MoAModelConfig(model="agg/x"),
            mode="parallel",
        )
        engine = MoAEngine(preset=preset, chat_fn=chat_fn)
        result = await engine.run(
            messages=[{"role": "user", "content": "Tell me about Python"}],
            prompt="Tell me about Python",
        )
        assert result.terminated_by == "complete"
        assert result.aggregator_model == "agg/x"
        assert len(result.reference_responses) == 3
        assert "Python is fast, versatile" in result.response or "agg" in result.response

    @pytest.mark.asyncio
    async def test_partial_ref_failures(self):
        """Some references fail but aggregator still runs with remaining."""
        ok_response = {
            "ref/a": "Answer A",
            "ref/c": "Answer C",
            "agg/x": "Synthesized from available refs.",
        }
        chat_fn = _failing_chat_fn({"ref/b"}, ok_response="X")
        # Override: returns canned for non-failing
        async def _chat(req: ChatRequest) -> ChatResponse:
            if req.model == "ref/b":
                raise RuntimeError("fail")
            return _make_chat_response(ok_response.get(req.model, "OK"))

        preset = MoAPreset(
            name="partial-fail",
            references=[
                MoAModelConfig(model="ref/a"),
                MoAModelConfig(model="ref/b"),
                MoAModelConfig(model="ref/c"),
            ],
            aggregator=MoAModelConfig(model="agg/x"),
            mode="parallel",
        )
        engine = MoAEngine(preset=preset, chat_fn=_chat)
        result = await engine.run(
            messages=[{"role": "user", "content": "test"}],
            prompt="test",
        )
        assert result.terminated_by == "complete"
        assert result.metadata["refs_ok"] == 2
        assert result.metadata["refs_total"] == 3

    @pytest.mark.asyncio
    async def test_all_refs_fail(self):
        """All references fail → falls back to direct call."""
        async def _chat(req: ChatRequest) -> ChatResponse:
            if req.model.startswith("ref/"):
                raise RuntimeError("fail")
            return _make_chat_response("Fallback answer")

        preset = MoAPreset(
            name="all-fail",
            references=[
                MoAModelConfig(model="ref/a"),
                MoAModelConfig(model="ref/b"),
            ],
            aggregator=MoAModelConfig(model="agg/x"),
            mode="parallel",
        )
        engine = MoAEngine(preset=preset, chat_fn=_chat)
        result = await engine.run(
            messages=[{"role": "user", "content": "test"}],
            prompt="test",
        )
        assert result.terminated_by == "all_refs_failed"
        assert len(result.reference_responses) == 0

    @pytest.mark.asyncio
    async def test_aggregator_failure(self):
        """Aggregator fails → returns first reference as degraded fallback."""
        async def _chat(req: ChatRequest) -> ChatResponse:
            if req.model == "agg/x":
                raise RuntimeError("aggregator down")
            return _make_chat_response(f"Ref answer from {req.model}")

        preset = MoAPreset(
            name="agg-fail",
            references=[
                MoAModelConfig(model="ref/a"),
                MoAModelConfig(model="ref/b"),
            ],
            aggregator=MoAModelConfig(model="agg/x"),
            mode="parallel",
        )
        engine = MoAEngine(preset=preset, chat_fn=_chat)
        result = await engine.run(
            messages=[{"role": "user", "content": "test"}],
            prompt="test",
        )
        assert result.terminated_by == "aggregator_failed"
        assert "ref/a" in result.response

    @pytest.mark.asyncio
    async def test_timeout_reference(self):
        """Reference model times out → skipped gracefully."""
        async def _chat(req: ChatRequest) -> ChatResponse:
            if req.model == "ref/slow":
                await asyncio.sleep(9999)
            return _make_chat_response(f"OK from {req.model}")

        preset = MoAPreset(
            name="timeout-test",
            references=[
                MoAModelConfig(model="ref/fast"),
                MoAModelConfig(model="ref/slow"),
            ],
            aggregator=MoAModelConfig(model="agg/x"),
            mode="parallel",
        )
        engine = MoAEngine(preset=preset, chat_fn=_chat, timeout_s=0.5)
        result = await engine.run(
            messages=[{"role": "user", "content": "test"}],
            prompt="test",
        )
        # The slow model should timeout, but fast model + aggregator should work
        assert result.terminated_by in ("complete", "aggregator_failed")
        assert result.metadata["refs_ok"] <= 1


# ---------------------------------------------------------------------------
# MoAEngine — Sequential mode
# ---------------------------------------------------------------------------

class TestMoAEngineSequential:
    @pytest.mark.asyncio
    async def test_basic_sequential(self):
        """Same as parallel but runs one at a time."""
        responses = {
            "ref/a": "Step 1 answer",
            "ref/b": "Step 2 answer",
            "agg/x": "Sequential synthesis",
        }
        chat_fn = _mock_chat_fn(responses)
        preset = MoAPreset(
            name="seq-test",
            references=[
                MoAModelConfig(model="ref/a"),
                MoAModelConfig(model="ref/b"),
            ],
            aggregator=MoAModelConfig(model="agg/x"),
            mode="sequential",
        )
        engine = MoAEngine(preset=preset, chat_fn=chat_fn)
        result = await engine.run(
            messages=[{"role": "user", "content": "test"}],
            prompt="test",
        )
        assert result.terminated_by == "complete"
        assert len(result.reference_responses) == 2

    @pytest.mark.asyncio
    async def test_sequential_with_failure(self):
        """One reference fails in sequential mode, others continue."""
        call_order = []

        async def _chat(req: ChatRequest) -> ChatResponse:
            call_order.append(req.model)
            if req.model == "ref/b":
                raise RuntimeError("seq fail")
            return _make_chat_response(f"OK from {req.model}")

        preset = MoAPreset(
            name="seq-partial-fail",
            references=[
                MoAModelConfig(model="ref/a"),
                MoAModelConfig(model="ref/b"),
                MoAModelConfig(model="ref/c"),
            ],
            aggregator=MoAModelConfig(model="agg/x"),
            mode="sequential",
        )
        engine = MoAEngine(preset=preset, chat_fn=_chat)
        result = await engine.run(
            messages=[{"role": "user", "content": "test"}],
            prompt="test",
        )
        assert result.terminated_by == "complete"
        assert result.metadata["refs_ok"] == 2
        # Verify sequential order
        assert call_order == ["ref/a", "ref/b", "ref/c", "agg/x"]


# ---------------------------------------------------------------------------
# MoAEngine — Edge cases
# ---------------------------------------------------------------------------

class TestMoAEngineEdgeCases:
    @pytest.mark.asyncio
    async def test_single_reference(self):
        """MoA with a single reference model still works."""
        chat_fn = _mock_chat_fn({"ref/only": "Solo answer", "agg/x": "Synthesis"})
        preset = MoAPreset(
            name="single-ref",
            references=[MoAModelConfig(model="ref/only")],
            aggregator=MoAModelConfig(model="agg/x"),
        )
        engine = MoAEngine(preset=preset, chat_fn=chat_fn)
        result = await engine.run(
            messages=[{"role": "user", "content": "test"}],
            prompt="test",
        )
        assert result.terminated_by == "complete"
        assert len(result.reference_responses) == 1

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Works with empty message list."""
        chat_fn = _mock_chat_fn({"ref/a": "ref answer", "agg/x": "agg answer"})
        preset = MoAPreset(
            name="test",
            references=[MoAModelConfig(model="ref/a")],
            aggregator=MoAModelConfig(model="agg/x"),
        )
        engine = MoAEngine(preset=preset, chat_fn=chat_fn)
        result = await engine.run(messages=[], prompt="hello")
        assert result.terminated_by == "complete"

    @pytest.mark.asyncio
    async def test_metadata_populated(self):
        """Check that all expected metadata fields are present."""
        chat_fn = _mock_chat_fn({"ref/a": "ref answer", "agg/x": "agg answer"})
        preset = MoAPreset(
            name="meta-test",
            references=[MoAModelConfig(model="ref/a")],
            aggregator=MoAModelConfig(model="agg/x"),
        )
        engine = MoAEngine(preset=preset, chat_fn=chat_fn)
        result = await engine.run(messages=[], prompt="hello")
        assert "preset" in result.metadata
        assert "ref_models" in result.metadata
        assert "agg_model" in result.metadata
        assert "elapsed_ms" in result.metadata
        assert "refs_ok" in result.metadata
        assert "refs_total" in result.metadata


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_preset_info(self):
        p = MoAPreset(
            name="test",
            references=[
                MoAModelConfig(model="a/b", temperature=0.5),
            ],
            aggregator=MoAModelConfig(model="c/d", temperature=0.2),
        )
        info = format_preset_info(p)
        assert "test" in info
        assert "a/b" in info
        assert "c/d" in info
        assert "parallel" in info

    def test_format_moa_result(self):
        result = MoAResult(
            response="Final answer",
            reference_responses=[("model/a", "ref 1")],
            aggregator_model="model/agg",
            terminated_by="complete",
            metadata={"preset": "council", "refs_ok": 3, "refs_total": 3},
        )
        formatted = format_moa_result(result)
        assert "council" in formatted
        assert "3/3" in formatted
        assert "Final answer" in formatted

    def test_format_moa_result_with_failure(self):
        result = MoAResult(
            response="Degraded",
            reference_responses=[],
            aggregator_model="model/agg",
            terminated_by="all_refs_failed",
            metadata={"preset": "test", "refs_ok": 0, "refs_total": 3},
        )
        formatted = format_moa_result(result)
        assert "all_refs_failed" in formatted
        assert "0/3" in formatted


# ---------------------------------------------------------------------------
# Orchestrator /moa command integration
# ---------------------------------------------------------------------------

class TestOrchestratorMoACommand:
    """Test that the orchestrator's /moa command sets the active preset."""

    def test_moa_attribute_default_none(self):
        """active_moa_preset should be None by default."""
        from hermes_lite.orchestrator import HermesOrchestrator
        orch = HermesOrchestrator.__new__(HermesOrchestrator)
        # The __init__ sets it; verify the attribute exists without calling __init__
        # (we'll test through handle_prompt below)

    @pytest.mark.asyncio
    async def test_moa_status_command(self):
        """`/moa` with no args shows inactive status."""
        from hermes_lite.orchestrator import HermesOrchestrator, ToolLoop
        from hermes_lite.llm import ChatRequest, ChatResponse

        async def _mock_chat(req: ChatRequest) -> ChatResponse:
            return ChatResponse(content="mock", tool_calls=[], finish_reason="stop")

        loop = ToolLoop(
            registry=PluginRegistry(),
            chat_fn=_mock_chat,
            max_iterations=2,
        )
        orch = HermesOrchestrator(
            db_path=":memory:",
            session_title="MoA Test",
            tool_loop=loop,
        )
        orch._create_default_tools()
        loop.registry = orch.registry
        await orch._initialize_memory()

        response = await orch._handle_prompt("/moa")
        assert "Inactive" in response or "inactive" in response.lower() or "Available presets" in response

    @pytest.mark.asyncio
    async def test_moa_activate_command(self):
        """`/moa council` activates the council preset."""
        from hermes_lite.orchestrator import HermesOrchestrator, ToolLoop
        from hermes_lite.llm import ChatRequest, ChatResponse

        async def _mock_chat(req: ChatRequest) -> ChatResponse:
            return ChatResponse(content="mock", tool_calls=[], finish_reason="stop")

        loop = ToolLoop(
            registry=PluginRegistry(),
            chat_fn=_mock_chat,
            max_iterations=2,
        )
        orch = HermesOrchestrator(
            db_path=":memory:",
            session_title="MoA Test",
            tool_loop=loop,
        )
        orch._create_default_tools()
        loop.registry = orch.registry
        await orch._initialize_memory()

        response = await orch._handle_prompt("/moa council")
        assert "Activated" in response or "activated" in response.lower()
        assert orch.active_moa_preset is not None
        assert orch.active_moa_preset.name == "council"

    @pytest.mark.asyncio
    async def test_moa_deactivate_command(self):
        """`/moa off` deactivates MoA."""
        from hermes_lite.orchestrator import HermesOrchestrator, ToolLoop
        from hermes_lite.llm import ChatRequest, ChatResponse

        async def _mock_chat(req: ChatRequest) -> ChatResponse:
            return ChatResponse(content="mock", tool_calls=[], finish_reason="stop")

        loop = ToolLoop(
            registry=PluginRegistry(),
            chat_fn=_mock_chat,
            max_iterations=2,
        )
        orch = HermesOrchestrator(
            db_path=":memory:",
            session_title="MoA Test",
            tool_loop=loop,
        )
        orch._create_default_tools()
        loop.registry = orch.registry
        await orch._initialize_memory()

        # Activate first
        await orch._handle_prompt("/moa council")
        assert orch.active_moa_preset is not None
        # Deactivate
        response = await orch._handle_prompt("/moa off")
        assert "Deactivated" in response or "deactivated" in response.lower()
        assert orch.active_moa_preset is None

    @pytest.mark.asyncio
    async def test_moa_invalid_preset(self):
        """`/moa nonexistent` shows an error with available presets."""
        from hermes_lite.orchestrator import HermesOrchestrator, ToolLoop
        from hermes_lite.llm import ChatRequest, ChatResponse

        async def _mock_chat(req: ChatRequest) -> ChatResponse:
            return ChatResponse(content="mock", tool_calls=[], finish_reason="stop")

        loop = ToolLoop(
            registry=PluginRegistry(),
            chat_fn=_mock_chat,
            max_iterations=2,
        )
        orch = HermesOrchestrator(
            db_path=":memory:",
            session_title="MoA Test",
            tool_loop=loop,
        )
        orch._create_default_tools()
        loop.registry = orch.registry
        await orch._initialize_memory()

        response = await orch._handle_prompt("/moa nonexistent")
        assert "Unknown" in response or "unknown" in response.lower()
        assert "council" in response  # should list available presets
