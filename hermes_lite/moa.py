"""hermes_lite.moa — Mixture-of-Agents (MoA) engine.

Runs multiple "reference" LLMs in parallel on the same prompt, then
feeds their collected outputs to an "aggregator" model that synthesises
a single, high-quality response. This mirrors the MoA pattern used in
Hermes Agent proper.

Architecture
------------
1. **Reference models** — N diverse LLMs generate independent responses
   (parallel via ``asyncio.gather``). Diversity is key: different model
   families produce different perspectives.

2. **Aggregator model** — A stronger synthesis model receives all
   reference outputs plus the original prompt and produces the final
   response. Lower temperature than references for deterministic
   synthesis.

3. **Presets** — Named configurations (council, speed-first,
   verification, coding) that pre-select model rosters and temperatures.

4. **Fallback** — If a reference model fails (rate limit / timeout /
   auth error), the engine continues with the remaining responses. If
   *all* references fail or the aggregator fails, the engine falls back
   to a direct single-model call.

Integration
-----------
- The ``HermesOrchestrator`` activates MoA when the user sends
  ``/moa <preset>`` or when ``HERMES_LITE_MOA_PRESET`` is set.
- When MoA is inactive, the orchestrator runs its normal ToolLoop
  path. When MoA is active, the user prompt goes through the MoA
  pipeline *instead of* (or *before*) the ToolLoop.

Env configuration:

- ``HERMES_LITE_MOA_PRESET``         — default preset name (optional)
- ``HERMES_LITE_MOA_REF_TEMPERATURE`` — default reference temp (0.4)
- ``HERMES_LITE_MOA_AGG_TEMPERATURE`` — default aggregator temp (0.2)
- ``HERMES_LITE_MOA_MAX_TOKENS``      — per-call token budget (4096)
- ``HERMES_LITE_MOA_TIMEOUT_S``      — per-reference timeout (30s)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from hermes_lite.llm import ChatRequest, ChatResponse, chat, Tier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

MoAMode = Literal["parallel", "sequential"]


@dataclass(frozen=True)
class MoAModelConfig:
    """One model entry in an MoA preset.

    Attributes
    ----------
    model:
        Model identifier accepted by :func:`hermes_lite.llm.chat`.
    temperature:
        Sampling temperature override for this model.
    max_tokens:
        Token budget for this model (overrides preset default).
    """

    model: str
    temperature: float = 0.4
    max_tokens: int = 4096


@dataclass
class MoAPreset:
    """A named MoA configuration: reference models + aggregator.

    Attributes
    ----------
    name:
        Preset identifier (e.g. ``"council"``, ``"speed"``).
    references:
        List of reference model configs (the "committee").
    aggregator:
        The aggregator model config (the "synthesizer").
    mode:
        ``"parallel"`` (default) runs all references concurrently;
        ``"sequential"`` runs them one at a time.
    max_tokens:
        Default token budget per call (overridable per-model).
    enabled:
        Whether this preset is active. ``False`` = skip.
    """

    name: str
    references: list[MoAModelConfig] = field(default_factory=list)
    aggregator: MoAModelConfig = field(
        default_factory=lambda: MoAModelConfig(
            model="moonshotai/kimi-k2.6",
            temperature=0.2,
            max_tokens=4096,
        )
    )
    mode: MoAMode = "parallel"
    max_tokens: int = 4096
    enabled: bool = True


@dataclass
class MoAResult:
    """Outcome of a single MoA run.

    Attributes
    ----------
    response:
        The final aggregated response text.
    reference_responses:
        List of (model_id, response_text) from each reference.
    aggregator_model:
        The model that produced the final synthesis.
    iterations:
        Always 1 for MoA (one prompt → N refs → 1 aggregator).
    terminated_by:
        ``"complete"`` on success, ``"all_refs_failed"`` if every
        reference model failed, ``"aggregator_failed"`` if the
        aggregator could not produce a response.
    metadata:
        Structured data for logging / observability.
    """

    response: str
    reference_responses: list[tuple[str, str]] = field(default_factory=list)
    aggregator_model: str = ""
    iterations: int = 1
    terminated_by: str = "complete"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in presets — all verified-working NIM free-tier models (2026-06-30)
#
# Available NIM free models:
#   minimaxai/minimax-m3            — general reasoning (primary default)
#   moonshotai/kimi-k2.6            — strong general model
#   qwen/qwen3.5-397b-a17b          — MoE, efficient at scale
#   minimaxai/minimax-m2.7          — general reasoning
#   deepseek-ai/deepseek-v4-pro     — reasoning/code (timeout-sensitive)
#   nvidia/nemotron-3-ultra-550b-a55b — high throughput
#   deepseek-ai/deepseek-v4-flash   — fast, lightweight
#   qwen/qwen3.5-122b-a10b         — lighter MoE
# ---------------------------------------------------------------------------

BUILTIN_PRESETS: dict[str, MoAPreset] = {
    "council": MoAPreset(
        name="council",
        references=[
            MoAModelConfig(model="minimaxai/minimax-m3", temperature=0.4, max_tokens=4096),
            MoAModelConfig(model="moonshotai/kimi-k2.6", temperature=0.5, max_tokens=4096),
            MoAModelConfig(model="qwen/qwen3.5-397b-a17b", temperature=0.4, max_tokens=4096),
        ],
        aggregator=MoAModelConfig(
            model="moonshotai/kimi-k2.6",
            temperature=0.2,
            max_tokens=4096,
        ),
        mode="parallel",
        max_tokens=4096,
    ),
    "speed": MoAPreset(
        name="speed",
        references=[
            MoAModelConfig(model="minimaxai/minimax-m3", temperature=0.4, max_tokens=2048),
            MoAModelConfig(model="deepseek-ai/deepseek-v4-flash", temperature=0.5, max_tokens=2048),
            MoAModelConfig(model="qwen/qwen3.5-122b-a10b", temperature=0.4, max_tokens=2048),
        ],
        aggregator=MoAModelConfig(
            model="minimaxai/minimax-m3",
            temperature=0.3,
            max_tokens=4096,
        ),
        mode="parallel",
        max_tokens=2048,
    ),
    "verification": MoAPreset(
        name="verification",
        references=[
            MoAModelConfig(model="moonshotai/kimi-k2.6", temperature=0.1, max_tokens=4096),
            MoAModelConfig(model="qwen/qwen3.5-397b-a17b", temperature=0.1, max_tokens=4096),
            MoAModelConfig(model="deepseek-ai/deepseek-v4-pro", temperature=0.1, max_tokens=4096),
        ],
        aggregator=MoAModelConfig(
            model="minimaxai/minimax-m3",
            temperature=0.05,
            max_tokens=8192,
        ),
        mode="parallel",
        max_tokens=4096,
    ),
    "coding": MoAPreset(
        name="coding",
        references=[
            MoAModelConfig(model="deepseek-ai/deepseek-v4-pro", temperature=0.3, max_tokens=4096),
            MoAModelConfig(model="qwen/qwen3.5-397b-a17b", temperature=0.3, max_tokens=4096),
        ],
        aggregator=MoAModelConfig(
            model="deepseek-ai/deepseek-v4-pro",
            temperature=0.1,
            max_tokens=8192,
        ),
        mode="parallel",
        max_tokens=4096,
    ),
    "creative": MoAPreset(
        name="creative",
        references=[
            MoAModelConfig(model="minimaxai/minimax-m3", temperature=0.8, max_tokens=4096),
            MoAModelConfig(model="moonshotai/kimi-k2.6", temperature=0.9, max_tokens=4096),
            MoAModelConfig(model="qwen/qwen3.5-122b-a10b", temperature=0.7, max_tokens=4096),
        ],
        aggregator=MoAModelConfig(
            model="minimaxai/minimax-m3",
            temperature=0.4,
            max_tokens=8192,
        ),
        mode="parallel",
        max_tokens=4096,
    ),
}

# Resolve env overrides for defaults
_DEFAULT_REF_TEMP = float(os.environ.get("HERMES_LITE_MOA_REF_TEMPERATURE", "0.4"))
_DEFAULT_AGG_TEMP = float(os.environ.get("HERMES_LITE_MOA_AGG_TEMPERATURE", "0.2"))
_DEFAULT_MAX_TOKENS = int(os.environ.get("HERMES_LITE_MOA_MAX_TOKENS", "4096"))
_DEFAULT_TIMEOUT_S = float(os.environ.get("HERMES_LITE_MOA_TIMEOUT_S", "60"))


# ---------------------------------------------------------------------------
# MoA Engine
# ---------------------------------------------------------------------------


class MoAEngine:
    """Runs the Mixture-of-Agents pipeline.

    Usage::

        engine = MoAEngine(preset=BUILTIN_PRESETS["council"])
        result = await engine.run(messages=[...], prompt="Explain quantum computing")

    The engine is stateless between calls — each ``run()`` is independent.
    """

    def __init__(
        self,
        preset: MoAPreset | None = None,
        chat_fn=None,
        timeout_s: float | None = None,
    ) -> None:
        self.preset = preset or BUILTIN_PRESETS["council"]
        self.chat_fn = chat_fn or chat
        self.timeout_s = timeout_s or _DEFAULT_TIMEOUT_S

    # -- public API --------------------------------------------------------

    async def run(
        self,
        messages: list[dict[str, Any]],
        prompt: str,
    ) -> MoAResult:
        """Execute the Mixture-of-Agents pipeline.

        1. Send the same prompt to all reference models in parallel
        2. Collect successful responses (skipping failures)
        3. Feed collected responses + original prompt to the aggregator
        4. Return the aggregated response

        If *all* reference models fail, returns a direct single-model
        fallback. If the aggregator fails, returns the first successful
        reference response as a degraded fallback.
        """
        started = time.monotonic()
        ref_models = self.preset.references
        agg_config = self.preset.aggregator

        # Phase 1: Reference models (parallel or sequential)
        if self.preset.mode == "parallel":
            ref_results = await self._run_references_parallel(messages, prompt, ref_models)
        else:
            ref_results = await self._run_references_sequential(messages, prompt, ref_models)

        # Collect successful references
        ref_responses: list[tuple[str, str]] = []
        for model_id, response_text, success in ref_results:
            if success and response_text:
                ref_responses.append((model_id, response_text))

        elapsed_ms = int((time.monotonic() - started) * 1000)

        # All references failed — fall back to direct call
        if not ref_responses:
            logger.warning("MoA: all %d reference models failed", len(ref_models))
            fallback_text = await self._direct_fallback(messages, prompt, agg_config)
            return MoAResult(
                response=fallback_text or "MoA: all reference models failed and aggregator fallback also failed.",
                reference_responses=[],
                aggregator_model=agg_config.model,
                iterations=1,
                terminated_by="all_refs_failed",
                metadata={
                    "preset": self.preset.name,
                    "ref_models": [r.model for r in ref_models],
                    "elapsed_ms": elapsed_ms,
                    "refs_ok": 0,
                    "refs_total": len(ref_models),
                },
            )

        # Phase 2: Aggregator synthesises
        agg_response = await self._run_aggregator(messages, prompt, ref_responses, agg_config)

        if agg_response:
            return MoAResult(
                response=agg_response,
                reference_responses=ref_responses,
                aggregator_model=agg_config.model,
                iterations=1,
                terminated_by="complete",
                metadata={
                    "preset": self.preset.name,
                    "ref_models": [r.model for r in ref_models],
                    "agg_model": agg_config.model,
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "refs_ok": len(ref_responses),
                    "refs_total": len(ref_models),
                },
            )

        # Aggregator failed — return the best reference as degraded fallback
        logger.warning("MoA: aggregator failed, returning first reference as fallback")
        _, best_ref = ref_responses[0]
        return MoAResult(
            response=best_ref,
            reference_responses=ref_responses,
            aggregator_model=agg_config.model,
            iterations=1,
            terminated_by="aggregator_failed",
            metadata={
                "preset": self.preset.name,
                "ref_models": [r.model for r in ref_models],
                "agg_model": agg_config.model,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "refs_ok": len(ref_responses),
                "refs_total": len(ref_models),
                "fallback": "first_reference",
            },
        )

    # -- internals ---------------------------------------------------------

    async def _run_references_parallel(
        self,
        messages: list[dict[str, Any]],
        prompt: str,
        ref_models: list[MoAModelConfig],
    ) -> list[tuple[str, str, bool]]:
        """Run all reference models concurrently via asyncio.gather.

        Returns list of (model_id, response_text, success) tuples.
        """
        tasks = [
            self._call_reference(messages, prompt, config)
            for config in ref_models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: list[tuple[str, str, bool]] = []
        for config, result in zip(ref_models, results):
            if isinstance(result, Exception):
                logger.warning(
                    "MoA reference %s failed: %s: %s",
                    config.model, type(result).__name__, result,
                )
                output.append((config.model, "", False))
            else:
                model_id, response_text, success = result
                output.append((model_id, response_text, success))
        return output

    async def _run_references_sequential(
        self,
        messages: list[dict[str, Any]],
        prompt: str,
        ref_models: list[MoAModelConfig],
    ) -> list[tuple[str, str, bool]]:
        """Run reference models one at a time (for rate-limited contexts).

        Returns list of (model_id, response_text, success) tuples.
        """
        output: list[tuple[str, str, bool]] = []
        for config in ref_models:
            try:
                result = await self._call_reference(messages, prompt, config)
                output.append(result)
            except Exception as exc:
                logger.warning(
                    "MoA reference %s failed: %s: %s",
                    config.model, type(exc).__name__, exc,
                )
                output.append((config.model, "", False))
        return output

    async def _call_reference(
        self,
        messages: list[dict[str, Any]],
        prompt: str,
        config: MoAModelConfig,
    ) -> tuple[str, str, bool]:
        """Call a single reference model with a timeout.

        Returns (model_id, response_text, success).
        """
        # Build a simple message list: original context + current prompt
        ref_messages = list(messages)

        try:
            resp: ChatResponse = await asyncio.wait_for(
                self.chat_fn(ChatRequest(
                    messages=ref_messages,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens or self.preset.max_tokens,
                )),
                timeout=self.timeout_s,
            )
            return (config.model, resp.content or "", True)
        except asyncio.TimeoutError:
            logger.warning("MoA reference %s timed out after %.0fs", config.model, self.timeout_s)
            return (config.model, "", False)
        except Exception as exc:
            logger.warning(
                "MoA reference %s error: %s: %s",
                config.model, type(exc).__name__, exc,
            )
            return (config.model, "", False)

    async def _run_aggregator(
        self,
        original_messages: list[dict[str, Any]],
        original_prompt: str,
        ref_responses: list[tuple[str, str]],
        agg_config: MoAModelConfig,
    ) -> str | None:
        """Call the aggregator model with all reference outputs.

        Returns the aggregated response text, or None on failure.
        """
        # Build the aggregator prompt: insert reference outputs
        # between the system message and the user prompt.
        agg_messages: list[dict[str, Any]] = []

        # Carry forward the system message (if present)
        for msg in original_messages:
            if msg.get("role") == "system":
                agg_messages.append(msg)
                break

        # Add MoA instruction block
        agg_instructions = (
            "You are synthesising a response from multiple AI model perspectives. "
            "Below are independent responses from different reference models. "
            "Integrate the best elements from each, resolve contradictions, "
            "and produce a single coherent, comprehensive response.\n\n"
            "Do NOT just copy one response — synthetically merge the strongest "
            "points from each perspective.\n\n"
        )

        # Add reference outputs
        ref_block_parts = []
        for i, (model_id, response) in enumerate(ref_responses, 1):
            ref_block_parts.append(f"**Reference {i} ({model_id}):**\n{response}")
        ref_block = "\n\n---\n\n".join(ref_block_parts)

        agg_messages.append({
            "role": "system",
            "content": agg_instructions,
        })

        # Add the original history (skip system, we already added it)
        for msg in original_messages:
            if msg.get("role") != "system":
                agg_messages.append(msg)

        # Add the reference outputs as a "system" context message
        agg_messages.append({
            "role": "user",
            "content": (
                f"Here are the reference model responses:\n\n"
                f"{ref_block}\n\n"
                f"---\n\n"
                f"Now synthesise a comprehensive response to my original question: {original_prompt}"
            ),
        })

        try:
            resp: ChatResponse = await asyncio.wait_for(
                self.chat_fn(ChatRequest(
                    messages=agg_messages,
                    model=agg_config.model,
                    temperature=agg_config.temperature,
                    max_tokens=agg_config.max_tokens or self.preset.max_tokens,
                )),
                timeout=self.timeout_s * 2,  # aggregator gets double timeout
            )
            return resp.content or ""
        except asyncio.TimeoutError:
            logger.warning("MoA aggregator %s timed out", agg_config.model)
            return None
        except Exception as exc:
            logger.warning(
                "MoA aggregator %s error: %s: %s",
                agg_config.model, type(exc).__name__, exc,
            )
            return None

    async def _direct_fallback(
        self,
        messages: list[dict[str, Any]],
        prompt: str,
        agg_config: MoAModelConfig,
    ) -> str | None:
        """Direct single-model call as a last resort."""
        try:
            resp: ChatResponse = await asyncio.wait_for(
                self.chat_fn(ChatRequest(
                    messages=messages,
                    model=agg_config.model,
                    temperature=0.3,
                    max_tokens=self.preset.max_tokens,
                )),
                timeout=self.timeout_s,
            )
            return resp.content or ""
        except Exception as exc:
            logger.error("MoA direct fallback failed: %s: %s", type(exc).__name__, exc)
            return None


# ---------------------------------------------------------------------------
# Preset lookup
# ---------------------------------------------------------------------------


def get_preset(name: str) -> MoAPreset | None:
    """Look up a built-in preset by name (case-insensitive).

    Returns None if no preset matches.
    """
    return BUILTIN_PRESETS.get(name.lower())


def list_presets() -> list[str]:
    """Return the names of all available built-in presets."""
    return list(BUILTIN_PRESETS.keys())


# ---------------------------------------------------------------------------
# Formatting helpers (used by orchestrator / CLI)
# ---------------------------------------------------------------------------


def format_preset_info(preset: MoAPreset) -> str:
    """Human-readable summary of a MoAPreset (for /moa command)."""
    lines = [
        f"**MoA Preset: {preset.name}**\\n",
        f"  Mode: {preset.mode}",
        f"  Max tokens: {preset.max_tokens}",
        "",
        "  **Reference models:**",
    ]
    for i, ref in enumerate(preset.references, 1):
        lines.append(f"    {i}. `{ref.model}` (temp={ref.temperature}, max_tokens={ref.max_tokens})")
    lines.append("")
    lines.append(f"  **Aggregator:** `{preset.aggregator.model}` (temp={preset.aggregator.temperature}, max_tokens={preset.aggregator.max_tokens})")
    return "\n".join(lines)


def format_moa_result(result: MoAResult) -> str:
    """Format an MoAResult for CLI display."""
    parts = []
    parts.append(f"🧠 _MoA: {result.metadata.get('preset', 'unknown')}_")
    refs_ok = result.metadata.get("refs_ok", 0)
    refs_total = result.metadata.get("refs_total", 0)
    parts.append(f"   {refs_ok}/{refs_total} references succeeded · aggregator: `{result.aggregator_model}`")
    if result.terminated_by != "complete":
        parts.append(f"   ⚠ Ended: {result.terminated_by}")
    parts.append("")
    parts.append(result.response)
    return "\n".join(parts)


__all__ = [
    "MoAEngine",
    "MoAPreset",
    "MoAModelConfig",
    "MoAResult",
    "MoAMode",
    "BUILTIN_PRESETS",
    "get_preset",
    "list_presets",
    "format_preset_info",
    "format_moa_result",
]
