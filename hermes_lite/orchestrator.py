"""
hermes_lite/orchestrator.py — Orchestrator Engine

Wires the PluginRegistry (tools), SQLite memory layer (conversation history),
LLM client, and CLI shell into a working agent loop.

Key components:
- HermesOrchestrator: Coordinates tool registry + memory + prompt handling
- ToolLoop / ToolLoopResult: re-exported from hermes_lite.tool_loop
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from hermes_lite.registry import PluginRegistry
from hermes_lite.memory import (
    AsyncSQLitePool,
    ensure_schema,
    session_context,
    insert_message,
    get_messages,
)
from hermes_lite.llm import AllKeysExhausted, ChatRequest, chat_stream
from hermes_lite.cli import run_cli
from hermes_lite.router import LiteRouter, RoutingDecision
from hermes_lite.tools_builtins import register_builtins as _register_essentials
from hermes_lite.observability import log_turn
from hermes_lite.moa import (
    MoAEngine,
    MoAPreset,
    MoAResult,
    format_moa_result,
)
from hermes_lite.config import get_config
from hermes_lite.commands import (
    dispatch_command,
    dispatch_direct_tool,
    is_command,
    is_direct_tool,
)
from hermes_lite.tool_loop import ToolLoop, ToolLoopResult
from hermes_lite.sanitize import scrub_control_tokens
from hermes_lite.prompts import build_system_prompt

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("hermes-lite")
except Exception:
    __version__ = "0.9.0"


def _dedupe(seq: list[str]) -> list[str]:
    """Return a de-duplicated copy of *seq*, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class HermesOrchestrator:
    """Wires the PluginRegistry, SQLite memory, and CLI into a working loop.

    Typical usage::

        orchestrator = HermesOrchestrator()
        orchestrator.start()
    """

    def __init__(
        self,
        db_path: str | Path = "~/.hermes_lite/sessions.db",
        session_title: str = "Hermes-Lite Session",
        router: LiteRouter | None = None,
        tool_loop: ToolLoop | None = None,
        auth_token: str | None = None,
    ) -> None:
        cfg = get_config()
        self.db_path = str(Path(db_path).expanduser())
        self.session_title = session_title
        self.session_id: str = ""
        self.pool: AsyncSQLitePool | None = None
        if auth_token is None:
            # Live env wins so tests/handlers changing HERMES_LITE_AUTH_TOKEN
            # at runtime still see it, even though get_config() is cached.
            import os
            env_token = os.environ.get("HERMES_LITE_AUTH_TOKEN")
            auth_token = env_token or cfg.auth_token or None
        self.auth_token = auth_token
        self.registry = PluginRegistry(strict_validation=True, auth_token=auth_token)
        self.router: LiteRouter = router or LiteRouter()
        self._tool_loop: ToolLoop | None = tool_loop
        self.active_moa_preset: MoAPreset | None = None
        # Explicit routing override: None = auto, "cloud" = force cloud
        self.force_tier: str | None = None
        if cfg.moa_preset:
            from hermes_lite.moa import get_preset
            self.active_moa_preset = get_preset(cfg.moa_preset)

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    # Back-compat for code that still reads _force_tier
    @property
    def _force_tier(self) -> str | None:
        return self.force_tier

    @_force_tier.setter
    def _force_tier(self, value: str | None) -> None:
        self.force_tier = value

    def _create_default_tools(self) -> None:
        """Register essentials + subagent (deferred import for cycle break)."""
        _register_essentials(self.registry)
        from hermes_lite.subagent import register_subagent_tool
        register_subagent_tool(self.registry)

    @property
    def tool_loop(self) -> ToolLoop:
        if self._tool_loop is None:
            self._tool_loop = ToolLoop(
                registry=self.registry,
                on_tool_call=self._print_tool_progress,
            )
        return self._tool_loop

    @staticmethod
    def _print_tool_progress(tool_name: str, first_arg: str) -> None:
        preview = first_arg[:50] if first_arg else ""
        if preview:
            print(f"⚡ {tool_name}({preview!r})", file=sys.stderr)
        else:
            print(f"⚡ {tool_name}(...)", file=sys.stderr)

    def get_tool_descriptions(self) -> list[dict[str, Any]]:
        return self.registry.tool_descriptions()

    async def _initialize_memory(self) -> None:
        self.pool = AsyncSQLitePool(self.db_path, min_size=1, max_size=2)
        await self.pool.initialize()
        await ensure_schema(self.pool)
        self.session_id = uuid.uuid4().hex[:16]
        async with session_context(self.pool, self.session_id, title=self.session_title):
            pass

    async def _get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.pool is None:
            return []
        return await get_messages(self.pool, self.session_id, limit=limit)

    async def _save_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.pool is None:
            return
        msg_id = uuid.uuid4().hex[:16]
        await insert_message(
            self.pool,
            msg_id,
            self.session_id,
            role,
            content,
            metadata=metadata,
        )

    def _first_cloud_model(self) -> str:
        cfg = get_config()
        for entry in self.router.fallback_chain:
            if not entry.startswith("local:") and "/" in entry:
                return entry
        return cfg.cloud_model

    async def _route_decision(self, prompt: str) -> RoutingDecision:
        if self.force_tier == "cloud":
            cloud_model = self._first_cloud_model()
            score = self.router.complexity(prompt, 0, 0)
            return RoutingDecision(
                model_id=cloud_model,
                tier="cloud",
                complexity_score=score,
                reason="forced cloud mode (/cloud override)",
                fell_back=True,
            )

        history = await self._get_history(limit=20)
        history_turns = sum(
            1 for m in history if m.get("role") in {"user", "assistant"}
        )
        full_text = "\n".join(
            str(m.get("content") or "")
            for m in history
            if m.get("role") in {"user", "assistant"}
        )
        context_tokens = len(full_text) // 4
        return self.router.route(prompt, context_tokens, history_turns)

    def _format_loop_response(
        self,
        loop_result: ToolLoopResult,
        decision: RoutingDecision,
    ) -> str:
        tier_emoji = "⚡" if decision.tier == "local" else "☁️"
        parts: list[str] = []
        if loop_result.tool_names:
            tool_summary = ", ".join(
                f"`{n}`" for n in _dedupe(loop_result.tool_names)
            )
            parts.append(
                f"{tier_emoji} _{decision.tier}_ · "
                f"{loop_result.iterations} turn(s) · tools: {tool_summary}"
            )
        else:
            parts.append(
                f"{tier_emoji} _{decision.tier}_ · "
                f"{loop_result.iterations} turn(s)"
            )
        if loop_result.terminated_by != "complete":
            parts.append(f"_⚠ Loop ended: {loop_result.terminated_by}_")
        parts.append("")
        parts.append(loop_result.response)
        return "\n".join(parts)

    async def _persist_loop_outcome(
        self,
        loop_result: ToolLoopResult,
        decision: RoutingDecision,
        routing_meta: dict[str, Any],
    ) -> None:
        # Persist a tool-name-only marker per invocation. Full tool I/O goes
        # to observability (log_turn) — keeping it out of the conversation
        # history avoids pollution from large tool outputs while still
        # giving downstream tools a "which tools ran this turn" view.
        for i, tn in enumerate(loop_result.tool_names):
            await self._save_message(
                "tool", f"called {tn}", metadata={"tool_index": i}
            )
        routing_meta["tool_loop"] = {
            "iterations": loop_result.iterations,
            "tool_calls_made": loop_result.tool_calls_made,
            "tool_names": loop_result.tool_names,
            "terminated_by": loop_result.terminated_by,
        }
        self.router.record_outcome(
            decision,
            succeeded=(loop_result.terminated_by in ("complete", "repeated_error")),
            tool_calls_malformed=(loop_result.terminated_by == "malformed_tool_call"),
        )
        _obs = loop_result.metadata.get("_obs", {})
        log_turn(
            turn_id=uuid.uuid4().hex[:16],
            tier=decision.tier,
            model=decision.model_id,
            prompt_tokens=_obs.get("prompt_tokens", 0),
            completion_tokens=_obs.get("completion_tokens", 0),
            elapsed_ms=_obs.get("elapsed_ms", 0),
            tools_called=list(_dedupe(loop_result.tool_names)),
            errors=_obs.get("errors", []),
        )

    async def _run_tool_loop_turn(
        self,
        prompt: str,
        decision: RoutingDecision,
        routing_meta: dict[str, Any],
    ) -> tuple[str, bool]:
        """Run ToolLoop and persist outcome.

        Returns ``(response, already_saved)``. When API keys are exhausted,
        the error message is saved immediately and ``already_saved`` is True.
        """
        try:
            loop_result = await self.tool_loop.run(
                messages=await self._build_llm_history(prompt, decision),
                model=decision.model_id,
            )
        except AllKeysExhausted as exc:
            logger.warning(
                "ToolLoop: all API keys exhausted (%s), returning graceful error", exc
            )
            response = (
                f"⚠ All API keys exhausted. "
                f"Earliest key available in {exc.cooldown_remaining:.0f}s. "
                f"Try again later or set HERMES_LITE_LOCAL_MODEL for local fallback."
            )
            await self._save_message(
                "assistant",
                response,
                metadata={
                    "tool_called": False,
                    "routing": {
                        "error": "all_keys_exhausted",
                        "cooldown_remaining": exc.cooldown_remaining,
                    },
                },
            )
            return response, True
        response = self._format_loop_response(loop_result, decision)
        await self._persist_loop_outcome(loop_result, decision, routing_meta)
        return response, False

    async def _handle_prompt(self, prompt: str) -> str:
        """Process a user prompt: route, command/tool/agent path, persist turn.

        Routing is the most expensive single step (history lookup +
        complexity score). Commands and direct-tool shortcuts skip it
        entirely so router state is not polluted by slash commands.
        """
        await self._save_message("user", prompt)

        if is_direct_tool(prompt):
            tool_result = True
            response = dispatch_direct_tool(self, prompt)
            routing_meta: dict[str, Any] = {"path": "direct_tool"}
            await self._save_message(
                "assistant",
                response,
                metadata={"tool_called": True, "routing": routing_meta},
            )
            return response

        if is_command(prompt):
            response = await dispatch_command(self, prompt)
            # /clear wipes history, so it must NOT also persist a phantom
            # assistant turn. Other commands still save the assistant
            # response so the chat log stays coherent.
            cleared = prompt.lower().strip() == "/clear"
            if not cleared:
                await self._save_message(
                    "assistant",
                    response,
                    metadata={"tool_called": False, "routing": {"path": "command"}},
                )
            return response

        # ---- LLM paths below: route + run ----
        decision = await self._route_decision(prompt)
        routing_meta = {
            "model_id": decision.model_id,
            "tier": decision.tier,
            "complexity_score": round(decision.complexity_score, 4),
            "reason": decision.reason,
            "fell_back": decision.fell_back,
        }

        tool_result = None
        already_saved = False

        if self.active_moa_preset is not None:
            try:
                moa_engine = MoAEngine(preset=self.active_moa_preset)
                moa_messages = await self._build_llm_history(prompt, decision)
                moa_result: MoAResult = await moa_engine.run(
                    messages=moa_messages,
                    prompt=prompt,
                )
                response = format_moa_result(moa_result)
                routing_meta["moa"] = {
                    "preset": moa_result.metadata.get("preset"),
                    "refs_ok": moa_result.metadata.get("refs_ok"),
                    "refs_total": moa_result.metadata.get("refs_total"),
                    "agg_model": moa_result.aggregator_model,
                    "terminated_by": moa_result.terminated_by,
                }
                # Use the routing decision that triggered this MoA turn,
                # not a hardcoded "cloud" label — local-tier requests that
                # trigger MoA should still log as their real tier.
                log_turn(
                    turn_id=uuid.uuid4().hex[:16],
                    tier=decision.tier,
                    model=moa_result.aggregator_model,
                    errors=[],
                )
            except AllKeysExhausted as exc:
                logger.warning(
                    "MoA: all API keys exhausted (%s), falling back to ToolLoop", exc
                )
                response, already_saved = await self._run_tool_loop_turn(
                    prompt, decision, routing_meta
                )
        else:
            response, already_saved = await self._run_tool_loop_turn(
                prompt, decision, routing_meta
            )

        if not already_saved:
            await self._save_message(
                "assistant",
                response,
                metadata={
                    "tool_called": tool_result is not None,
                    "routing": routing_meta,
                },
            )

        return response

    async def stream_prompt(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream a single-turn LLM response token-by-token.

        This method routes the prompt, builds the system prompt + memory
        context, and streams the model output via ``chat_stream``.

        **Tool calls are silently dropped during streaming; only text
        deltas are yielded.** This is a direct-answer path for simple
        Q&A. Prompts that require tool calls should use
        ``_handle_prompt`` (batch) instead. The WebUI ``stream`` message
        type uses this; if the LLM emits tool_calls during streaming,
        the generated text up to that point is yielded and the stream
        ends (the client can retry via batch).

        Commands, direct-tool calls, and active MoA presets are delegated
        to ``_handle_prompt`` so they execute correctly.
        """
        # Guard: commands, direct-tool calls, and MoA presets must use
        # the batch path which handles them correctly.
        if is_direct_tool(prompt) or is_command(prompt) or self.active_moa_preset is not None:
            response = await self._handle_prompt(prompt)
            yield response
            return

        # Persist user message before streaming so history is complete.
        await self._save_message("user", prompt)

        decision = await self._route_decision(prompt)
        history = await self._build_llm_history(prompt, decision)
        req = ChatRequest(
            messages=history,
            model=decision.model_id,
            temperature=0.2,
            max_tokens=2048,
        )
        collected = ""
        success = False
        try:
            async for token in chat_stream(req):
                if token:
                    collected += token
                    yield token
            success = True
        except Exception:
            success = False
            raise
        finally:
            # Persist the full response only on clean completion.
            # Partial responses from failed streams are NOT persisted
            # to avoid polluting conversation history.
            if collected and success:
                await self._save_message(
                    "assistant",
                    collected,
                    metadata={
                        "tool_called": False,
                        "routing": {
                            "model_id": decision.model_id,
                            "tier": decision.tier,
                            "complexity_score": round(decision.complexity_score, 4),
                            "reason": decision.reason,
                            "fell_back": decision.fell_back,
                        },
                    },
                )
            self.router.record_outcome(
                decision,
                succeeded=success,
                tool_calls_malformed=False,
            )

    async def _build_llm_history(
        self,
        current_prompt: str,
        decision: RoutingDecision | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages from memory + current prompt."""
        cfg = get_config()
        memory_block = ""
        try:
            from hermes_lite.memory_bridge import get_default_bridge

            memory_block = get_default_bridge().load_into_prompt(
                max_chars=cfg.memory_max_chars
            )
        except Exception:
            memory_block = ""

        tool_extra = ""
        is_local_turn = (
            decision is not None and decision.tier == "local"
        ) or (
            decision is None and self.force_tier != "cloud"
        )
        if is_local_turn:
            tool_descs = self.get_tool_descriptions()
            if tool_descs:
                tool_lines = [
                    "Call a tool by outputting a JSON object with \"name\" and \"arguments\" keys.",
                    'Example: {"name": "web_search", "arguments": {"query": "hello world"}}\n',
                ]
                for d in tool_descs:
                    name = d.get("name", "")
                    desc = d.get("description", "")
                    params = d.get("parameters", {}).get("properties", {})
                    req = d.get("parameters", {}).get("required", [])
                    param_strs = []
                    for pname, pinfo in params.items():
                        req_mark = " (required)" if pname in req else ""
                        pdesc = pinfo.get("description", "")[:60]
                        param_strs.append(f'    "{pname}": {pdesc}{req_mark}')
                    tool_lines.append(f"- **{name}**: {desc}")
                    if param_strs:
                        tool_lines.append("  Arguments:")
                        tool_lines.extend(param_strs)
                tool_extra = "\n".join(tool_lines)

        extras = []
        if tool_extra:
            extras.append(tool_extra)
        if memory_block:
            extras.append(memory_block)
        extra = "\n\n".join(extras) if extras else None

        try:
            system_content = build_system_prompt(extra=extra)
        except Exception:
            # Fall back if prompt assets missing
            system_content = (
                "You are Hermes-Lite, a helpful AI assistant with access to tools. "
                "Use tools when they help answer the user's question. "
                "When you have enough information to answer directly, respond in plain "
                "text without tool calls."
            )
            if extra:
                system_content = system_content + "\n\n" + extra

        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        db_history = await self._get_history(limit=16)
        for m in db_history:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant"):
                msgs.append({"role": role, "content": content})

        safe_prompt = scrub_control_tokens(current_prompt)
        msgs.append({"role": "user", "content": safe_prompt})
        return msgs

    def start(self) -> None:
        """Launch the CLI with the orchestrator wired in as the prompt handler."""
        self._create_default_tools()
        import asyncio as _aio

        async def _setup_and_run() -> None:
            await self._initialize_memory()
            run_cli(
                on_prompt=self._handle_prompt,
                welcome_message=(
                    f"Hermes-Lite v{__version__} — Orchestrator Engine\n\n"
                    f"Session: {self.session_id[:8]}...\n"
                    f"Tools: {self.registry.tool_count} registered\n"
                    f"DB: {self.db_path}"
                ),
            )

        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(_aio.run, _setup_and_run()).result()
        else:
            _aio.run(_setup_and_run())


__all__ = [
    "HermesOrchestrator",
    "ToolLoop",
    "ToolLoopResult",
    "__version__",
]
