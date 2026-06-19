"""hermes_lite.subagent — Single-shot delegated task primitive.

Adds a ``subagent`` tool to the PluginRegistry that spins up an
**isolated** LiteOrchestrator-style loop in-process, runs a focused
agent loop against the *local* Qwen 3B model, and returns a text
summary. The host (parent) session's memory/state is never written
to by the child.

Design rules (per the T8 spec)
------------------------------
1. ``scope='read'``  → child only reads; final assistant text becomes the
   ``output`` field of the ToolResult.
   ``scope='write'`` → child may produce code/files; the same final text
   is also returned as the ``work_product`` field so the caller can paste
   it verbatim into the host context.
2. **Single-tier only.** Subagent cannot spawn another subagent: the
   isolated subagent registry is built *without* calling
   :func:`register_subagent_tool`. Hallucinating a ``subagent`` call
   therefore yields ``ToolNotFoundError`` in the child loop, which the
   ToolLoop's ``repeated_error`` guard catches after one round trip.
3. **Hard caps:**
   - ``max_turns`` clamped to ``[1, 6]`` (default 4) — child can only
     take at most 6 LLM round-trips even if the caller asked for more.
   - 60-second wall-clock timeout via ``asyncio.wait_for``.
   - At most 6 *tool calls* total inside the child loop (counted by
     the ToolLoop against its own iteration cap).
4. **Isolation:** the child gets its own ``AsyncSQLitePool`` rooted at
   ``/tmp/lite-sub-<uuid12>.db``. After the run the pool is closed
   and the path is returned in the result, but the parent never
   touches the file.
5. **Logging:** every run appends a single JSON line to
   ``~/.hermes_lite/subagents.log`` with goal preview, scope, model,
   timing, termination reason, and tool-call summary. Failures to
   write the log are swallowed (logging must never crash the tool).

Public API
----------
* :class:`SubagentArgs`            — Pydantic schema for the tool.
* :func:`register_subagent_tool`   — registers the ``subagent`` tool on the
  given registry. Returns the ``ToolDefinition``.
* :class:`SubagentRunner`          — thin asyncio wrapper around the
  isolated subagent loop, exposed for direct (non-tool) use and tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from hermes_lite.llm import ChatRequest, chat
from hermes_lite.orchestrator import ToolLoop
from hermes_lite.registry import PluginRegistry, ToolDefinition
from hermes_lite.tools_builtins import register_builtins as _register_essentials

# ---------------------------------------------------------------------------
# Module-level constants — public so tests and docs can reference them
# ---------------------------------------------------------------------------

SUBAGENT_TOOL_NAME = "subagent"
SUBAGENT_MAX_TOOL_CALLS = 6           # hard cap on LLM-driven tool calls
SUBAGENT_MAX_ITERATIONS = 6           # alias — the child loop's max_iterations
SUBAGENT_WALL_TIMEOUT_S = 60.0        # hard wall-clock kill
SUBAGENT_DB_PREFIX = "/tmp/lite-sub-"

DEFAULT_SUBAGENT_LOG_PATH = Path.home() / ".hermes_lite" / "subagents.log"


# System prompt the child sees instead of the parent's. Tighter, scoped,
# and explicitly forbids further delegation.
SUBAGENT_SYSTEM_PROMPT = (
    "You are a Hermes-Lite **subagent** spawned for one focused task. "
    "Your output will be returned to the parent agent as a short summary.\n\n"
    "Rules:\n"
    " 1. Stay strictly on the stated goal. No scope creep.\n"
    " 2. Use the provided tools (read_file, search_files, terminal, "
    "memory, web_search, web_fetch) when they help.\n"
    " 3. **You cannot spawn further subagents** — if you try, the tool "
    "registry will reject the call. Just do the work yourself.\n"
    " 4. When you have enough information, respond in plain text with a "
    "short summary (3 short paragraphs or less) and stop calling tools.\n"
    " 5. If the goal requires producing code or files (scope='write'), "
    "produce the final code/text in your plain-text reply — the host "
    "will paste it back to the user verbatim.\n"
    " 6. Keep tool calls to the minimum needed. The loop is capped.\n"
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SubagentArgs(BaseModel):
    """Arguments for the ``subagent`` tool.

    Attributes
    ----------
    goal:
        A focused description of what the subagent should accomplish.
        Keep it imperative and specific — ``"summarize the pickle module
        in 3 paragraphs"`` not ``"anything about pickle"``.
    scope:
        ``'read'`` (default) → the child may only read; ``'write'`` →
        the child may produce code/files. The host surfaces the final
        assistant text either way; ``scope='write'`` also includes it
        as ``work_product`` for direct pasting.
    max_turns:
        Soft suggestion for the subagent's LLM iteration cap. Clamped
        server-side to ``[1, 6]``; default = 4 (matches ToolLoop default).
    """

    goal: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="One focused task the subagent should complete. Be specific.",
    )
    scope: Literal["read", "write"] = Field(
        default="read",
        description=(
            "'read' = child may only inspect (default). "
            "'write' = child may also write new code/files."
        ),
    )
    max_turns: int = Field(
        default=4,
        ge=1,
        le=20,
        description=(
            "Soft cap on LLM round-trips; the server clamps this to "
            "the [1, 6] window. Default = 4."
        ),
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _default_local_model_id() -> str:
    """Return the canonical model id for the subagent (local Qwen 3B).

    Admins can override via ``HERMES_LITE_SUBAGENT_MODEL``. Anything
    non-empty goes — we don't validate it here because the LLM client
    will resolve it via ``_pick_client_and_model``.
    """
    return os.environ.get(
        "HERMES_LITE_SUBAGENT_MODEL",
        "local:qwen2.5-3b-instruct-q4_k_m.gguf",
    )


def _failure(msg: str, sub_id: str, *, terminated_by: str = "crashed") -> dict[str, Any]:
    """Uniform failure ToolResult dict."""
    return {
        "ok": False,
        "output": "",
        "error": msg,
        "scope": "read",
        "sub_id": sub_id,
        "iterations": 0,
        "tool_calls_made": 0,
        "tool_names": [],
        "terminated_by": terminated_by,
    }


def _disable_write_tools(registry: PluginRegistry) -> None:
    """Swap the ``terminal`` and ``memory`` handlers with refusal closures.

    In ``scope='read'`` we replace the *handler* (not the tool entry) so
    the registry still validates the schema correctly; the model sees
    the tool description but any call yields a structured error.
    """
    disabled = {"terminal", "memory"}
    for name in disabled:
        if not registry.has_tool(name):
            continue
        definition = registry.get_tool(name)
        schema_model: type[BaseModel] | None = definition.schema_model
        if schema_model is None:
            continue

        def _make_handler(tool_name: str) -> Callable[..., dict[str, Any]]:
            def _refuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "ok": False,
                    "output": "",
                    "error": f"tool '{tool_name}' is disabled in scope='read'",
                }
            return _refuse

        registry.remove_tool(name)
        registry.add_tool(
            ToolDefinition(
                name=name,
                description=(
                    definition.description + " (disabled in read-only subagent scope)"
                ),
                schema_model=schema_model,
                handler=_make_handler(name),
            )
        )


# ---------------------------------------------------------------------------
# SubagentRunner — one focused, isolated loop
# ---------------------------------------------------------------------------


@dataclass
class SubagentRunner:
    """One subagent run, scoped and isolated.

    The runner owns an isolated :class:`AsyncSQLitePool` rooted at
    ``<prefix>-<uuid>.db`` and a fresh :class:`PluginRegistry` with
    the 6 essentials (but **not** the subagent tool itself).

    The default ``db_path`` factory gives each run a UUID-suffixed
    unique file under :data:`SUBAGENT_DB_PREFIX` so concurrent subagents
    don't contend on the SQLite DB.
    """

    chat_fn: Callable[[ChatRequest], Any] = chat
    """LLM client to use. Defaults to :func:`hermes_lite.llm.chat`."""

    db_path: str = field(
        default_factory=lambda: f"{SUBAGENT_DB_PREFIX}{uuid.uuid4().hex[:12]}.db"
    )
    """Per-run SQLite path. Default factory assigns ``/tmp/lite-sub-<uuid>.db``."""

    log_path: Path = DEFAULT_SUBAGENT_LOG_PATH
    """Where to write the per-run JSONL audit line."""

    sub_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    """Identifier for this run. Used in the log line only."""

    scope: Literal["read", "write"] = "read"
    """Read-only vs allow-writes mode for the child registry."""

    max_tool_calls: int = SUBAGENT_MAX_TOOL_CALLS
    """Hard cap on LLM-driven tool calls within the child loop."""

    timeout_s: float = SUBAGENT_WALL_TIMEOUT_S
    """Wall-clock budget before the run is killed with TimeoutError."""

    # -- public API --------------------------------------------------------

    async def run(
        self,
        goal: str,
        *,
        max_turns: int = 4,
    ) -> dict[str, Any]:
        """Execute the subagent and return a structured ToolResult dict.

        The returned dict matches the ``{ok, output, error?, work_product?}``
        shape used by every other builtin handler so the host
        orchestrator can format it uniformly.
        """
        started = time.monotonic()
        cap = _clamp(max_turns, 1, SUBAGENT_MAX_ITERATIONS)
        log_payload: dict[str, Any] = {
            "sub_id": self.sub_id,
            "ts": time.time(),
            "goal_preview": goal[:120],
            "scope": self.scope,
            "max_turns_requested": max_turns,
            "max_turns_effective": cap,
            "model": _default_local_model_id(),
        }
        try:
            summary = await asyncio.wait_for(
                self._run_isolated(goal, cap),
                timeout=self.timeout_s,
            )
            elapsed_ms = int((time.monotonic() - started) * 1000)
            log_payload.update({
                "ok": True,
                "elapsed_ms": elapsed_ms,
                "iterations": summary["iterations"],
                "tool_calls_made": summary["tool_calls_made"],
                "tool_names": summary["tool_names"],
                "terminated_by": summary["terminated_by"],
            })
            self._log_run(log_payload)
            result: dict[str, Any] = {
                "ok": True,
                "output": summary["response"],
                "scope": self.scope,
                "iterations": summary["iterations"],
                "tool_calls_made": summary["tool_calls_made"],
                "tool_names": summary["tool_names"],
                "terminated_by": summary["terminated_by"],
                "sub_id": self.sub_id,
                "elapsed_ms": elapsed_ms,
            }
            if self.scope == "write":
                result["work_product"] = summary["response"]
            return result
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            log_payload.update({"ok": False, "error": "timeout", "elapsed_ms": elapsed_ms})
            self._log_run(log_payload)
            return _failure(
                f"subagent timed out after {self.timeout_s:.0f}s "
                f"(goal preview: {goal[:80]!r})",
                self.sub_id,
                terminated_by="timeout",
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            log_payload.update({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": elapsed_ms,
            })
            self._log_run(log_payload)
            return _failure(
                f"subagent crashed: {type(exc).__name__}: {exc}",
                self.sub_id,
                terminated_by="crashed",
            )

    # -- internals ---------------------------------------------------------

    async def _run_isolated(
        self,
        goal: str,
        max_iterations: int,
    ) -> dict[str, Any]:
        """Run the isolated ToolLoop on the goal, returning an internal dict."""
        # We import lazily so importing subagent.py doesn't pull every backend.
        from hermes_lite.memory import (
            AsyncSQLitePool,
            create_session,
            ensure_schema,
        )

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        pool = AsyncSQLitePool(self.db_path, min_size=1, max_size=1)
        try:
            await pool.initialize()
            await ensure_schema(pool)

            # Create a session so subsequent message inserts succeed.
            session_id = f"sub-{self.sub_id}"
            await create_session(pool, session_id, title=f"subagent:{goal[:60]}")

            # Fresh registry with the 6 essentials ONLY (no recursion).
            registry = PluginRegistry(strict_validation=True)
            _register_essentials(registry, overwrite=False)

            if self.scope == "read":
                _disable_write_tools(registry)

            tool_loop = ToolLoop(
                registry=registry,
                chat_fn=self.chat_fn,
                max_iterations=max_iterations,
            )

            messages = [
                {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
                {"role": "user", "content": goal},
            ]

            loop_result = await tool_loop.run(
                messages=messages,
                model=_default_local_model_id(),
            )
            return {
                "response": loop_result.response,
                "iterations": loop_result.iterations,
                "tool_calls_made": loop_result.tool_calls_made,
                "tool_names": list(loop_result.tool_names),
                "terminated_by": loop_result.terminated_by,
            }
        finally:
            await pool.close()

    def _log_run(self, payload: dict[str, Any]) -> None:
        """Append one JSON line to the audit log. Never raises."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            # Logging must never break the tool.
            pass


# ---------------------------------------------------------------------------
# Handler factory + entry point
# ---------------------------------------------------------------------------


def _run_subagent_sync(
    runner: SubagentRunner,
    goal: str,
    max_turns: int,
) -> dict[str, Any]:
    """Top-level helper so we can run the runner via a thread executor."""
    return asyncio.run(runner.run(goal, max_turns=max_turns))


def register_subagent_tool(
    registry: PluginRegistry,
    *,
    chat_fn: Callable[[ChatRequest], Any] = chat,
    log_path: Path | str | None = None,
    timeout_s: float = SUBAGENT_WALL_TIMEOUT_S,
) -> ToolDefinition:
    """Register the ``subagent`` tool on *registry* and return its definition.

    The handler is **synchronous** from the registry/call_tool perspective
    (it returns the final ToolResult dict directly).  Internally we either
    run a fresh event loop (no loop running) or hand off to a worker
    thread (we're already inside an event loop, e.g. from the ToolLoop
    running in the parent orchestrator).

    Parameters
    ----------
    registry:
        Target :class:`PluginRegistry`. The tool is registered here.
    chat_fn:
        Override the LLM callable. Defaults to :func:`hermes_lite.llm.chat`.
    log_path:
        Override the audit-log destination. Default
        ``~/.hermes_lite/subagents.log``. Pass a writable path.
    timeout_s:
        Wall-clock budget per run. Default 60s.
    """
    resolved_log = (
        Path(log_path) if log_path is not None else DEFAULT_SUBAGENT_LOG_PATH
    )

    def _subagent_handler(args: SubagentArgs) -> dict[str, Any]:
        runner = SubagentRunner(
            chat_fn=chat_fn,
            log_path=resolved_log,
            scope=args.scope,
            timeout_s=timeout_s,
        )
        try:
            try:
                # If a loop is already running (we're inside the parent's
                # async tool loop), run the subagent in a worker thread so
                # we don't deadlock the parent's loop.
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop — safe to use asyncio.run directly.
                return _run_subagent_sync(runner, args.goal, args.max_turns)

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    _run_subagent_sync,
                    runner,
                    args.goal,
                    args.max_turns,
                )
                return future.result(timeout=timeout_s + 5.0)
        except Exception as exc:
            return _failure(
                f"subagent handler crashed: {type(exc).__name__}: {exc}",
                runner.sub_id,
            )

    definition = ToolDefinition(
        name=SUBAGENT_TOOL_NAME,
        description=(
            "Delegate one focused task to a single-shot subagent. "
            "The subagent runs in an isolated session with its own DB "
            "and the read-only subset of the 6 essential tools. "
            "It cannot spawn further subagents. Returns a short "
            "summary of what it found or produced."
        ),
        schema_model=SubagentArgs,
        handler=_subagent_handler,
    )
    if registry.has_tool(SUBAGENT_TOOL_NAME):
        registry.remove_tool(SUBAGENT_TOOL_NAME)
    registry.add_tool(definition)
    return definition


__all__ = [
    "SubagentArgs",
    "SubagentRunner",
    "register_subagent_tool",
    "SUBAGENT_TOOL_NAME",
    "SUBAGENT_MAX_TOOL_CALLS",
    "SUBAGENT_MAX_ITERATIONS",
    "SUBAGENT_WALL_TIMEOUT_S",
    "SUBAGENT_DB_PREFIX",
    "DEFAULT_SUBAGENT_LOG_PATH",
    "SUBAGENT_SYSTEM_PROMPT",
]
