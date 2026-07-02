"""
hermes_lite/orchestrator.py — Orchestrator Engine

Wires the PluginRegistry (tools), SQLite memory layer (conversation history),
LLM client, and CLI shell into a working agent loop.

Key components:
- ToolLoop: Two-tier tool-calling loop with termination guards
- HermesOrchestrator: Coordinates tool registry + memory + prompt handling
- register_builtins(): Populates the registry with default built-in tools
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from hermes_lite.registry import (
    PluginRegistry,
    ToolDefinition,
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
    ToolAuthError,
)
from hermes_lite.memory import (
    AsyncSQLitePool,
    ensure_schema,
    session_context,
    insert_message,
    get_messages,
    set_metadata,
    get_metadata,
    create_session,
    get_session,
)
from hermes_lite.llm import ChatRequest, ChatResponse, chat, tool_def, Tier, AllKeysExhausted
from hermes_lite.cli import run_cli
from hermes_lite.router import LiteRouter, RoutingDecision
from hermes_lite.tools_builtins import register_builtins as _register_essentials
from hermes_lite.observability import log_turn
from hermes_lite.moa import (
    MoAEngine,
    MoAPreset,
    MoAResult,
    BUILTIN_PRESETS,
    get_preset,
    list_presets as list_moa_presets,
    format_preset_info,
    format_moa_result,
)

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("hermes-lite")
except Exception:
    __version__ = "0.5.0"

from hermes_lite.sanitize import sanitize_tool_args, scrub_control_tokens

# ---------------------------------------------------------------------------
# Built-in tools are now provided by :mod:`hermes_lite.tools_builtins`.
# The 6 essentials (``read_file``, ``search_files``, ``terminal``,
# ``memory``, ``web_search``, ``web_fetch``) are the only tools the
# orchestrator registers by default. The previous ``echo`` /
# ``calculator`` / ``save_note`` set has been retired.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool Loop — the two-tier tool-calling loop (T6)
# ---------------------------------------------------------------------------


@dataclass
class ToolLoopResult:
    """Outcome of a ToolLoop run — fed back to the orchestrator."""

    response: str
    """Final text to show the user."""

    iterations: int
    """How many LLM round-trips the loop took."""

    tool_calls_made: int
    """Total tool invocations across all iterations."""

    tool_names: list[str]
    """Names of every tool called, in order."""

    terminated_by: str
    """Why the loop stopped: 'complete', 'max_iterations', 'repeated_error', 'malformed_tool_call'."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra structured data for persistence (routing, model, etc.)."""


class ToolLoop:
    """Two-tier tool-calling loop with termination guards.

    The loop drives an LLM through multiple rounds of tool invocation:
    1. Send messages + tool defs to the LLM
    2. If the LLM responds with ``tool_calls``, validate + dispatch each one
    3. Append tool results to the conversation history
    4. Re-invoke the LLM with the updated history
    5. Repeat until the LLM returns a plain text response, or a termination
       condition fires.

    Termination conditions:
    - **complete**: LLM returned a response with no ``tool_calls``
    - **max_iterations**: exceeded ``max_iterations`` (default 4)
    - **repeated_error**: the same tool produced the same error on 2
      consecutive iterations
    - **malformed_tool_call**: the LLM produced a tool call with invalid
      JSON arguments, and a single retry nudge also produced bad JSON

    Parameters
    ----------
    registry:
        Tool registry for validation and dispatch.
    chat_fn:
        Async callable ``(ChatRequest) -> ChatResponse`` — normally
        :func:`hermes_lite.llm.chat`, but injectable for testing.
    max_iterations:
        Hard cap on LLM round-trips. Default 4.
    on_tool_call:
        Optional callback invoked each time a tool is about to execute.
        Receives ``(tool_name, first_arg_value)`` so the caller can print
        a ⚡ progress line to the terminal.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        chat_fn: Callable[[ChatRequest], Any] = chat,
        max_iterations: int = 4,
        on_tool_call: Callable[[str, str], None] | None = None,
    ) -> None:
        self.registry = registry
        self.chat_fn = chat_fn
        self.max_iterations = max_iterations
        self.on_tool_call = on_tool_call

    # -- public entry -----------------------------------------------------

    async def run(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ToolLoopResult:
        """Execute the tool-calling loop.

        Parameters
        ----------
        messages:
            Conversation history so far (will be copied; not mutated in-place).
        model:
            Model identifier accepted by :func:`hermes_lite.llm.chat`.
        temperature:
            Sampling temperature for every LLM call in the loop.
        max_tokens:
            Token budget per LLM call.

        Returns
        -------
        ToolLoopResult
            Summary of the loop outcome for the orchestrator to persist
            and display.
        """
        history = list(messages)
        tools = self._build_openai_tools()
        iterations = 0
        total_tool_calls = 0
        all_tool_names: list[str] = []
        last_error_key: tuple[str, str] | None = None
        # Track whether we've already issued a JSON-nudge retry.
        nudged_json = False
        # Observability accumulators
        turn_start_ms = int(time.monotonic() * 1000)
        cumulative_prompt_tokens = 0
        cumulative_completion_tokens = 0
        turn_errors: list[str] = []

        while iterations < self.max_iterations:
            iterations += 1
            req = ChatRequest(
                messages=history,
                model=model,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp: ChatResponse = await self.chat_fn(req)

            # Accumulate token usage (observability T12)
            cumulative_prompt_tokens += resp.usage.get("prompt_tokens", 0)
            cumulative_completion_tokens += resp.usage.get("completion_tokens", 0)

            # -- No tool calls → LLM is done talking ---------------
            if not resp.tool_calls:
                elapsed_ms = int((time.monotonic() * 1000) - turn_start_ms)
                return ToolLoopResult(
                    response=resp.content or "",
                    iterations=iterations,
                    tool_calls_made=total_tool_calls,
                    tool_names=all_tool_names,
                    terminated_by="complete",
                    metadata={
                        "model": model,
                        "tier": resp.tier,
                        "_obs": {
                            "elapsed_ms": elapsed_ms,
                            "prompt_tokens": cumulative_prompt_tokens,
                            "completion_tokens": cumulative_completion_tokens,
                            "errors": turn_errors,
                        },
                    },
                )

            # -- Process each tool call ----------------------------
            # Append the assistant's tool-calling message to history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.content or None,
            }
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in resp.tool_calls
                ]
            history.append(assistant_msg)

            any_malformed = False

            for tc in resp.tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
                tc_args_str = tc.get("arguments", "{}")
                total_tool_calls += 1
                all_tool_names.append(tc_name)

                # Extract first arg value for the streaming callback
                first_arg = self._extract_first_arg(tc_args_str)

                # -- Validate + dispatch ---------------------------
                result_content: str
                try:
                    args = json.loads(tc_args_str)
                except json.JSONDecodeError:
                    # Malformed JSON — nudge once, then break
                    any_malformed = True
                    if not nudged_json:
                        nudged_json = True
                        result_content = (
                            "Error: your tool call arguments were not valid JSON. "
                            "Please respond with strict JSON arguments only — "
                            "no comments, no trailing commas."
                        )
                    else:
                        # Second malformed call — terminate
                        result_content = (
                            "Error: repeated malformed JSON in tool call arguments. "
                            "Stopping."
                        )
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result_content,
                        })
                        elapsed_ms = int((time.monotonic() * 1000) - turn_start_ms)
                        turn_errors.append("malformed JSON in tool call")
                        return ToolLoopResult(
                            response="I encountered repeated errors formatting tool calls. Please try rephrasing your request.",
                            iterations=iterations,
                            tool_calls_made=total_tool_calls,
                            tool_names=all_tool_names,
                            terminated_by="malformed_tool_call",
                            metadata={
                                "model": model,
                                "tier": resp.tier,
                                "_obs": {
                                    "elapsed_ms": elapsed_ms,
                                    "prompt_tokens": cumulative_prompt_tokens,
                                    "completion_tokens": cumulative_completion_tokens,
                                    "errors": turn_errors,
                                },
                            },
                        )
                else:
                    # Valid JSON — validate against registry and dispatch
                    # Fire the streaming callback
                    if self.on_tool_call:
                        self.on_tool_call(tc_name, first_arg)

                    try:
                        # Sanitize tool arguments before dispatch
                        sanitized = sanitize_tool_args(tool_name=tc_name, args=args)
                        if not sanitized.is_clean:
                            issues = "; ".join(sanitized.issues)
                            self.logger.warning(f"[SANITIZE] '{tc_name}' blocked: {issues}")
                            result_content = f"Tool call blocked: security policy violation — {issues}"
                            turn_errors.append(f"{tc_name}: {issues}")
                            tool_response_data = {
                                "role": "tool_response",
                                "name": tc_name,
                                "output": result_content,
                            }
                            continue

                        raw = self.registry.call_tool(tc_name, sanitized.args, auth_token=self.registry.auth_token)
                        if isinstance(raw, dict) and raw.get("ok"):
                            result_content = raw.get("output", "")
                        elif isinstance(raw, dict) and not raw.get("ok"):
                            err = raw.get("error", "unknown error")
                            result_content = f"Tool error: {err}"
                        else:
                            result_content = str(raw)
                    except ToolNotFoundError:
                        result_content = f"Tool error: '{tc_name}' is not a registered tool."
                        turn_errors.append(f"{tc_name}: not registered")
                    except ToolValidationError as exc:
                        result_content = f"Tool error: validation failed — {exc}"
                        turn_errors.append(f"{tc_name}: validation failed")
                    except ToolError as exc:
                        result_content = f"Tool error: {exc}"
                        turn_errors.append(f"{tc_name}: {exc}")
                    except Exception as exc:
                        result_content = f"Tool error: unexpected {type(exc).__name__}: {exc}"
                        turn_errors.append(f"{tc_name}: {type(exc).__name__}")

                    # Repeated-error detection: (tool_name, error_prefix)
                    # Surface the same tool+error twice → break
                    if result_content.startswith("Tool error:"):
                        error_key = (tc_name, result_content[:80])
                        if error_key == last_error_key:
                            history.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": result_content,
                            })
                            elapsed_ms = int((time.monotonic() * 1000) - turn_start_ms)
                            turn_errors.append(f"{tc_name}: repeated error")
                            return ToolLoopResult(
                                response=f"I hit a repeated error with the `{tc_name}` tool and stopped. Last error: {result_content}",
                                iterations=iterations,
                                tool_calls_made=total_tool_calls,
                                tool_names=all_tool_names,
                                terminated_by="repeated_error",
                                metadata={
                                    "model": model,
                                    "tier": resp.tier,
                                    "repeated_error_tool": tc_name,
                                    "_obs": {
                                        "elapsed_ms": elapsed_ms,
                                        "prompt_tokens": cumulative_prompt_tokens,
                                        "completion_tokens": cumulative_completion_tokens,
                                        "errors": turn_errors,
                                    },
                                },
                            )
                        last_error_key = error_key
                    else:
                        last_error_key = None

                # Append tool result to history
                history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_content,
                })

            # If any tool call had malformed JSON but we didn't break
            # (first nudge), continue the loop so the LLM can retry.
            if any_malformed:
                continue

        # -- max_iterations exhausted ----------------------------
        elapsed_ms = int((time.monotonic() * 1000) - turn_start_ms)
        return ToolLoopResult(
            response=(
                f"I reached the maximum number of reasoning steps ({self.max_iterations}) "
                f"and stopped. Here is what I have so far:\n\n"
                + "[no partial response]"
            ),
            iterations=iterations,
            tool_calls_made=total_tool_calls,
            tool_names=all_tool_names,
            terminated_by="max_iterations",
            metadata={
                "model": model,
                "tier": "local",
                "_obs": {
                    "elapsed_ms": elapsed_ms,
                    "prompt_tokens": cumulative_prompt_tokens,
                    "completion_tokens": cumulative_completion_tokens,
                    "errors": turn_errors,
                },
            },
        )

    # -- helpers ----------------------------------------------------------

    def _build_openai_tools(self) -> list[dict[str, Any]]:
        """Build the ``tools`` payload for the OpenAI chat API from the registry."""
        result = []
        for desc in self.registry.tool_descriptions():
            result.append(tool_def(
                name=desc["name"],
                description=desc["description"],
                parameters=desc.get("parameters", {"type": "object", "properties": {}}),
            ))
        return result

    @staticmethod
    def _extract_first_arg(args_str: str) -> str:
        """Try to pull the first argument value out of a JSON args string.

        Used for the streaming callback so the user sees something like
        ``⚡ read_file("README.md")``.
        """
        try:
            obj = json.loads(args_str)
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str):
                        return v
                    return str(v)
        except (json.JSONDecodeError, ValueError):
            pass
        # Best-effort: return the raw string, truncated
        return args_str[:60]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


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
        self.db_path = str(Path(db_path).expanduser())
        self.session_title = session_title
        self.session_id: str = ""
        self.pool: AsyncSQLitePool | None = None
        # Read auth token from env if not provided
        if auth_token is None:
            auth_token = os.environ.get("HERMES_LITE_AUTH_TOKEN")
        self.auth_token = auth_token
        self.registry = PluginRegistry(strict_validation=True, auth_token=auth_token)
        # Routing controller picks local vs cloud before any LLM call.
        # None means "construct a default LiteRouter inside handle_prompt"
        # so existing tests that don't care about routing keep working,
        # but consumers who want to override thresholds/fallback chain
        # can pass their own.
        self.router: LiteRouter = router or LiteRouter()
        # ToolLoop — set up lazily in start() after tools are registered.
        # Accept an override for testing.
        self._tool_loop: ToolLoop | None = tool_loop
        # MoA (Mixture-of-Agents) — None means inactive (default).
        # Set via /moa <preset> command or HERMES_LITE_MOA_PRESET env var.
        self.active_moa_preset: MoAPreset | None = None

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Logger for this instance
        self.logger = logging.getLogger(__name__)

    def _create_default_tools(self) -> None:
        """Register the 6 essential tools via hermes_lite.tools_builtins,
        then add the subagent tool (single-shot delegated LLM calls).

        The subagent import is deferred to break a circular dependency:
        ``subagent`` imports ``ToolLoop`` from this module at load time.
        """
        _register_essentials(self.registry)
        from hermes_lite.subagent import register_subagent_tool
        register_subagent_tool(self.registry)

    @property
    def tool_loop(self) -> ToolLoop:
        """Lazy-accessor: builds the ToolLoop on first use if not injected."""
        if self._tool_loop is None:
            self._tool_loop = ToolLoop(
                registry=self.registry,
                on_tool_call=self._print_tool_progress,
            )
        return self._tool_loop

    @staticmethod
    def _print_tool_progress(tool_name: str, first_arg: str) -> None:
        """Print a bolt-icon progress line to stderr for streaming."""
        preview = first_arg[:50] if first_arg else ""
        if preview:
            print(f"⚡ {tool_name}({preview!r})", file=sys.stderr)
        else:
            print(f"⚡ {tool_name}(...)", file=sys.stderr)

    def get_tool_descriptions(self) -> list[dict[str, Any]]:
        """Return tool descriptions formatted for the prompt."""
        return self.registry.tool_descriptions()

    async def _initialize_memory(self) -> None:
        """Set up the SQLite pool and create/load the session."""
        self.pool = AsyncSQLitePool(
            self.db_path,
            min_size=1,
            max_size=2,
        )
        await self.pool.initialize()
        await ensure_schema(self.pool)

        # Create a unique session ID and persist it
        self.session_id = uuid.uuid4().hex[:16]

        async with session_context(self.pool, self.session_id, title=self.session_title):
            pass

    async def _get_history(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Retrieve recent conversation history for this session."""
        if self.pool is None:
            return []
        return await get_messages(self.pool, self.session_id, limit=limit)

    async def _save_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save a message to the conversation history."""
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

    async def _route_decision(self, prompt: str) -> RoutingDecision:
        """Compute a routing decision for ``prompt``.

        Counts the user's recent history as ``history_turns`` and a
        rough prompt-length-based token estimate for ``context_tokens``
        so the router has realistic signals to work with. The decision
        is purely advisory here — the orchestrator records it on the
        assistant message so users (and downstream tests) can see the
        chosen tier, but no real LLM call is made yet (the LLM wiring
        lives in T4/T6).
        """
        history = await self._get_history(limit=20)
        history_turns = sum(1 for m in history if m.get("role") in {"user", "assistant"})
        # Coarse context token estimate: ~4 chars/token. Cheap, good
        # enough to drive the router.
        full_text = "\n".join(
            str(m.get("content") or "")
            for m in history
            if m.get("role") in {"user", "assistant"}
        )
        context_tokens = len(full_text) // 4
        return self.router.route(prompt, context_tokens, history_turns)

    async def _handle_prompt(self, prompt: str) -> str:
        """Process a user prompt: route, run tool loop, persist turn.

        Flow:
        1. Save the user message to memory
        2. Route to local or cloud tier
        3. Handle special commands (!tool, /tools, /history, /help)
        4. For general prompts, build conversation history and run the
           ToolLoop against the LLM
        5. Persist the assistant response
        6. Feed the outcome back into the router for escalation tracking
        """
        # Save user message
        await self._save_message("user", prompt)

        # Decide which tier/model this prompt should run against.
        decision = await self._route_decision(prompt)
        routing_meta = {
            "model_id": decision.model_id,
            "tier": decision.tier,
            "complexity_score": round(decision.complexity_score, 4),
            "reason": decision.reason,
            "fell_back": decision.fell_back,
        }

        # Check for direct tool invocation (tool_call prefix)
        tool_result = None
        if prompt.startswith("!"):
            # Direct tool call syntax: !tool_name {"arg": "value"}
            try:
                space_idx = prompt.find(" ")
                if space_idx > 1:
                    tool_name = prompt[1:space_idx].strip()
                    args_str = prompt[space_idx + 1:].strip()
                    args = json.loads(args_str) if args_str else {}
                else:
                    tool_name = prompt[1:].strip()
                    args = {}

                # Sanitize tool arguments before dispatch
                sanitized = sanitize_tool_args(tool_name=tool_name, args=args)
                if not sanitized.is_clean:
                    issues = "; ".join(sanitized.issues)
                    self.logger.warning(f"[SANITIZE] '{tool_name}' blocked: {issues}")
                    response = f"Tool call blocked: security policy violation — {issues}"
                else:
                    tool_result = self.registry.call_tool(tool_name, sanitized.args, auth_token=self.registry.auth_token)
                    response = f"**Tool: {tool_name}**\n\nResult: {tool_result}"
            except (ToolNotFoundError, ToolValidationError, ToolAuthError, ToolError, json.JSONDecodeError) as exc:
                response = f"**Tool error:** {exc}"

        # Check for system/internal commands
        elif prompt.lower().strip() == "/tools":
            descs = self.get_tool_descriptions()
            if not descs:
                response = "No tools registered."
            else:
                lines = ["**Available Tools:**\n"]
                for d in descs:
                    lines.append(f"- **{d['name']}**: {d['description']}")
                    if d.get("parameters"):
                        props = d["parameters"].get("properties", {})
                        for pname, pinfo in props.items():
                            required = pname in d["parameters"].get("required", [])
                            req_mark = " (required)" if required else ""
                            lines.append(f"  - `{pname}`: {pinfo.get('description', '')}{req_mark}")
                response = "\n".join(lines)

        elif prompt.lower().strip() == "/history":
            history = await self._get_history(limit=10)
            if not history:
                response = "No conversation history."
            else:
                lines = ["**Recent History:**\n"]
                for msg in history:
                    label = "You" if msg["role"] == "user" else "Hermes"
                    preview = msg["content"][:80]
                    if len(msg["content"]) > 80:
                        preview += "..."
                    lines.append(f"- **{label}**: {preview}")
                response = "\n".join(lines)

        elif prompt.lower().strip() == "/model":
            chain = self.router.fallback_chain
            preferred = chain[0] if chain else "(none)"
            preferred_tier = "local" if preferred.startswith("local:") or "/" not in preferred else "cloud"
            lines = [
                "**Current Model:**\n",
                f"- **Model**: `{preferred}`",
                f"- **Tier**: {preferred_tier}",
                f"- **Complexity threshold**: {self.router.local_max_complexity}",
                f"- **Escalation after**: {self.router.escalate_after_failures} consecutive local failures",
                f"- **Consecutive local failures**: {self.router.consecutive_local_failures}",
                "",
                "**Fallback Chain:**\n",
            ]
            for i, entry in enumerate(chain):
                tier_label = "local" if entry.startswith("local:") or "/" not in entry else "cloud"
                marker = " ← current" if i == 0 else ""
                lines.append(f"  {i + 1}. `{entry}` ({tier_label}){marker}")
            response = "\n".join(lines)

        elif prompt.lower().strip() == "/stats":
            history = await self._get_history(limit=10)
            user_msgs = [m for m in history if m.get("role") == "user"]
            asst_msgs = [m for m in history if m.get("role") == "assistant"]
            if not user_msgs and not asst_msgs:
                response = "No conversation turns yet."
            else:
                lines = ["**Last 10 Turns Summary:**\n"]
                turn_num = 1
                for m in history:
                    if m.get("role") == "user":
                        content = m.get("content", "")
                        preview = content[:60] + ("..." if len(content) > 60 else "")
                        lines.append(f"- **Turn {turn_num}:** You: {preview}")
                        turn_num += 1
                    elif m.get("role") == "assistant":
                        content = m.get("content", "")
                        preview = content[:60] + ("..." if len(content) > 60 else "")
                        lines.append(f"  Hermes: {preview}")
                lines.append(f"\n_Total: {len(user_msgs)} user, {len(asst_msgs)} assistant messages_")
                response = "\n".join(lines)

        elif prompt.lower().strip() == "/clear":
            if self.pool is not None:
                from hermes_lite.memory import delete_messages
                deleted = await delete_messages(self.pool, self.session_id)
                response = f"Conversation cleared ({deleted} messages removed). Cross-session memory preserved."
            else:
                self.session_id = uuid.uuid4().hex[:16]
                response = "Conversation cleared. Cross-session memory preserved."

        elif prompt.lower().strip() == "/memory":
            try:
                from hermes_lite.memory_bridge import get_default_bridge
                bridge = get_default_bridge()
                mem_entries = bridge.list("memory")
                user_entries = bridge.list("user")
                lines = ["**Memory:**\n"]
                if mem_entries:
                    lines.append(f"**Agent memory** ({len(mem_entries)} entries):")
                    for e in mem_entries[:10]:
                        lines.append(f"  - {e.content[:80]}")
                    if len(mem_entries) > 10:
                        lines.append(f"  _...and {len(mem_entries) - 10} more_")
                else:
                    lines.append("**Agent memory**: (none)")
                if user_entries:
                    lines.append(f"\n**User profile** ({len(user_entries)} entries):")
                    for e in user_entries[:10]:
                        lines.append(f"  - {e.content[:80]}")
                    if len(user_entries) > 10:
                        lines.append(f"  _...and {len(user_entries) - 10} more_")
                else:
                    lines.append("\n**User profile**: (none)")
                response = "\n".join(lines)
            except Exception as exc:
                response = f"**Memory**: Unable to load memory bridge ({exc})"

        elif prompt.lower().strip() == "/help":
            response = (
                "**Hermes-Lite Commands**\n\n"
                "Just type your message for a general response.\n\n"
                "**Tool invocation:**\n"
                "  `!tool_name {\"arg\": \"value\"}` — Call a registered tool directly\n\n"
                "**Commands:**\n"
                "  `/tools` — List registered tools and their schemas\n"
                "  `/model` — Show current model and fallback chain\n"
                "  `/stats` — Show last 10 turns summary\n"
                "  `/clear` — Reset conversation (keeps memory)\n"
                "  `/memory` — Show what's loaded into context\n"
                "  `/moa` — Show MoA status and available presets\n"
                "  `/moa <preset>` — Activate Mixture-of-Agents with a preset\n"
                "  `/moa off` — Deactivate MoA (return to normal)\n"
                "  `/history` — View recent conversation history\n"
                "  `/help` — Show this help message\n"
                "  `/exit`, `/quit`, `/q` — Exit the CLI\n"
                "  `Ctrl+C` or `Ctrl+D` — Exit the CLI"
            )

        elif prompt.lower().strip().startswith("/moa"):
            # /moa command — activate, deactivate, or show MoA status
            parts = prompt.strip().split(None, 1)  # ["/moa", "<arg>"]
            if len(parts) == 1:
                # /moa with no arg — show status + available presets
                if self.active_moa_preset:
                    response = (
                        f"**MoA Status:** Active\n\n"
                        f"{format_preset_info(self.active_moa_preset)}\n\n"
                        f"Use `/moa off` to deactivate."
                    )
                else:
                    presets_list = ", ".join(f"`{p}`" for p in list_moa_presets())
                    response = (
                        f"**MoA Status:** Inactive\n\n"
                        f"Available presets: {presets_list}\n\n"
                        f"Activate with `/moa <preset>` (e.g. `/moa council`)"
                    )
            elif parts[1].lower().strip() == "off":
                self.active_moa_preset = None
                response = "**MoA:** Deactivated. Using normal ToolLoop."
            else:
                preset_name = parts[1].strip().lower()
                preset = get_preset(preset_name)
                if preset:
                    self.active_moa_preset = preset
                    response = (
                        f"**MoA:** Activated with `{preset_name}` preset\n\n"
                        f"{format_preset_info(preset)}"
                    )
                else:
                    presets_list = ", ".join(f"`{p}`" for p in list_moa_presets())
                    response = (
                        f"**Unknown preset:** `{preset_name}`\n\n"
                        f"Available presets: {presets_list}"
                    )


        else:
            # General prompt — check if MoA is active, otherwise ToolLoop

            if self.active_moa_preset is not None:
                # MoA path — run reference models, then aggregator
                try:
                    moa_engine = MoAEngine(preset=self.active_moa_preset)
                    moa_messages = await self._build_llm_history(prompt)
                    moa_result: MoAResult = await moa_engine.run(
                        messages=moa_messages,
                        prompt=prompt,
                    )
                    response = format_moa_result(moa_result)

                    # Persist MoA metadata
                    routing_meta["moa"] = {
                        "preset": moa_result.metadata.get("preset"),
                        "refs_ok": moa_result.metadata.get("refs_ok"),
                        "refs_total": moa_result.metadata.get("refs_total"),
                        "agg_model": moa_result.aggregator_model,
                        "terminated_by": moa_result.terminated_by,
                    }

                    # Observability: log the turn
                    log_turn(
                        turn_id=uuid.uuid4().hex[:16],
                        tier="cloud",
                        model=moa_result.aggregator_model,
                        errors=[r.error for r in moa_result.reference_results if r.error],
                    )
                except AllKeysExhausted as exc:
                    logger.warning("MoA: all API keys exhausted (%s), falling back to local ToolLoop", exc)
                    # Fall back to local ToolLoop
                    loop_result = await self.tool_loop.run(
                        messages=await self._build_llm_history(prompt),
                        model=decision.model_id,
                    )
                    # Build the user-facing response with transparency info
                    tier_emoji = "⚡" if decision.tier == "local" else "☁️"
                    parts = []
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
                    parts.append("")  # blank line
                    parts.append(loop_result.response)
                    response = "\n".join(parts)

                    # Persist each tool-call and tool-result to memory
                    for i, tn in enumerate(loop_result.tool_names):
                        await self._save_message("tool", f"called {tn}", metadata={"tool_index": i})
                    # Save the loop outcome metadata for the assistant message
                    routing_meta["tool_loop"] = {
                        "iterations": loop_result.iterations,
                        "tool_calls_made": loop_result.tool_calls_made,
                        "tool_names": loop_result.tool_names,
                        "terminated_by": loop_result.terminated_by,
                    }

                    # Record the actual LLM outcome back into the router
                    self.router.record_outcome(
                        decision,
                        succeeded=(loop_result.terminated_by in ("complete", "repeated_error")),
                        tool_calls_malformed=(loop_result.terminated_by == "malformed_tool_call"),
                    )

                    # Observability: log the turn (T12)
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

            else:
                # Normal ToolLoop path
                try:
                    loop_result = await self.tool_loop.run(
                        messages=await self._build_llm_history(prompt),
                        model=decision.model_id,
                    )
                except AllKeysExhausted as exc:
                    logger.warning("ToolLoop: all API keys exhausted (%s), returning graceful error", exc)
                    response = (
                        f"⚠ All API keys exhausted. "
                        f"Earliest key available in {exc.cooldown_remaining:.0f}s. "
                        f"Try again later or set HERMES_LITE_LOCAL_MODEL for local fallback."
                    )
                    # Save assistant response with error metadata
                    await self._save_message(
                        "assistant",
                        response,
                        metadata={
                            "tool_called": False,
                            "routing": {"error": "all_keys_exhausted", "cooldown_remaining": exc.cooldown_remaining},
                        },
                    )
                    return response

                # Build the user-facing response with transparency info
                tier_emoji = "⚡" if decision.tier == "local" else "☁️"
                parts = []
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
                parts.append("")  # blank line
                parts.append(loop_result.response)
                response = "\n".join(parts)

                # Persist each tool-call and tool-result to memory
                for i, tn in enumerate(loop_result.tool_names):
                    await self._save_message("tool", f"called {tn}", metadata={"tool_index": i})
                # Save the loop outcome metadata for the assistant message
                routing_meta["tool_loop"] = {
                    "iterations": loop_result.iterations,
                    "tool_calls_made": loop_result.tool_calls_made,
                    "tool_names": loop_result.tool_names,
                    "terminated_by": loop_result.terminated_by,
                }

                # Record the actual LLM outcome back into the router
                self.router.record_outcome(
                    decision,
                    succeeded=(loop_result.terminated_by in ("complete", "repeated_error")),
                    tool_calls_malformed=(loop_result.terminated_by == "malformed_tool_call"),
                )

                # Observability: log the turn (T12)
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

        # For /clear we intentionally skip saving the assistant response
        # so the conversation history is truly empty after clearing.
        _skip_save = prompt.lower().strip() == "/clear"

        # Save assistant response — embed routing decision and tool
        # outcome so downstream workers (and the test suite) can verify
        # both.
        if not _skip_save:
            await self._save_message(
                "assistant",
                response,
                metadata={
                    "tool_called": tool_result is not None,
                    "routing": routing_meta,
                },
            )
        # For direct-tool and command paths, record a generic success
        # into the router (no real LLM call).
        _command_set = ("/tools", "/history", "/help", "/model", "/stats", "/memory", "/clear")
        _command_prefixes = ("/moa",)
        _is_command = prompt.lower().strip() in _command_set or any(
            prompt.lower().strip().startswith(p) for p in _command_prefixes
        )
        if not prompt.startswith("!") and not _is_command:
            pass  # already recorded above in the else branch
        else:
            self.router.record_outcome(decision, succeeded=True, tool_calls_malformed=False)
        return response

    async def _build_llm_history(
        self,
        current_prompt: str,
    ) -> list[dict[str, Any]]:
        """Build the OpenAI-format messages list from memory + current prompt.

        Includes a system message, the cross-session memory block injected
        via :class:`hermes_lite.memory_bridge.MemoryBridge.load_into_prompt`,
        recent history from SQLite, and the current user prompt as the
        last message.
        """
        # Cross-session memory (T9). Sync — the bridge holds its own lock
        # and SQLite is fast enough that we don't need to async-ify it
        # here. If the bridge isn't initialized (e.g. tests skip start()),
        # we still produce a clean system prompt with no memory block.
        memory_block = ""
        try:
            from hermes_lite.memory_bridge import get_default_bridge

            memory_block = get_default_bridge().load_into_prompt(max_chars=800)
        except Exception:
            memory_block = ""

        system_parts = [
            "You are Hermes-Lite, a helpful AI assistant with access to tools. "
            "Use tools when they help answer the user's question. "
            "When you have enough information to answer directly, respond in plain text without tool calls."
        ]
        if memory_block:
            system_parts.append(memory_block)

        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": "\n\n".join(system_parts)}
        ]

        # Pull recent conversation from memory
        db_history = await self._get_history(limit=16)
        for m in db_history:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant"):
                msgs.append({"role": role, "content": content})
            # Tool messages from memory are omitted here — they're
            # only relevant within a single ToolLoop run.

        # Append the current user prompt (it may already be in history
        # from the _save_message call, but we guarantee it's the last
        # message by adding it explicitly).
        # Scrub control tokens from user input to prevent prompt injection
        safe_prompt = scrub_control_tokens(current_prompt)
        msgs.append({"role": "user", "content": safe_prompt})
        return msgs

    def start(self) -> None:
        """Launch the CLI with the orchestrator wired in as the prompt handler."""
        self._create_default_tools()

        # We use asyncio.run inside run_cli — but run_cli calls asyncio.run
        # on its internal _run_async, so we need to provide a sync wrapper.
        # Launch the CLI — run_cli handles the event-loop detection internally
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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _dedupe(seq: list[str]) -> list[str]:
    """Return a de-duplicated copy of *seq*, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


__all__ = [
    "HermesOrchestrator",
    "ToolLoop",
    "ToolLoopResult",
]
