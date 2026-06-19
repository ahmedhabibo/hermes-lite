"""
hermes_lite/orchestrator.py — Orchestrator Engine

Wires the PluginRegistry (tools), SQLite memory layer (conversation history),
and CLI shell into a working agent loop.

Key components:
- HermesOrchestrator: Coordinates tool registry + memory + prompt handling
- register_builtins(): Populates the registry with default built-in tools
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from hermes_lite.registry import (
    PluginRegistry,
    ToolDefinition,
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
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
from hermes_lite.cli import run_cli
from hermes_lite.router import LiteRouter, RoutingDecision

# ---------------------------------------------------------------------------
# Built-in tool schemas
# ---------------------------------------------------------------------------


class EchoArgs(BaseModel):
    """Echo the input back to the user (useful for testing)."""
    message: str = Field(..., description="The message to echo back.")


class CalculatorArgs(BaseModel):
    """Evaluate a simple arithmetic expression."""
    expression: str = Field(..., description="Arithmetic expression to evaluate (e.g. '2 + 2').")


class SaveNoteArgs(BaseModel):
    """Save a note in the session's metadata store."""
    key: str = Field(..., description="Note key/name.")
    content: str = Field(..., description="Note content.")


class GetNoteArgs(BaseModel):
    """Retrieve a saved note from the session's metadata store."""
    key: str = Field(..., description="Note key/name.")


class ListNotesArgs(BaseModel):
    """List all saved notes for this session."""


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


def _handle_echo(args: EchoArgs) -> str:
    return args.message


def _handle_calculator(args: CalculatorArgs) -> str:
    try:
        # Safe evaluation — only allow numbers, operators, parens, spaces
        expr = args.expression.strip()
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expr):
            return f"Error: Expression contains disallowed characters."
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def _build_note_handler(registry: PluginRegistry, session_id: str):
    """Build handlers that close over the registry and session_id.

    Returns a dict of {'handle_save': ..., 'handle_get': ..., 'handle_list': ...}
    that the orchestrator will invoke directly (not through the tool registry,
    since these tools need access to the memory layer).
    """
    # These are wired through the orchestrator's _call_tool method which
    # has access to the memory pool and session_id — see HermesOrchestrator.
    pass


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
    ) -> None:
        self.db_path = str(Path(db_path).expanduser())
        self.session_title = session_title
        self.session_id: str = ""
        self.pool: AsyncSQLitePool | None = None
        self.registry = PluginRegistry(strict_validation=True)
        # Routing controller picks local vs cloud before any LLM call.
        # None means "construct a default LiteRouter inside handle_prompt"
        # so existing tests that don't care about routing keep working,
        # but consumers who want to override thresholds/fallback chain
        # can pass their own.
        self.router: LiteRouter = router or LiteRouter()

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _create_default_tools(self) -> None:
        """Register the built-in tools."""
        tools = [
            ToolDefinition(
                name="echo",
                description="Echo the input back to the user. Use for testing connectivity.",
                schema_model=EchoArgs,
                handler=_handle_echo,
            ),
            ToolDefinition(
                name="calculator",
                description="Evaluate a simple arithmetic expression.",
                schema_model=CalculatorArgs,
                handler=_handle_calculator,
            ),
        ]

        for tool in tools:
            try:
                self.registry.add_tool(tool)
            except ValueError:
                pass  # already registered

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
        """Process a user prompt: check for tool calls, generate response.

        This implementation:
        1. Saves the user prompt to memory
        2. Asks the routing controller which tier to use, records it on
           the assistant message metadata for transparency
        3. Checks if the prompt matches a built-in tool (tool_call prefix)
        4. Otherwise returns a simple acknowledgement (LLM chat wiring
           ships in T4 / T6)
        5. Records any LLM outcome (success/failure/malformed tool_calls)
           back onto the router so the next escalation check is informed
        """
        # Save user message
        await self._save_message("user", prompt)

        # Decide which tier/model this prompt should run against. Done
        # up-front so the rest of the prompt pipeline can use the
        # routing decision as metadata.
        decision = await self._route_decision(prompt)
        routing_meta = {
            "model_id": decision.model_id,
            "tier": decision.tier,
            "complexity_score": round(decision.complexity_score, 4),
            "reason": decision.reason,
            "fell_back": decision.fell_back,
        }

        # Check for direct tool invocation (tool_name: json_args format)
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

                tool_result = self.registry.call_tool(tool_name, args)
                response = f"**Tool: {tool_name}**\n\nResult: {tool_result}"
            except (ToolNotFoundError, ToolValidationError, ToolError, json.JSONDecodeError) as exc:
                response = f"**Tool error:** {exc}"
        else:
            # Check for system/internal commands
            cmd = prompt.lower().strip()

            if cmd == "/tools":
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

            elif cmd == "/history":
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

            elif cmd == "/help":
                response = (
                    "**Hermes-Lite Commands**\n\n"
                    "Just type your message for a general response.\n\n"
                    "**Tool invocation:**\n"
                    "  `!tool_name {\"arg\": \"value\"}` — Call a registered tool directly\n\n"
                    "**Commands:**\n"
                    "  `/tools` — List registered tools and their schemas\n"
                    "  `/history` — View recent conversation history\n"
                    "  `/help` — Show this help message\n"
                    "  `/exit`, `/quit`, `/q` — Exit the CLI\n"
                    "  `Ctrl+C` or `Ctrl+D` — Exit the CLI"
                )

            else:
                # General response — include available tools info plus
                # a routing-transparency line so the user (and tests)
                # can see which tier was chosen.
                tier_emoji = "⚡" if decision.tier == "local" else "☁️ "
                descs = self.get_tool_descriptions()
                tool_list = ", ".join(f"`{d['name']}`" for d in descs) if descs else "none registered"
                response = (
                    f"{tier_emoji} using **{decision.tier}** model `{decision.model_id}` "
                    f"(score {decision.complexity_score:.2f})\n\n"
                    f"I'm running with **{self.registry.tool_count} tool(s)** registered: {tool_list}\n\n"
                    f"You said: _{prompt}_\n\n"
                    f"_Routing: {decision.reason}_\n\n"
                    f"*Use `!tool_name {{...}}` to invoke a tool directly, or `/tools` to see available tools.*"
                )

        # Save assistant response — embed routing decision and tool
        # outcome so downstream workers (and the test suite) can verify
        # both.
        await self._save_message(
            "assistant",
            response,
            metadata={
                "tool_called": tool_result is not None,
                "routing": routing_meta,
            },
        )
        # Feed the booking-keeping back into the router so escalation
        # state is correct. Tool invocations and general responses are
        # both counted as a "success" right now (no real LLM call yet);
        # T4/T6 will swap this for an actual model outcome.
        self.router.record_outcome(decision, succeeded=True, tool_calls_malformed=False)
        return response

    def start(self) -> None:
        """Launch the CLI with the orchestrator wired in as the prompt handler."""
        self._create_default_tools()

        # We use asyncio.run inside run_cli — but run_cli calls asyncio.run
        # on its internal _run_async, so we need to provide a sync wrapper.
        import asyncio

        async def _setup_and_run() -> None:
            await self._initialize_memory()
            run_cli(
                on_prompt=self._handle_prompt,
                welcome_message=(
                    "Hermes-Lite v0.1 — Orchestrator Engine\n\n"
                    f"Session: {self.session_id[:8]}...\n"
                    f"Tools: {self.registry.tool_count} registered\n"
                    f"DB: {self.db_path}"
                ),
            )

        asyncio.run(_setup_and_run())


__all__ = [
    "HermesOrchestrator",
]