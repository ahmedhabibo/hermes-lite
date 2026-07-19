"""hermes_lite.commands — Slash-command and direct-tool dispatch.

Keeps the command surface out of the orchestrator's main agent loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Protocol

from hermes_lite.moa import (
    format_preset_info,
    get_preset,
    list_presets as list_moa_presets,
)
from hermes_lite.registry import (
    ToolAuthError,
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
)
from hermes_lite.sanitize import sanitize_tool_args

logger = logging.getLogger(__name__)

# Exact-match slash commands (not including /moa which takes args)
COMMAND_SET = frozenset({
    "/tools",
    "/history",
    "/help",
    "/model",
    "/stats",
    "/memory",
    "/clear",
    "/cloud",
    "/local",
})
COMMAND_PREFIXES = ("/moa",)


class CommandContext(Protocol):
    """Minimal surface the command handlers need from the orchestrator."""

    session_id: str
    pool: Any
    router: Any
    registry: Any
    auth_token: Optional[str]
    active_moa_preset: Any
    force_tier: Optional[str]

    def get_tool_descriptions(self) -> list[dict[str, Any]]: ...
    async def _get_history(self, limit: int = 20) -> list[dict[str, Any]]: ...


def is_command(prompt: str) -> bool:
    text = prompt.lower().strip()
    if text in COMMAND_SET:
        return True
    return any(text.startswith(p) for p in COMMAND_PREFIXES)


def is_direct_tool(prompt: str) -> bool:
    return prompt.startswith("!")


async def dispatch_command(ctx: CommandContext, prompt: str) -> str:
    """Handle a slash command; mutates ctx for /cloud /local /moa /clear."""
    text = prompt.lower().strip()

    if text == "/tools":
        return _cmd_tools(ctx)
    if text == "/history":
        return await _cmd_history(ctx)
    if text == "/model":
        return _cmd_model(ctx)
    if text == "/stats":
        return await _cmd_stats(ctx)
    if text == "/clear":
        return await _cmd_clear(ctx)
    if text == "/memory":
        return _cmd_memory()
    if text == "/cloud":
        ctx.force_tier = "cloud"
        return (
            "☁️ **Cloud mode forced** — all requests will use NVIDIA NIM cloud models.\n"
            "Use `/local` to switch back to local-first routing."
        )
    if text == "/local":
        ctx.force_tier = None
        ctx.router.reset()
        return (
            "🏠 **Local mode restored** — requests will use the local Qwen model by default,\n"
            "escalating to cloud only for heavy tasks (multi-step, large context, complex reasoning)."
        )
    if text == "/help":
        return _cmd_help()
    if text.startswith("/moa"):
        return _cmd_moa(ctx, prompt)
    return f"Unknown command: {prompt}"


def dispatch_direct_tool(ctx: CommandContext, prompt: str) -> str:
    """Handle ``!tool_name {json}`` direct invocation."""
    try:
        space_idx = prompt.find(" ")
        if space_idx > 1:
            tool_name = prompt[1:space_idx].strip()
            args_str = prompt[space_idx + 1 :].strip()
            args = json.loads(args_str) if args_str else {}
        else:
            tool_name = prompt[1:].strip()
            args = {}

        sanitized = sanitize_tool_args(tool_name=tool_name, args=args)
        if not sanitized.is_clean:
            issues = "; ".join(sanitized.issues)
            logger.warning("[SANITIZE] '%s' blocked: %s", tool_name, issues)
            return f"Tool call blocked: security policy violation — {issues}"

        tool_result = ctx.registry.call_tool(
            tool_name, sanitized.args, auth_token=ctx.registry.auth_token
        )
        return f"**Tool: {tool_name}**\n\nResult: {tool_result}"
    except (
        ToolNotFoundError,
        ToolValidationError,
        ToolAuthError,
        ToolError,
        json.JSONDecodeError,
    ) as exc:
        return f"**Tool error:** {exc}"


def _cmd_tools(ctx: CommandContext) -> str:
    descs = ctx.get_tool_descriptions()
    if not descs:
        return "No tools registered."
    lines = ["**Available Tools:**\n"]
    for d in descs:
        lines.append(f"- **{d['name']}**: {d['description']}")
        if d.get("parameters"):
            props = d["parameters"].get("properties", {})
            for pname, pinfo in props.items():
                required = pname in d["parameters"].get("required", [])
                req_mark = " (required)" if required else ""
                lines.append(
                    f"  - `{pname}`: {pinfo.get('description', '')}{req_mark}"
                )
    return "\n".join(lines)


async def _cmd_history(ctx: CommandContext) -> str:
    history = await ctx._get_history(limit=10)
    if not history:
        return "No conversation history."
    lines = ["**Recent History:**\n"]
    for msg in history:
        label = "You" if msg["role"] == "user" else "Hermes"
        preview = msg["content"][:80]
        if len(msg["content"]) > 80:
            preview += "..."
        lines.append(f"- **{label}**: {preview}")
    return "\n".join(lines)


def _cmd_model(ctx: CommandContext) -> str:
    chain = ctx.router.fallback_chain
    preferred = chain[0] if chain else "(none)"
    preferred_tier = (
        "local" if preferred.startswith("local:") or "/" not in preferred else "cloud"
    )
    lines = [
        "**Current Model:**\n",
        f"- **Model**: `{preferred}`",
        f"- **Tier**: {preferred_tier}",
        f"- **Complexity threshold**: {ctx.router.local_max_complexity}",
        f"- **Escalation after**: {ctx.router.escalate_after_failures} consecutive local failures",
        f"- **Consecutive local failures**: {ctx.router.consecutive_local_failures}",
        "",
        "**Fallback Chain:**\n",
    ]
    for i, entry in enumerate(chain):
        tier_label = (
            "local" if entry.startswith("local:") or "/" not in entry else "cloud"
        )
        marker = " ← current" if i == 0 else ""
        lines.append(f"  {i + 1}. `{entry}` ({tier_label}){marker}")
    return "\n".join(lines)


async def _cmd_stats(ctx: CommandContext) -> str:
    history = await ctx._get_history(limit=10)
    user_msgs = [m for m in history if m.get("role") == "user"]
    asst_msgs = [m for m in history if m.get("role") == "assistant"]
    if not user_msgs and not asst_msgs:
        return "No conversation turns yet."
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
    lines.append(
        f"\n_Total: {len(user_msgs)} user, {len(asst_msgs)} assistant messages_"
    )
    return "\n".join(lines)


async def _cmd_clear(ctx: CommandContext) -> str:
    import uuid

    if ctx.pool is not None:
        from hermes_lite.memory import delete_messages

        deleted = await delete_messages(ctx.pool, ctx.session_id)
        return (
            f"Conversation cleared ({deleted} messages removed). "
            "Cross-session memory preserved."
        )
    ctx.session_id = uuid.uuid4().hex[:16]
    return "Conversation cleared. Cross-session memory preserved."


def _cmd_memory() -> str:
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
        return "\n".join(lines)
    except Exception as exc:
        return f"**Memory**: Unable to load memory bridge ({exc})"


def _cmd_help() -> str:
    return (
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
        "  `/cloud` — Force cloud NIM for all requests (heavy mode)\n"
        "  `/local` — Return to local-first routing (default)\n"
        "  `/moa` — Show MoA status and available presets\n"
        "  `/moa <preset>` — Activate Mixture-of-Agents with a preset\n"
        "  `/moa off` — Deactivate MoA (return to normal)\n"
        "  `/history` — View recent conversation history\n"
        "  `/help` — Show this help message\n"
        "  `/exit`, `/quit`, `/q` — Exit the CLI\n"
        "  `Ctrl+C` or `Ctrl+D` — Exit the CLI"
    )


def _cmd_moa(ctx: CommandContext, prompt: str) -> str:
    parts = prompt.strip().split(None, 1)
    if len(parts) == 1:
        if ctx.active_moa_preset:
            return (
                f"**MoA Status:** Active\n\n"
                f"{format_preset_info(ctx.active_moa_preset)}\n\n"
                f"Use `/moa off` to deactivate."
            )
        presets_list = ", ".join(f"`{p}`" for p in list_moa_presets())
        return (
            f"**MoA Status:** Inactive\n\n"
            f"Available presets: {presets_list}\n\n"
            f"Activate with `/moa <preset>` (e.g. `/moa council`)"
        )
    if parts[1].lower().strip() == "off":
        ctx.active_moa_preset = None
        return "**MoA:** Deactivated. Using normal ToolLoop."
    preset_name = parts[1].strip().lower()
    preset = get_preset(preset_name)
    if preset:
        ctx.active_moa_preset = preset
        return (
            f"**MoA:** Activated with `{preset_name}` preset\n\n"
            f"{format_preset_info(preset)}"
        )
    presets_list = ", ".join(f"`{p}`" for p in list_moa_presets())
    return (
        f"**Unknown preset:** `{preset_name}`\n\n"
        f"Available presets: {presets_list}"
    )


__all__ = [
    "COMMAND_SET",
    "CommandContext",
    "dispatch_command",
    "dispatch_direct_tool",
    "is_command",
    "is_direct_tool",
]
