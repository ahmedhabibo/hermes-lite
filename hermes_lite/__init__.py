"""Hermes-Lite: Standalone local-first AI agent.

Hermes-Lite is fully standalone as of v0.8.0.  It does NOT depend on
Hermes Agent or any related runtime.  All configuration is loaded from
``HERMES_LITE_*`` environment variables and (optionally) a YAML file at
``~/.hermes_lite/config.yaml``.
"""

__version__ = "0.8.0"

import os

# Load .env on package import so env vars are available before llm.py reads them
def _load_env() -> None:
    # Skip .env loading during tests to avoid polluting env
    if os.getenv("HERMES_LITE_NO_DOTENV"):
        return
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        # 1. Walk up from this package to find project .env (highest priority)
        p = Path(__file__).resolve().parent
        for _ in range(5):
            env_path = p / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=True)  # project .env wins
                break
            if (p / "pyproject.toml").exists():
                break
            p = p.parent

        # 2. Also load CWD .env if different (for user overrides), but don't override
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(cwd_env, override=False)
    except ImportError:
        pass  # python-dotenv not installed

_load_env()

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
    create_session,
    get_session,
    update_session,
    delete_session,
    list_sessions,
    insert_message,
    get_messages,
    get_message_count,
    delete_messages,
    set_metadata,
    get_metadata,
    list_metadata,
    delete_metadata,
    session_context,
)
from hermes_lite.router import LiteRouter, RoutingDecision, parse_fallback_chain
from hermes_lite.orchestrator import HermesOrchestrator
from hermes_lite.cli import run_cli, PromptHandler
from hermes_lite.config import (
    HermesLiteConfig,
    get_config,
    reload_config,
    DEFAULT_FALLBACK_CHAIN,
)
from hermes_lite.tools_builtins import (
    register_builtins,
    ESSENTIAL_TOOL_NAMES,
    ToolResult,
    ReadFileArgs,
    SearchFilesArgs,
    TerminalArgs,
    MemoryArgs,
    WebSearchArgs,
    WebFetchArgs,
)
from hermes_lite.observability import (
    log_turn,
    read_turns,
    compute_stats,
    print_stats,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILE,
    MAX_LOG_BYTES,
)
from hermes_lite.subagent import (
    SubagentArgs,
    SubagentRunner,
    register_subagent_tool,
    SUBAGENT_TOOL_NAME,
    SUBAGENT_MAX_TOOL_CALLS,
    SUBAGENT_MAX_ITERATIONS,
    SUBAGENT_WALL_TIMEOUT_S,
    SUBAGENT_SYSTEM_PROMPT,
)
from hermes_lite.llm import (
    ChatRequest,
    ChatResponse,
    Tier,
    chat,
    tool_def,
    parse_tool_calls_from_text,
    RateLimiter,
    APIKeyRotator,
    AllKeysExhausted,
    get_rate_limiter,
    get_key_rotator,
    DEFAULT_RPM,
    DEFAULT_MAX_RETRIES,
)
from hermes_lite.moa import (
    MoAEngine,
    MoAPreset,
    MoAModelConfig,
    MoAResult,
    BUILTIN_PRESETS,
    get_preset,
    list_presets as list_moa_presets,
    format_preset_info,
    format_moa_result,
)

__all__ = [
    "PluginRegistry",
    "ToolDefinition",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolAuthError",
    "AsyncSQLitePool",
    "ensure_schema",
    "create_session",
    "get_session",
    "update_session",
    "delete_session",
    "list_sessions",
    "insert_message",
    "get_messages",
    "get_message_count",
    "delete_messages",
    "set_metadata",
    "get_metadata",
    "list_metadata",
    "delete_metadata",
    "session_context",
    "LiteRouter",
    "RoutingDecision",
    "parse_fallback_chain",
    "HermesOrchestrator",
    "run_cli",
    "PromptHandler",
    # Config (v0.8.0+)
    "HermesLiteConfig",
    "get_config",
    "reload_config",
    "DEFAULT_FALLBACK_CHAIN",
    # LLM layer
    "ChatRequest",
    "ChatResponse",
    "Tier",
    "chat",
    "tool_def",
    "parse_tool_calls_from_text",
    "RateLimiter",
    "APIKeyRotator",
    "AllKeysExhausted",
    "get_rate_limiter",
    "get_key_rotator",
    "DEFAULT_RPM",
    "DEFAULT_MAX_RETRIES",
    # 6 essentials
    "register_builtins",
    "ESSENTIAL_TOOL_NAMES",
    "ToolResult",
    "ReadFileArgs",
    "SearchFilesArgs",
    "TerminalArgs",
    "MemoryArgs",
    "WebSearchArgs",
    "WebFetchArgs",
    # Observability (T12)
    "log_turn",
    "read_turns",
    "compute_stats",
    "print_stats",
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_FILE",
    "MAX_LOG_BYTES",
    # Subagent (T8)
    "SubagentArgs",
    "SubagentRunner",
    "register_subagent_tool",
    "SUBAGENT_TOOL_NAME",
    "SUBAGENT_MAX_TOOL_CALLS",
    "SUBAGENT_MAX_ITERATIONS",
    "SUBAGENT_WALL_TIMEOUT_S",
    "SUBAGENT_SYSTEM_PROMPT",
    # MoA
    "MoAEngine",
    "MoAPreset",
    "MoAModelConfig",
    "MoAResult",
    "BUILTIN_PRESETS",
    "get_preset",
    "list_moa_presets",
    "format_preset_info",
    "format_moa_result",
]