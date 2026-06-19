"""Hermes-Lite: Lightweight tool-execution framework for LLM agents."""

from hermes_lite.registry import PluginRegistry, ToolDefinition, ToolError, ToolNotFoundError, ToolValidationError
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

__all__ = [
    "PluginRegistry",
    "ToolDefinition",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
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
]