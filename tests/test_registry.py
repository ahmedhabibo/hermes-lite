"""
Unit tests for hermes_lite.registry -- PluginRegistry & ToolDefinition.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from hermes_lite.registry import (
    PluginRegistry,
    ToolDefinition,
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
)


# ===================================================================
# Fixtures & Helpers
# ===================================================================


class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, ge=1, le=100)


class NoArgs(BaseModel):
    """Tool that accepts no arguments."""


class EmailArgs(BaseModel):
    recipient: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    subject: str = Field(default="(no subject)")
    body: str = Field(default="")


# Handlers
def handle_search(args: SearchArgs) -> str:
    return f"Searched for '{args.query}' (max={args.max_results})"


def handle_noop() -> str:
    return "noop done"


def handle_email(args: EmailArgs) -> dict[str, Any]:
    return {"to": args.recipient, "subject": args.subject, "body": args.body}


# Standard tool definitions
SEARCH_TOOL = ToolDefinition(
    name="web_search",
    description="Search the web.",
    schema_model=SearchArgs,
    handler=handle_search,
)

NOOP_TOOL = ToolDefinition(
    name="noop",
    description="Do nothing.",
    schema_model=NoArgs,
    handler=handle_noop,
)

EMAIL_TOOL = ToolDefinition(
    name="send_email",
    description="Send an email.",
    schema_model=EmailArgs,
    handler=handle_email,
)


@pytest.fixture
def registry() -> PluginRegistry:
    return PluginRegistry(strict_validation=True)


@pytest.fixture
def populated_registry(registry: PluginRegistry) -> PluginRegistry:
    registry.add_tool(SEARCH_TOOL)
    registry.add_tool(NOOP_TOOL)
    registry.add_tool(EMAIL_TOOL)
    return registry


# ===================================================================
# ToolDefinition tests
# ===================================================================


class TestToolDefinition:
    def test_create_minimal(self) -> None:
        """A tool with name-only (no schema, no handler) is valid."""
        t = ToolDefinition(name="ping", description="Ping tool")
        assert t.name == "ping"
        assert t.description == "Ping tool"
        assert t.schema_model is None
        assert t.handler is None

    def test_create_with_schema_and_handler(self) -> None:
        t = ToolDefinition(
            name="echo",
            description="Echo input",
            schema_model=SearchArgs,
            handler=lambda x: x.query,
        )
        assert t.name == "echo"
        assert t.schema_model is SearchArgs

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ToolDefinition(name="", description="bad")
        with pytest.raises(ValueError, match="non-empty"):
            ToolDefinition(name="   ", description="bad")

    def test_non_pydantic_schema_raises(self) -> None:
        with pytest.raises(TypeError, match="BaseModel"):
            ToolDefinition(
                name="bad",
                description="bad",
                schema_model=dict,  # type: ignore[arg-type]
            )

    # -- json_schema property

    def test_json_schema_none_when_no_schema(self) -> None:
        t = ToolDefinition(name="x", description="x")
        assert t.json_schema is None

    def test_json_schema_returns_valid_schema(self) -> None:
        schema = SEARCH_TOOL.json_schema
        assert schema is not None
        assert "properties" in schema
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["max_results"]["default"] == 5

    # -- argument_names

    def test_argument_names_no_schema(self) -> None:
        t = ToolDefinition(name="x", description="x")
        assert t.argument_names == frozenset()

    def test_argument_names_with_schema(self) -> None:
        assert SEARCH_TOOL.argument_names == frozenset({"query", "max_results"})

    # -- validate_args

    def test_validate_args_valid(self) -> None:
        validated = SEARCH_TOOL.validate_args({"query": "hello"})
        assert validated is not None
        assert validated.query == "hello"
        assert validated.max_results == 5  # default

    def test_validate_args_none_becomes_empty_and_fails_for_required(self) -> None:
            """None becomes {}, but required field still fails."""
            with pytest.raises(ToolValidationError, match="query"):
                SEARCH_TOOL.validate_args(None)

    def test_validate_args_fails_on_missing_required(self) -> None:
        with pytest.raises(ToolValidationError) as excinfo:
            SEARCH_TOOL.validate_args({})
        assert "web_search" in str(excinfo.value)
        assert len(excinfo.value.errors) > 0

    def test_validate_args_fails_on_type_mismatch(self) -> None:
        with pytest.raises(ToolValidationError) as excinfo:
            SEARCH_TOOL.validate_args({"query": 42})
        assert len(excinfo.value.errors) > 0

    def test_validate_args_fails_on_extra_fields_strict(self) -> None:
        """strict=True means extra fields cause validation failure."""
        with pytest.raises(ToolValidationError):
            SEARCH_TOOL.validate_args({"query": "hi", "unknown_field": "x"})

    def test_validate_args_no_schema_returns_none(self) -> None:
        t = ToolDefinition(name="x", description="x")
        assert t.validate_args({"anything": "goes"}) is None

    def test_validate_args_no_args_tool(self) -> None:
        validated = NOOP_TOOL.validate_args({})
        assert validated is not None

    def test_validate_args_no_args_tool_with_none(self) -> None:
        validated = NOOP_TOOL.validate_args(None)
        assert validated is not None


# ===================================================================
# PluginRegistry tests
# ===================================================================


class TestPluginRegistryRegistration:
    def test_add_tool(self, registry: PluginRegistry) -> None:
        registry.add_tool(SEARCH_TOOL)
        assert registry.tool_count == 1
        assert registry.has_tool("web_search")

    def test_add_tool_duplicate_raises(self, registry: PluginRegistry) -> None:
        registry.add_tool(SEARCH_TOOL)
        with pytest.raises(ValueError, match="already registered"):
            registry.add_tool(SEARCH_TOOL)

    def test_add_tool_duplicate_name_different_definition(self, registry: PluginRegistry) -> None:
        t1 = ToolDefinition(name="same", description="first", schema_model=NoArgs)
        t2 = ToolDefinition(name="same", description="second", schema_model=NoArgs)
        registry.add_tool(t1)
        with pytest.raises(ValueError, match="already registered"):
            registry.add_tool(t2)

    def test_add_tool_no_schema_in_strict_mode_raises(self) -> None:
        reg = PluginRegistry(strict_validation=True)
        t = ToolDefinition(name="bare", description="no schema")
        with pytest.raises(ValueError, match="strict_validation=True"):
            reg.add_tool(t)

    def test_add_tool_no_schema_in_non_strict_mode(self) -> None:
        reg = PluginRegistry(strict_validation=False)
        t = ToolDefinition(name="bare", description="no schema")
        reg.add_tool(t)  # should not raise
        assert reg.has_tool("bare")

    def test_remove_tool(self, populated_registry: PluginRegistry) -> None:
        populated_registry.remove_tool("web_search")
        assert not populated_registry.has_tool("web_search")
        assert populated_registry.tool_count == 2

    def test_remove_tool_missing_raises(self, registry: PluginRegistry) -> None:
        with pytest.raises(ToolNotFoundError):
            registry.remove_tool("nonexistent")

    def test_get_tool(self, populated_registry: PluginRegistry) -> None:
        t = populated_registry.get_tool("web_search")
        assert t.name == "web_search"
        assert t.schema_model is SearchArgs

    def test_get_tool_missing_raises(self, registry: PluginRegistry) -> None:
        with pytest.raises(ToolNotFoundError):
            registry.get_tool("ghost")

    def test_has_tool(self, populated_registry: PluginRegistry) -> None:
        assert populated_registry.has_tool("web_search")
        assert not populated_registry.has_tool("does_not_exist")

    def test_list_tools(self, populated_registry: PluginRegistry) -> None:
        tools = populated_registry.list_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"web_search", "noop", "send_email"}

    def test_tool_count(self, populated_registry: PluginRegistry) -> None:
        assert populated_registry.tool_count == 3


class TestPluginRegistryDispatch:
    def test_call_tool_valid(self, populated_registry: PluginRegistry) -> None:
        result = populated_registry.call_tool("web_search", {"query": "python"})
        assert result == "Searched for 'python' (max=5)"

    def test_call_tool_with_defaults(self, populated_registry: PluginRegistry) -> None:
        result = populated_registry.call_tool(
            "web_search", {"query": "test", "max_results": 10}
        )
        assert "max=10" in result

    def test_call_tool_no_args(self, populated_registry: PluginRegistry) -> None:
        result = populated_registry.call_tool("noop")
        assert result == "noop done"

    def test_call_tool_missing_name(self, registry: PluginRegistry) -> None:
        with pytest.raises(ToolNotFoundError):
            registry.call_tool("does_not_exist", {})

    def test_call_tool_invalid_args_missing_required(
        self, populated_registry: PluginRegistry
    ) -> None:
        with pytest.raises(ToolValidationError) as excinfo:
            populated_registry.call_tool("web_search", {})

        assert "web_search" in str(excinfo.value)
        assert any("query" in str(e["location"]) for e in excinfo.value.errors)

    def test_call_tool_invalid_args_wrong_type(
        self, populated_registry: PluginRegistry
    ) -> None:
        with pytest.raises(ToolValidationError):
            populated_registry.call_tool("web_search", {"query": 123})

    def test_call_tool_extra_fields_strict(self, populated_registry: PluginRegistry) -> None:
        """Strict mode rejects extra fields not in the schema."""
        with pytest.raises(ToolValidationError):
            populated_registry.call_tool(
                "web_search", {"query": "hello", "made_up_field": "x"}
            )

    def test_call_tool_email_valid(self, populated_registry: PluginRegistry) -> None:
        result = populated_registry.call_tool(
            "send_email",
            {"recipient": "alice@example.com", "subject": "Hi", "body": "Hello Alice"},
        )
        assert result["to"] == "alice@example.com"

    def test_call_tool_email_invalid_pattern(self, populated_registry: PluginRegistry) -> None:
        with pytest.raises(ToolValidationError):
            populated_registry.call_tool(
                "send_email", {"recipient": "not-an-email"}
            )

    def test_call_tool_no_handler_raises(self, registry: PluginRegistry) -> None:
        t = ToolDefinition(name="no_handler", description="no handler", schema_model=NoArgs)
        registry.add_tool(t)
        with pytest.raises(ToolError, match="no handler"):
            registry.call_tool("no_handler")

    def test_call_tool_handler_raises_exception(self, registry: PluginRegistry) -> None:
        def broken_handler(_: Any) -> None:
            raise RuntimeError("something broke")

        t = ToolDefinition(
            name="broken",
            description="broken",
            schema_model=SearchArgs,
            handler=broken_handler,
        )
        registry.add_tool(t)
        with pytest.raises(ToolError, match="broken"):
            registry.call_tool("broken", {"query": "x"})

    def test_call_tool_no_args_with_none(self, populated_registry: PluginRegistry) -> None:
        """Test calling with explicit None args on a no-arg tool."""
        result = populated_registry.call_tool("noop", None)
        assert result == "noop done"

    def test_call_tool_strict_validation_enforces_types(self, populated_registry: PluginRegistry) -> None:
        """Even '5' as string for an int field is rejected in strict mode."""
        with pytest.raises(ToolValidationError):
            populated_registry.call_tool(
                "web_search", {"query": "hello", "max_results": "5"}
            )


class TestPluginRegistryToolDescriptions:
    def test_tool_descriptions(self, populated_registry: PluginRegistry) -> None:
        descs = populated_registry.tool_descriptions()
        assert len(descs) == 3

        search_desc = next(d for d in descs if d["name"] == "web_search")
        assert search_desc["description"] == "Search the web."
        assert search_desc["parameters"] is not None
        assert "properties" in search_desc["parameters"]

    def test_tool_descriptions_empty(self, registry: PluginRegistry) -> None:
        assert registry.tool_descriptions() == []


class TestPluginRegistryStrictMode:
    def test_default_is_strict(self) -> None:
        reg = PluginRegistry()
        assert reg._strict is True

    def test_non_strict_allows_schema_free_tools(self) -> None:
        reg = PluginRegistry(strict_validation=False)
        t = ToolDefinition(name="bare", description="no schema")
        reg.add_tool(t)
        # Calling a handler with no schema should work
        def handler() -> str:
            return "bare ok"

        reg = PluginRegistry(strict_validation=False)
        t = ToolDefinition(name="bare", description="no schema", handler=handler)
        reg.add_tool(t)
        result = reg.call_tool("bare")
        assert result == "bare ok"

    def test_strict_rejects_schema_free_tools(self) -> None:
        reg = PluginRegistry(strict_validation=True)
        t = ToolDefinition(name="bare", description="no schema")
        with pytest.raises(ValueError, match="strict_validation"):
            reg.add_tool(t)

    def test_repr(self, populated_registry: PluginRegistry) -> None:
        r = repr(populated_registry)
        assert "PluginRegistry" in r
        assert "tools=3" in r
        assert "strict=True" in r

    def test_repr_empty(self, registry: PluginRegistry) -> None:
        r = repr(registry)
        assert "tools=0" in r