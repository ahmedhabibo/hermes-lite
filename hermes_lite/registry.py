"""
hermes_lite.registry -- Tool Registry & Strict Schema Validation

Core PluginRegistry that catches and validates structural arguments from
4B/7B models to block JSON hallucinations. All tools registered must carry
a Pydantic (BaseModel) schema for their arguments; the registry enforces
strict validation at dispatch time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Base exception for all tool-related errors."""


class ToolAuthError(ToolError):
    """Raised when tool execution is blocked due to missing/invalid auth token."""

    def __init__(self, tool_name: str, message: str | None = None) -> None:
        self.tool_name = tool_name
        msg = message or f"Tool '{tool_name}' requires authentication. Set HERMES_LITE_AUTH_TOKEN or use --auth-token."
        super().__init__(msg)


class ToolNotFoundError(ToolError):
    """Raised when a requested tool name is not registered."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' is not registered in the PluginRegistry.")


class ToolValidationError(ToolError):
    """Raised when tool arguments fail strict schema validation."""

    def __init__(self, tool_name: str, errors: list[dict[str, Any]]) -> None:
        self.tool_name = tool_name
        self.errors = errors
        msg = f"Tool '{tool_name}' argument validation failed: {errors}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# ToolDefinition -- the value object for registered tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable descriptor for a registered tool.

    Attributes:
        name: Unique tool name (lowercase, snake_case convention).
        description: Human-readable description shown to the LLM.
        schema_model: A **Pydantic BaseModel subclass** that defines the
            tool's expected arguments.  The model's JSON schema is extracted
            automatically and used for validation.  Pass ``None`` for tools
            that accept no arguments.
        handler: Callable that implements the tool.  If *schema_model* is
            ``None`` the handler is called with no positional/keyword args;
            otherwise it receives the validated Pydantic model instance.
        dangerous: If ``True``, this tool requires authentication when
            ``PluginRegistry.auth_token`` is set.  Default ``False``.
            Mark tools that can read/write files, execute commands, or
            access external resources as dangerous.
    """

    name: str
    description: str
    schema_model: type[BaseModel] | None = None
    handler: Callable[..., Any] | None = None
    dangerous: bool = False

    def __post_init__(self) -> None:
        """Light validation on construction -- catches obvious typos early."""
        if not self.name or not self.name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if self.schema_model is not None and (
            not isinstance(self.schema_model, type) or not issubclass(self.schema_model, BaseModel)
        ):
            raise TypeError(
                f"schema_model must be a Pydantic BaseModel subclass or None, "
                f"got {type(self.schema_model).__name__}."
            )

    # -- convenience accessors ------------------------------------------

    @property
    def json_schema(self) -> dict[str, Any] | None:
        """Return the JSON Schema dict for this tool's arguments, or None."""
        if self.schema_model is None:
            return None
        return self.schema_model.model_json_schema()

    @property
    def argument_names(self) -> frozenset[str]:
        """Return the set of expected argument names (empty set if no args)."""
        if self.schema_model is None:
            return frozenset()
        return frozenset(self.schema_model.model_fields.keys())

    @property
    def has_arguments(self) -> bool:
        """Return ``True`` if this tool expects any arguments (has schema fields)."""
        if self.schema_model is None:
            return False
        return len(self.schema_model.model_fields) > 0

    def validate_args(
        self,
        args: dict[str, Any] | None,
        forbid_extra: bool = True,
    ) -> BaseModel | None:
        """Validate *args* against the schema model.

        Returns the validated Pydantic model (or None if no schema).
        Raises ``ToolValidationError`` on failure.

        When *forbid_extra* is ``True`` (the default), any key in *args*
        that is not a field on the schema model will cause validation to
        fail.  This is separate from Pydantic's own ``strict`` mode which
        only controls type coercion.
        """
        if self.schema_model is None:
            return None
        if args is None:
            args = {}
        # -- forbid extra fields --------------------------------------------
        if forbid_extra and self.schema_model is not None:
            allowed = set(self.schema_model.model_fields.keys())
            extra = set(args.keys()) - allowed
            if extra:
                errors = [
                    {
                        "location": [k],
                        "message": f"Extra field '{k}' is not allowed.",
                        "input": args[k],
                    }
                    for k in sorted(extra)
                ]
                raise ToolValidationError(self.name, errors)
        # -------------------------------------------------------------------
        try:
            return self.schema_model.model_validate(args, strict=True)
        except ValidationError as exc:
            errors = []
            for err in exc.errors():
                loc = list(err.get("loc", []))
                msg = err.get("msg", "unknown error")
                inp = err.get("input")
                errors.append({
                    "location": loc,
                    "message": msg,
                    "input": inp,
                })
            raise ToolValidationError(self.name, errors) from exc


# ---------------------------------------------------------------------------
# PluginRegistry -- the core registry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Thread-safe-ish registry of tool plugins with strict argument validation.

    Typical usage::

        class SearchArgs(BaseModel):
            query: str = Field(..., description="Search query string")
            max_results: int = Field(default=5, ge=1, le=50)

        def search_handler(args: SearchArgs) -> str:
            return f"Searching for {args.query}..."

        registry = PluginRegistry()
        registry.add_tool(
            ToolDefinition(
                name="web_search",
                description="Search the web for information.",
                schema_model=SearchArgs,
                handler=search_handler,
            )
        )

        # Dispatch
        result = registry.call_tool("web_search", {"query": "hello", "max_results": 10})

    Authentication:
        If ``auth_token`` is provided (via constructor or ``HERMES_LITE_AUTH_TOKEN``
        env var), tools marked ``dangerous=True`` will require the token to be
        passed at dispatch time via the ``auth_token`` kwarg.  If the token
        doesn't match, ``ToolAuthError`` is raised.
    """

    def __init__(self, strict_validation: bool = True, auth_token: str | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._strict = strict_validation
        self.auth_token = auth_token  # None = no auth required

    # -- registration ---------------------------------------------------

    def add_tool(self, definition: ToolDefinition) -> None:
        """Register a tool.

        If *strict_validation* is ``True`` (the default) the tool's
        ``schema_model`` **must** be a Pydantic BaseModel -- tools without
        a schema are rejected.  When ``strict_validation`` is ``False``
        any tool may be registered regardless of schema presence.

        Raises ``ValueError`` if a tool with the same name is already
        registered.
        """
        if definition.name in self._tools:
            raise ValueError(
                f"Tool '{definition.name}' is already registered. "
                f"Remove it first or use a different name."
            )
        if self._strict and definition.schema_model is None:
            raise ValueError(
                f"Cannot register tool '{definition.name}' without a Pydantic schema "
                f"when strict_validation=True. Provide a BaseModel subclass or "
                f"instantiate PluginRegistry(strict_validation=False)."
            )
        self._tools[definition.name] = definition

    def remove_tool(self, name: str) -> None:
        """Unregister an existing tool.  Raises ``ToolNotFoundError`` if missing."""
        if name not in self._tools:
            raise ToolNotFoundError(name)
        del self._tools[name]

    # -- lookup ---------------------------------------------------------

    def get_tool(self, name: str) -> ToolDefinition:
        """Retrieve a tool definition by name.

        Raises ``ToolNotFoundError`` if the tool does not exist.
        """
        definition = self._tools.get(name)
        if definition is None:
            raise ToolNotFoundError(name)
        return definition

    def has_tool(self, name: str) -> bool:
        """Return ``True`` if a tool with *name* is registered."""
        return name in self._tools

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tool definitions (unsorted)."""
        return list(self._tools.values())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    # -- dispatch -------------------------------------------------------

    def call_tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        auth_token: str | None = None,
    ) -> Any:
        """Validate and call a tool by name.

        1. Looks up the tool (``ToolNotFoundError`` if missing).
        2. Validates *args* against the tool's schema (``ToolValidationError``).
        3. If tool is dangerous and registry has auth_token, checks auth_token matches.
        4. Invokes the handler with the validated model instance.

        Returns whatever the handler returns.
        """
        definition = self.get_tool(name)

        # Auth check for dangerous tools
        if definition.dangerous and self.auth_token is not None:
            if auth_token != self.auth_token:
                raise ToolAuthError(name)

        validated = definition.validate_args(args)

        if definition.handler is None:
            raise ToolError(f"Tool '{name}' has no handler function registered.")

        try:
            if validated is not None and definition.has_arguments:
                return definition.handler(validated)
            return definition.handler()
        except ToolError:
            raise
        except Exception as exc:
            # Wrap unexpected handler errors so callers get a consistent exception.
            raise ToolError(
                f"Tool '{name}' handler raised {type(exc).__name__}: {exc}"
            ) from exc

    # -- serialisation helpers ------------------------------------------

    def tool_descriptions(self) -> list[dict[str, Any]]:
        """Return tool descriptions suitable for LLM function-calling prompts."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.json_schema,
            }
            for t in self._tools.values()
        ]

    # -- context manager support ----------------------------------------

    def __repr__(self) -> str:
        return f"PluginRegistry(tools={self.tool_count}, strict={self._strict})"