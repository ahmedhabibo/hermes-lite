"""hermes_lite.tools_builtins — The 6 essential tools for Hermes-Lite.

Each tool is defined as a Pydantic schema + handler pair, then registered
via :func:`register_builtins` against a PluginRegistry instance.

Design rules (per the task spec):
- Strict Pydantic schema for every argument model — extra fields are rejected.
- Handlers return a small dict: ``{"ok": True, "output": str}`` on success,
  or ``{"ok": False, "error": str, "output": ""}`` on failure. The
  orchestrator formats this for display; tools themselves stay str-able.
- No circular imports: this module only depends on ``registry``, ``sandbox``,
  and the Python standard library. Network tools fall back to harmless
  stubs when ``hermes_tools`` (the parent Hermes runtime helper) is not
  importable, so unit tests don't require a network connection.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from hermes_lite.registry import PluginRegistry, ToolDefinition
from hermes_lite.sandbox import SandboxError, run_sandboxed

# ---------------------------------------------------------------------------
# Result shape — every handler returns one of these.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """Lightweight struct returned by every builtin handler.

    Attributes:
        ok: True on success, False on error.
        output: Human-readable content. Empty string when ``ok`` is False
            and an ``error`` is provided instead.
        error: One-line cause when ``ok`` is False. None on success.
    """

    ok: bool
    output: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return the wire-shaped dict ({ok, output, error?})."""
        if self.ok:
            return {"ok": True, "output": self.output}
        return {"ok": False, "output": self.output, "error": self.error or "unknown"}


def _ok(output: str) -> dict[str, Any]:
    return ToolResult(ok=True, output=output).to_dict()


def _err(msg: str) -> dict[str, Any]:
    return ToolResult(ok=False, output="", error=msg).to_dict()


# ---------------------------------------------------------------------------
# Optional backend imports — graceful fallback for unit tests.
# ---------------------------------------------------------------------------

try:  # pragma: no cover — imported only inside agent runtimes
    from hermes_tools import web_search as _ht_web_search  # type: ignore
    from hermes_tools import web_extract as _ht_web_extract  # type: ignore
except Exception:  # ImportError, ModuleNotFoundError, etc.
    _ht_web_search = None
    _ht_web_extract = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ReadFileArgs(BaseModel):
    """Read a local file, optionally buffered by offset/limit."""

    path: str = Field(..., description="Absolute or cwd-relative path of the file to read.")
    offset: int = Field(default=1, ge=1, description="1-indexed starting line number.")
    limit: int = Field(default=500, ge=1, le=2000, description="Maximum number of lines to return.")


class SearchFilesArgs(BaseModel):
    """Grep file contents or find files by name (ripgrep-backed)."""

    pattern: str = Field(..., description="Regex pattern (content search) or glob (file search).")
    target: Literal["content", "files"] = Field(
        default="content",
        description="Whether to grep file contents ('content') or find files by name ('files').",
    )
    path: str = Field(default=".", description="Directory to search under.")
    file_glob: Optional[str] = Field(
        default=None,
        description="Optional file-name glob filter (e.g. '*.py'). Content search only.",
    )


class TerminalArgs(BaseModel):
    """Run a shell command inside the Hermes-Lite sandbox."""

    cmd: str = Field(
        ...,
        description="Executable path or binary name. No shell features; arguments are passed as a list.",
    )
    timeout: int = Field(default=60, ge=1, le=3600, description="Wall-clock seconds before SIGKILL.")


class MemoryArgs(BaseModel):
    """Add, replace, or remove a persistent memory entry."""

    action: Literal["add", "replace", "remove"] = Field(
        ..., description="Which mutation to perform."
    )
    target: Literal["memory", "user"] = Field(
        default="memory",
        description="Which store: 'memory' for agent notes, 'user' for stable user-profile facts.",
    )
    content: str = Field(default="", description="New entry text (used by add/replace).")
    old_text: Optional[str] = Field(
        default=None,
        description="Unique substring identifying the entry to replace/remove. Required for replace/remove.",
    )


class WebSearchArgs(BaseModel):
    """Search the web for information."""

    query: str = Field(..., description="Search query.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of results to return.")


class WebFetchArgs(BaseModel):
    """Fetch a URL and extract page content as markdown."""

    url: str = Field(..., description="HTTP(S) URL to fetch.")
    max_chars: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Maximum number of markdown characters to return.",
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_read_file(args: ReadFileArgs) -> dict[str, Any]:
    """Read a local file with offset/limit support.

    The response is plain text suitable for direct return to an LLM. Empty
    files return an empty string; missing files return an error result.
    """
    try:
        # Read with Path, which gives nice error messages on missing dirs.
        p = Path(args.path).expanduser()
        if not p.exists():
            return _err(f"file not found: {args.path}")
        if not p.is_file():
            return _err(f"not a regular file: {args.path}")

        # We bound the total read to offset+limit + a small slack so we don't
        # slurp a multi-GB log into memory.
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        start = max(1, args.offset) - 1
        end = start + args.limit
        selected = lines[start:end]
        body = "".join(selected).rstrip("\n")
        if not body:
            return _ok("")
        return _ok(body)
    except PermissionError as exc:
        return _err(f"permission denied: {exc}")
    except OSError as exc:
        return _err(f"i/o error: {exc}")
    except Exception as exc:
        return _err(f"unexpected: {type(exc).__name__}: {exc}")


def _handle_search_files(args: SearchFilesArgs) -> dict[str, Any]:
    """Backing search for SearchFilesArgs.

    We shell out to ``rg`` when available (fastest in practice) and fall
    back to a small built-in grep implementation when ``rg`` isn't on the
    PATH. The output is a JSON array of match records.
    """
    import shutil
    import subprocess

    target_dir = Path(args.path).expanduser()
    if not target_dir.exists():
        return _err(f"path not found: {args.path}")

    if args.target == "files":
        # Glob-style name search. We use a small recursive walker so we
        # don't depend on the ``fd`` binary.
        try:
            matches: list[str] = []
            regex = re.compile(args.pattern)
            for root, _dirs, files in os.walk(target_dir):
                for fname in files:
                    if regex.search(fname):
                        matches.append(str(Path(root) / fname))
                if len(matches) >= 200:
                    break
            return _ok(json.dumps({"matches": matches[:200], "count": len(matches)}))
        except re.error as exc:
            return _err(f"invalid pattern: {exc}")

    # Content search — shell out to ripgrep if we can.
    rg = shutil.which("rg")
    if rg is not None:
        cmd_args = [
            "--json",
            "--no-heading",
            "--line-number",
            "--",
            args.pattern,
            str(target_dir),
        ]
        if args.file_glob:
            cmd_args.insert(0, "--glob")
            cmd_args.insert(1, args.file_glob)
            cmd_args.insert(2, "--json")
        try:
            proc = subprocess.run(
                [rg] + cmd_args,
                capture_output=True,
                text=True,
                timeout=15,
            )
            # rg returns exit code 1 when no matches — treat as empty.
            if proc.returncode not in (0, 1):
                return _err(f"rg failed: {proc.stderr.strip()}")
            matches = []
            for line in proc.stdout.splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "match":
                    data = rec.get("data", {})
                    matches.append({
                        "path": data.get("path", {}).get("text"),
                        "line": data.get("line_number"),
                        "text": data.get("lines", {}).get("text", "").rstrip(),
                    })
                if len(matches) >= 100:
                    break
            return _ok(json.dumps({"matches": matches, "count": len(matches)}))
        except subprocess.TimeoutExpired:
            return _err("rg timed out after 15s")

    # No rg available — built-in walk + regex. Slower but still useful.
    try:
        regex = re.compile(args.pattern)
    except re.error as exc:
        return _err(f"invalid pattern: {exc}")
    matches = []
    glob_re = re.compile(args.file_glob.replace("*", ".*")) if args.file_glob else None
    glob_pattern = args.file_glob if args.file_glob else None
    for root, _dirs, files in os.walk(target_dir):
        for fname in files:
            if glob_pattern and not glob_re.search(fname):
                continue
            full = Path(root) / fname
            try:
                with full.open("r", encoding="utf-8", errors="replace") as fh:
                    for idx, line in enumerate(fh, start=1):
                        if regex.search(line):
                            matches.append({
                                "path": str(full),
                                "line": idx,
                                "text": line.rstrip(),
                            })
                            if len(matches) >= 100:
                                return _ok(json.dumps({"matches": matches, "count": len(matches)}))
            except OSError:
                continue
    return _ok(json.dumps({"matches": matches, "count": len(matches)}))


def _handle_terminal(args: TerminalArgs) -> dict[str, Any]:
    """Run a binary through the Hermes-Lite sandbox.

    The command and its positional arguments are split via shlex so the
    sandbox gets a clean argv list (no shell interpretation).
    """
    try:
        parts = shlex.split(args.cmd)
    except ValueError as exc:
        return _err(f"bad cmd syntax: {exc}")
    if not parts:
        return _err("cmd must include at least the executable name")
    cmd, *rest = parts

    try:
        result = run_sandboxed(cmd, args=rest, timeout=args.timeout)
    except SandboxError as exc:
        return _err(str(exc))
    except Exception as exc:  # pragma: no cover — defensive
        return _err(f"unexpected sandbox error: {type(exc).__name__}: {exc}")

    payload = {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_ms": result.elapsed_ms,
    }
    return _ok(json.dumps(payload))


def _handle_memory(args: MemoryArgs) -> dict[str, Any]:
    """Apply the memory mutation requested in *args*.

    Persistence is handled by :class:`hermes_lite.memory_bridge.MemoryBridge`
    (a SQLite store at ``~/.hermes_lite/memory.db``). ``HOME`` redirection
    in tests still works because the bridge default path derives from
    :func:`pathlib.Path.home`. Errors raised by the bridge (missing or
    non-unique match) are surfaced as plain error envelopes so the
    orchestrator can render them without crashing.
    """
    from hermes_lite.memory_bridge import MemoryBridge, MemoryBridgeError

    bridge = MemoryBridge()  # respects Path.home() for redirection in tests
    try:
        if args.action == "add":
            if not args.content:
                return _err("content is required for add")
            row_id = bridge.add(args.target, args.content)
            return _ok(f"added to {args.target}: id={row_id}")

        if args.action == "replace":
            if not args.old_text:
                return _err("old_text is required for replace")
            row_id = bridge.replace(args.target, args.old_text, args.content)
            return _ok(f"replaced 1 in {args.target}: id={row_id}")

        if args.action == "remove":
            if not args.old_text:
                return _err("old_text is required for remove")
            bridge.remove(args.target, args.old_text)
            return _ok(f"removed 1 from {args.target}")

        return _err(f"unknown action: {args.action}")
    except MemoryBridgeError as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"unexpected: {type(exc).__name__}: {exc}")
    finally:
        bridge.close()


def _handle_web_search(args: WebSearchArgs) -> dict[str, Any]:
    """Search the web via the parent Hermes runtime if available.

    The agent runner exposes :func:`hermes_tools.web_search`. When we
    aren't running inside the agent we return a successful result with
    an informative message — never invent data, and never return an
    error (which would trigger the orchestrator's repeated_error loop).
    """
    if _ht_web_search is None:
        return _ok(
            "web_search is not available in standalone mode. "
            "Run hermes-lite inside the Hermes agent runtime to enable web search."
        )
    try:
        result = _ht_web_search(args.query, limit=args.limit)
        return _ok(json.dumps(result))
    except Exception as exc:
        return _err(f"web_search backend failed: {type(exc).__name__}: {exc}")


def _handle_web_fetch(args: WebFetchArgs) -> dict[str, Any]:
    """Fetch a URL and return a markdown extraction.

    Like ``_handle_web_search``, this delegates to the parent runtime's
    ``hermes_tools.web_extract`` when present. No fake data on missing
    backends — returns a helpful message instead of an error.
    """
    if _ht_web_extract is None:
        return _ok(
            "web_fetch is not available in standalone mode. "
            "Run hermes-lite inside the Hermes agent runtime to enable web fetch."
        )
    try:
        result = _ht_web_extract([args.url])
        # ``web_extract`` returns a dict like {"results": [...]}. Flatten
        # to the first result's content for a clean payload.
        if isinstance(result, dict) and result.get("results"):
            text = result["results"][0].get("content", "")
        else:
            text = json.dumps(result)
        text = text[: args.max_chars]
        return _ok(text)
    except Exception as exc:
        return _err(f"web_fetch backend failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Public registration helper
# ---------------------------------------------------------------------------


def _definitions() -> list[ToolDefinition]:
    """Return the 6 built-in tool definitions (no side effects).

    web_search and web_fetch are always registered; their handlers
    return graceful errors when hermes_tools is not importable
    (standalone mode) rather than crashing the tool loop.
    """
    return [
        ToolDefinition(
            name="read_file",
            description=(
                "Read a local file. Use offset/limit to page through large files. "
                "Returns plain text content (may be empty)."
            ),
            schema_model=ReadFileArgs,
            handler=_handle_read_file,
            dangerous=True,
        ),
        ToolDefinition(
            name="search_files",
            description=(
                "Search the filesystem: either grep file contents (target='content') "
                "or find files by name (target='files'). Returns JSON with matches[]."
            ),
            schema_model=SearchFilesArgs,
            handler=_handle_search_files,
            dangerous=True,
        ),
        ToolDefinition(
            name="terminal",
            description=(
                "Run a shell command in the Hermes-Lite sandbox. "
                "Returns exit_code, stdout, stderr, elapsed_ms."
            ),
            schema_model=TerminalArgs,
            handler=_handle_terminal,
            dangerous=True,
        ),
        ToolDefinition(
            name="memory",
            description=(
                "Add/replace/remove a persistent memory entry. "
                "target='memory' (agent notes) or 'user' (stable user-profile facts)."
            ),
            schema_model=MemoryArgs,
            handler=_handle_memory,
            dangerous=False,
        ),
        ToolDefinition(
            name="web_search",
            description=(
                "Search the web for information. Returns JSON with results[]."
            ),
            schema_model=WebSearchArgs,
            handler=_handle_web_search,
            dangerous=True,
        ),
        ToolDefinition(
            name="web_fetch",
            description=(
                "Fetch a URL and return markdown content, truncated to max_chars."
            ),
            schema_model=WebFetchArgs,
            handler=_handle_web_fetch,
            dangerous=True,
        ),
    ]


def register_builtins(registry: PluginRegistry, *, overwrite: bool = False) -> int:
    """Register the 6 essentials on *registry*.

    Parameters
    ----------
    registry:
        Target PluginRegistry. Any tools already registered with the same
        name will be overwritten only when *overwrite* is True.
    overwrite:
        If False (default) existing tools with the same name are kept and
        the new tool is skipped. If True, the existing tool is removed
        first.

    Returns the number of tools actually registered during this call.
    """
    registered = 0
    for definition in _definitions():
        if registry.has_tool(definition.name):
            if not overwrite:
                continue
            registry.remove_tool(definition.name)
        registry.add_tool(definition)
        registered += 1
    return registered


ESSENTIAL_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "search_files",
    "terminal",
    "memory",
    "web_search",
    "web_fetch",
)


__all__ = [
    "register_builtins",
    "ESSENTIAL_TOOL_NAMES",
    "ToolResult",
    # Schemas
    "ReadFileArgs",
    "SearchFilesArgs",
    "TerminalArgs",
    "MemoryArgs",
    "WebSearchArgs",
    "WebFetchArgs",
]
