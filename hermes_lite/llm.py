"""hermes_lite.llm — OpenAI-compatible LLM client (local + cloud).

Supports two endpoints via the same OpenAI Python SDK:
- Local llama.cpp server: ``http://127.0.0.1:8080/v1`` (Qwen 2.5 3B Q4)
- Cloud NVIDIA NIM:       ``https://integrate.api.nvidia.com/v1``

Both clients accept the full OpenAI Chat Completions schema, including
``tools`` and ``tool_choice``. Model selection is by string prefix:

- ``local:<model>`` → local endpoint
- ``nvidia/<model>``, ``minimaxai/<model>``, etc. → cloud endpoint

Local models: ``tools`` / ``tool_choice`` are **not** sent to the server
because llama.cpp auto-generates a PEG grammar from the tools schema that
small models (e.g. 3B Q4) cannot reliably follow.  Instead, tool calls are
parsed from the model's free-form text output (see :func:`parse_tool_calls_from_text`).

Configuration via env vars (sensible defaults):
- ``HERMES_LITE_LOCAL_URL``       (default ``http://127.0.0.1:8080/v1``)
- ``HERMES_LITE_LOCAL_MODEL``     (default ``qwen2.5-7b-instruct-q4_k_m.gguf``)
- ``HERMES_LITE_CLOUD_URL``       (default ``https://integrate.api.nvidia.com/v1``)
- ``HERMES_LITE_CLOUD_MODEL``     (default ``minimaxai/minimax-m3``)
- ``HERMES_LITE_NVIDIA_API_KEY``  (NVIDIA NIM token; required for cloud)
- ``HERMES_LITE_LOCAL_TOOLS``     set ``1`` to send tools to local endpoint
  (only use when the local model is large enough to handle the grammar)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------

LOCAL_URL_DEFAULT = "http://127.0.0.1:8080/v1"
LOCAL_MODEL_DEFAULT = "qwen2.5-7b-instruct-q4_k_m.gguf"

CLOUD_URL_DEFAULT = "https://integrate.api.nvidia.com/v1"
CLOUD_MODEL_DEFAULT = "minimaxai/minimax-m3"

# When True, tools/tool_choice are sent to the local endpoint just like cloud.
# Default False because small models + llama-server PEG grammar = 500 errors.
_LOCAL_TOOLS_ENABLED = os.environ.get("HERMES_LITE_LOCAL_TOOLS", "0") == "1"


def _resolve_local() -> tuple[str, str]:
    return (
        os.environ.get("HERMES_LITE_LOCAL_URL", LOCAL_URL_DEFAULT),
        os.environ.get("HERMES_LITE_LOCAL_MODEL", LOCAL_MODEL_DEFAULT),
    )


def _resolve_cloud() -> tuple[str, str, str | None]:
    url = os.environ.get("HERMES_LITE_CLOUD_URL", CLOUD_URL_DEFAULT)
    model = os.environ.get("HERMES_LITE_CLOUD_MODEL", CLOUD_MODEL_DEFAULT)
    api_key = os.environ.get(
        "HERMES_LITE_NVIDIA_API_KEY"
    ) or os.environ.get("NVIDIA_API_KEY")  # fall back to existing env
    return url, model, api_key


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Tier = Literal["local", "cloud"]

@dataclass(frozen=True)
class ChatRequest:
    """One chat completion call."""

    messages: list[dict[str, Any]]
    model: str
    tier: Tier | None = None  # auto-resolved by ``_pick_client_and_model``
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | dict[str, Any] = "auto"
    temperature: float = 0.2
    max_tokens: int = 512
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResponse:
    """Normalised response; doesn't depend on which client produced it."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    tier: Tier = "local"


# ---------------------------------------------------------------------------
# Local + cloud clients (lazy)
# ---------------------------------------------------------------------------


def _local_client() -> AsyncOpenAI:
    url, _ = _resolve_local()
    return AsyncOpenAI(base_url=url, api_key="not-needed", timeout=60.0)


def _cloud_client() -> AsyncOpenAI:
    url, _, key = _resolve_cloud()
    if not key:
        raise RuntimeError(
            "Cloud LLM requested but HERMES_LITE_NVIDIA_API_KEY (or NVIDIA_API_KEY) not set"
        )
    return AsyncOpenAI(base_url=url, api_key=key, timeout=120.0)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def _pick_client_and_model(model: str) -> tuple[Callable[[], AsyncOpenAI], str, Tier]:
    """Given a model id with optional ``local:`` or cloud prefix, return
    a *client factory*, the bare model name, and the tier.

    The factory pattern lets us defer instantiation until we know an actual
    request is going out — important for testing routing without an API key
    present.

    - ``local:<m>``                 → local client, model=``<m>``
    - starts with ``nvidia/``/``minimaxai/``/``moonshotai/``/``qwen/``/``deepseek-ai/``
      → cloud client, kept as-is
    - bare model name              → local client
    """
    if model.startswith("local:"):
        bare = model[len("local:"):]
        return _local_client, bare, "local"
    if model.startswith(("nvidia/", "minimaxai/", "moonshotai/", "qwen/", "deepseek-ai/")):
        return _cloud_client, model, "cloud"
    return _local_client, model, "local"


# ---------------------------------------------------------------------------
# Text-based tool-call parser (for local models without grammar)
# ---------------------------------------------------------------------------

# Pattern 1: tool_call code fences  ```tool_call\n{...}\n```
_FENCED_TOOL_RE = re.compile(
    r"```tool_call\s*\n\s*(\{.*?\})\s*\n\s*```",
    re.DOTALL,
)

# Pattern 2: any fenced JSON block containing "name" and "arguments"
_FENCED_JSON_RE = re.compile(
    r"```(?:json|tool)?\s*\n?\s*(\{\s*\"name\"\s*:.*?\})\s*\n?\s*```",
    re.DOTALL,
)

# Most permissive: bare JSON object on its own line with "name" key
_BARE_JSON_RE = re.compile(
    r"^\s*(\{\s*\"name\"\s*:\s*\"[^\"]+\".*\})\s*$",
    re.MULTILINE,
)

# Qwen 2.5 Instruct tool-call output — the model wraps JSON with
# blank lines above/below when emitting tool calls from the
# system-prompt tool list.  The _BARE_JSON_RE already catches
# single-line objects; this pattern catches multi-line JSON
# blocks that are separated by blank lines from surrounding text.
_QWEN_TOOL_CALL_RE = re.compile(
    r"\n\n(\{\"name\"\s*:\s*\"[^\"]+\".*?\})\n\n",
    re.DOTALL,
)


def parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from a model's free-form text output.

    Tries four patterns in order of specificity:

    1. Qwen 2.5 native format (JSON between blank lines)
    2. ``tool_call`` code fences
    3. Any fenced JSON block with ``"name"`` key
    4. Bare JSON object on its own line with ``"name"`` key

    Returns a list of dicts with keys ``id``, ``name``, and
    ``arguments`` (a JSON string).
    """
    found: list[dict[str, Any]] = []

    for pattern in (_QWEN_TOOL_CALL_RE, _FENCED_TOOL_RE, _FENCED_JSON_RE, _BARE_JSON_RE):
        for m in pattern.finditer(text):
            raw = m.group(1).strip()
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                found.append({
                    "id": f"tc_{len(found)}",
                    "name": obj["name"],
                    "arguments": json.dumps(args),
                })
        if found:
            break  # first pattern that matches wins

    return found


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def chat(req: ChatRequest) -> ChatResponse:
    """Send ``req`` to the appropriate endpoint and return a normalised response."""
    client_factory, model, tier = _pick_client_and_model(req.model)
    client = client_factory()

    # For local models, skip sending `tools` / `tool_choice` so llama-server
    # does NOT build a PEG grammar.  Small local models (e.g. 3B Q4) fail
    # to follow the grammar reliably, causing 500 errors.  Instead we parse
    # tool calls from the free-form text output after the fact.
    send_tools = bool(req.tools) and (tier != "local" or _LOCAL_TOOLS_ENABLED)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if send_tools:
        kwargs["tools"] = req.tools
        kwargs["tool_choice"] = req.tool_choice
    kwargs.update(req.extra)

    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message

    tool_calls: list[dict[str, Any]] = []
    content_text: str = msg.content or ""

    if msg.tool_calls:
        # Cloud tier or _LOCAL_TOOLS_ENABLED — native tool calls from the API.
        for tc in msg.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
    elif not send_tools and req.tools:
        # Local tier — tools were withheld; parse from text instead.
        tool_calls = parse_tool_calls_from_text(content_text)
        if tool_calls:
            # Strip tool_call fences from the displayed text.
            for _ in tool_calls:
                content_text = re.sub(
                    r"\n?```tool_call\s*\n.*?```\n?",
                    "",
                    content_text,
                    count=1,
                    flags=re.DOTALL,
                )
            content_text = content_text.strip()

    return ChatResponse(
        content=content_text,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason or "stop",
        model=model,
        usage={
            k: v
            for k, v in {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            }.items()
        },
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------


def tool_def(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build an OpenAI-compatible tool descriptor from a JSON-Schema dict.

    Example::

        tool_def(
            "read_file",
            "Read a local file.",
            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Tier",
    "chat",
    "tool_def",
    "parse_tool_calls_from_text",
    "_resolve_local",
    "_resolve_cloud",
    "_pick_client_and_model",
]
