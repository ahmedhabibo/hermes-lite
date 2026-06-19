"""hermes_lite.llm — OpenAI-compatible LLM client (local + cloud).

Supports two endpoints via the same OpenAI Python SDK:
- Local llama.cpp server: ``http://127.0.0.1:8080/v1`` (Qwen 2.5 3B Q4)
- Cloud NVIDIA NIM:       ``https://integrate.api.nvidia.com/v1``

Both clients accept the full OpenAI Chat Completions schema, including
``tools`` and ``tool_choice``. Model selection is by string prefix:

- ``local:<model>`` → local endpoint
- ``nvidia/<model>``, ``minimaxai/<model>``, etc. → cloud endpoint

Configuration via env vars (sensible defaults):
- ``HERMES_LITE_LOCAL_URL``       (default ``http://127.0.0.1:8080/v1``)
- ``HERMES_LITE_LOCAL_MODEL``     (default ``qwen2.5-3b-instruct-q4_k_m.gguf``)
- ``HERMES_LITE_CLOUD_URL``       (default ``https://integrate.api.nvidia.com/v1``)
- ``HERMES_LITE_CLOUD_MODEL``     (default ``minimaxai/minimax-m3``)
- ``HERMES_LITE_NVIDIA_API_KEY``  (NVIDIA NIM token; required for cloud)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------

LOCAL_URL_DEFAULT = "http://127.0.0.1:8080/v1"
LOCAL_MODEL_DEFAULT = "qwen2.5-3b-instruct-q4_k_m.gguf"

CLOUD_URL_DEFAULT = "https://integrate.api.nvidia.com/v1"
CLOUD_MODEL_DEFAULT = "minimaxai/minimax-m3"


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
# Chat
# ---------------------------------------------------------------------------


async def chat(req: ChatRequest) -> ChatResponse:
    """Send ``req`` to the appropriate endpoint and return a normalised response."""
    client_factory, model, tier = _pick_client_and_model(req.model)
    client = client_factory()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.tools:
        kwargs["tools"] = req.tools
        kwargs["tool_choice"] = req.tool_choice
    kwargs.update(req.extra)

    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message

    tool_calls: list[dict[str, Any]] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            # Each tc has: id, type, function={name, arguments}
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )

    return ChatResponse(
        content=msg.content or "",
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
    "_resolve_local",
    "_resolve_cloud",
    "_pick_client_and_model",
]
