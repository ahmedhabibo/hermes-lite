"""hermes_lite.llm — OpenAI-compatible LLM client (local-first with cloud escalation).

Supports two endpoints via the same OpenAI Python SDK:
- Local llama.cpp server: ``http://127.0.0.1:8080/v1`` (default)
- Cloud NVIDIA NIM: ``https://integrate.api.nvidia.com/v1`` (escalation)

**Local-first (v0.7+):** The default model is the local Qwen2.5-Coder-7B.
Cloud NIM is used for heavy tasks (multi-step tool-calling, large context,
complex reasoning) or when the local model fails. The router handles the
decision; this module provides the transport.

Rate limiting
-------------
NVIDIA NIM Free API enforces **40 requests per minute (RPM)**. This module
includes a token-bucket rate limiter that throttles automatically:

- 40 RPM budget (1 request per 1.5s steady-state)
- Exponential backoff with jitter on 429 responses (1s → 2s → 4s → 8s, max 16s)
- API key rotation from a comma-separated pool in ``HERMES_LITE_NVIDIA_API_KEYS``
- Per-key rate limiting: each API key has its own token bucket, so exhaustion
  of one key does not affect others.

Configuration via env vars:

- ``HERMES_LITE_LOCAL_URL``       (default ``http://127.0.0.1:8080/v1``)
- ``HERMES_LITE_LOCAL_MODEL``     (default ``Qwen2.5-Coder-7B-Instruct-IQ3_XS.gguf``)
- ``HERMES_LITE_CLOUD_URL``       (default ``https://integrate.api.nvidia.com/v1``)
- ``HERMES_LITE_CLOUD_MODEL``     (default ``z-ai/glm-5.2``)
- ``HERMES_LITE_NVIDIA_API_KEY``  (single key, or first key in pool)
- ``HERMES_LITE_NVIDIA_API_KEYS`` (comma-separated key pool for rotation)
- ``HERMES_LITE_LOCAL_TOOLS``     set ``1`` to send tools to local endpoint
- ``HERMES_LITE_RPM``             (default ``40``, NIM Free API rate limit)
- ``HERMES_LITE_MAX_RETRIES``     (default ``4``, max retry attempts on 429)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, List, Literal, Optional, Tuple

from openai import AsyncOpenAI

from hermes_lite.config import (
    CLOUD_MODEL_DEFAULT,
    CLOUD_URL_DEFAULT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RPM,
    LOCAL_MODEL_DEFAULT,
    LOCAL_URL_DEFAULT,
    get_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint resolution (via canonical config)
# ---------------------------------------------------------------------------


def _local_tools_enabled() -> bool:
    return get_config().local_tools_enabled


def _resolve_local() -> tuple[str, str]:
    cfg = get_config()
    return cfg.local_url, cfg.local_model


def _resolve_cloud() -> tuple[str, str, str | None]:
    cfg = get_config()
    return cfg.cloud_url, cfg.cloud_model, cfg.cloud_api_key or None


def _resolve_api_key_pool() -> list[str]:
    """Return the list of available API keys for rotation."""
    cfg = get_config()
    if cfg.cloud_api_keys:
        return list(cfg.cloud_api_keys)
    if cfg.cloud_api_key:
        return [cfg.cloud_api_key]
    return []


# ---------------------------------------------------------------------------
# Rate limiter — token bucket for NIM Free API (40 RPM default per key)
# ---------------------------------------------------------------------------

_BACKOFF_BASE = 1.0  # seconds
_BACKOFF_CAP = 16.0  # max single backoff


class RateLimiter:
    """Token-bucket rate limiter for cloud API calls.

    Allows burst up to the full RPM budget, then enforces the refill rate.
    Thread-safe for async use (single-threaded event loop assumed).
    """

    def __init__(self, rpm: int | None = None) -> None:
        rpm = rpm if rpm is not None else get_config().rpm
        self._rpm = max(1, rpm)
        self._max_tokens = float(self._rpm)  # bucket size = 1 minute budget
        self._refill_per_sec = self._rpm / 60.0
        self._tokens = self._max_tokens  # start full
        self._last_refill = time.monotonic()

    @property
    def rpm(self) -> int:
        return self._rpm

    async def acquire(self) -> None:
        """Wait until a token is available. Throttles to the RPM budget."""
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # No token available — sleep for the time to earn one
            wait_time = 1.0 / self._refill_per_sec  # seconds per token
            logger.debug("rate-limiter: waiting %.2fs for token", wait_time)
            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_per_sec)
        self._last_refill = now


# ---------------------------------------------------------------------------
# API key rotation
# ---------------------------------------------------------------------------


class AllKeysExhausted(Exception):
    """Raised when all API keys in the pool are rate-limited or auth-failed.

    This is a recoverable condition — the orchestrator should catch this
    and fall back to local model or return a graceful error message.
    """

    __slots__ = ("keys_tried", "cooldown_remaining")

    def __init__(self, keys_tried: int, cooldown_remaining: float):
        self.keys_tried = keys_tried
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"All {keys_tried} API keys exhausted. "
            f"Earliest key available in {cooldown_remaining:.0f}s."
        )


class APIKeyRotator:
    """Round-robin API key rotation with failure tracking.

    On a 401/403/429 response, the current key is marked as "cooling down"
    and the next key in the pool is selected. Keys cool down for 60 seconds.
    """

    COOLDOWN_SECONDS = 60.0

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = keys if keys is not None else _resolve_api_key_pool()
        self._index = 0
        self._cooldowns: dict[int, float] = {}  # index → expiry monotonic time

    @property
    def current(self) -> str | None:
        """Return the current active key, or None if pool is empty.

        This property has the side effect of advancing ``self._index`` past
        any keys that are currently in cooldown, so that the next call to
        ``current`` or ``mark_failure`` starts from the first non-cooled-down
        key (or wraps around if all are cooled down).
        """
        if not self._keys:
            return None
        now = time.monotonic()
        for _ in range(len(self._keys)):
            idx = self._index % len(self._keys)
            expiry = self._cooldowns.get(idx, 0.0)
            if now >= expiry:
                return self._keys[idx]
            self._index += 1
        # All in cooldown — return the first one anyway
        return self._keys[0]

    def mark_failure(self) -> str | None:
        """Mark the current key as failing, rotate to next.

        Returns the new active key, or None if pool is exhausted.
        """
        if not self._keys:
            return None
        current_idx = self._index % len(self._keys)
        self._cooldowns[current_idx] = time.monotonic() + self.COOLDOWN_SECONDS
        self._index += 1
        return self.current

    def is_exhausted(self) -> tuple[bool, float]:
        """Check if all keys are currently in cooldown.

        Returns (exhausted, cooldown_remaining_seconds).
        If exhausted, cooldown_remaining is the time until the earliest key recovers.
        """
        if not self._keys:
            return True, 0.0
        now = time.monotonic()
        earliest_expiry = float("inf")
        all_in_cooldown = True
        for idx in range(len(self._keys)):
            expiry = self._cooldowns.get(idx, 0.0)
            if now >= expiry:
                # This key is not in cooldown (available or already expired)
                all_in_cooldown = False
            else:
                # This key is still in cooldown
                if expiry < earliest_expiry:
                    earliest_expiry = expiry
        if all_in_cooldown:
            # All keys are in cooldown
            return True, max(0.0, earliest_expiry - now)
        return False, 0.0

    def reset(self) -> None:
        """Clear all cooldowns."""
        self._cooldowns.clear()
        self._index = 0


# ---------------------------------------------------------------------------
# Module-level singletons (lazy initialization — _load_env() must run first)
# ---------------------------------------------------------------------------

_key_rotator: APIKeyRotator | None = None
_per_key_rate_limiters: List[RateLimiter] | None = None


def _ensure_rotator() -> APIKeyRotator:
    """Lazily initialize the key rotator and per-key rate limiters.

    This is called on first use (not at import time) so that .env loading
    via _load_env() in __main__ has a chance to populate env vars first.
    """
    global _key_rotator, _per_key_rate_limiters
    if _key_rotator is None:
        _key_rotator = APIKeyRotator()
        _per_key_rate_limiters = [RateLimiter() for _ in range(len(_key_rotator._keys))]
    return _key_rotator


def get_key_rotator() -> APIKeyRotator:
    """Return the module-level API key rotator (for testing/config)."""
    return _ensure_rotator()


def get_per_key_rate_limiters() -> List[RateLimiter]:
    """Return the list of per-key rate limiters."""
    _ensure_rotator()
    return _per_key_rate_limiters  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Tier = Literal["local", "cloud"]


@dataclass(frozen=True)
class ChatRequest:
    """One chat completion call."""

    messages: list[dict[str, Any]]
    model: str
    tier: Optional[Tier] = None  # auto-resolved by ``_pick_client_and_model``
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


def _cloud_client(key: str | None = None) -> AsyncOpenAI:
    """Create a cloud NIM client, optionally with a specific API key.

    If *key* is omitted, uses the key rotator's current key.
    """
    url, _, _ = _resolve_cloud()
    api_key = key or _ensure_rotator().current
    if not api_key:
        raise RuntimeError(
            "Cloud LLM requested but HERMES_LITE_NVIDIA_API_KEY "
            "(or NVIDIA_API_KEY) not set"
        )
    return AsyncOpenAI(base_url=url, api_key=api_key, timeout=120.0)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

# Cloud prefixes — any model ID starting with these goes to the cloud endpoint.
_CLOUD_PREFIXES = (
    "nvidia/",
    "minimaxai/",
    "moonshotai/",
    "qwen/",
    "deepseek-ai/",
    "stepfun-ai/",
    "z-ai/",
)


def _pick_client_and_model(model: str) -> tuple[Callable[[], AsyncOpenAI], str, Tier]:
    """Given a model id with optional ``local:`` or cloud prefix, return
    a *client factory*, the bare model name, and the tier.

    The factory pattern lets us defer instantiation until we know an actual
    request is going out — important for testing routing without an API key
    present.

    - ``local:<m>``                 → local client, model=``<m>``
    - starts with a cloud prefix    → cloud client, kept as-is
    - bare model name (e.g. gguf)   → local client
    - bare cloud-style name         → cloud client (fallback heuristic)
    """
    if model.startswith("local:"):
        bare = model[len("local:"):]
        return _local_client, bare, "local"
    if model.startswith(_CLOUD_PREFIXES):
        return _cloud_client, model, "cloud"
    # Bare name — if it contains a / it's likely a cloud model id
    if "/" in model:
        return _cloud_client, model, "cloud"
    return _local_client, model, "local"


# -------------------------------------------------------------------------__
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
    r"^\s*(\{\s*\"name\"\s*:.*?\})\s*$",
    re.MULTILINE,
)

# Qwen 2.5 Instruct tool-call output — the model wraps JSON with
# blank lines above/below when emitting tool calls from the
# system-prompt tool list.  The _BARE_JSON_RE already catches
# single-line objects; this pattern catches multi-line JSON
# blocks that are separated by blank lines from surrounding text.
_QWEN_TOOL_CALL_RE = re.compile(
    r"\n\n(\{\"name\"\s*:.*?\})\n\n",
    re.DOTALL,
)

# Qwen2.5-Coder XML-style tool calls:
#   <function name="get_weather" arguments='{"city": "Cairo"}' />
#   <function="get_weather">{...}</function>
_QWEN_XML_TOOL_RE = re.compile(
    r'<function\s+name="([^"]+)"\s+arguments=\'([^\']+)\'\s*/?>'
    r'|<function="([^"]+)">(.*?)</function>',
    re.DOTALL,
)


def parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from a model's free-form text output.

    Tries five patterns in order of specificity:

    1. Qwen2.5-Coder XML-style (``<function name="..." arguments='...' />``)
    2. Qwen 2.5 native format (JSON between blank lines)
    3. ``tool_call`` code fences
    4. Any fenced JSON block with ``"name"`` key
    5. Bare JSON object on its own line with ``"name"`` key

    Returns a list of dicts with keys ``id``, ``name``, and
    ``arguments`` (a JSON string).
    """
    found: list[dict[str, Any]] = []

    # 1. Qwen2.5-Coder XML-style tool calls
    for m in _QWEN_XML_TOOL_RE.finditer(text):
        name = m.group(1) or m.group(3)
        args_raw = m.group(2) or m.group(4)
        if not name:
            continue
        # Parse arguments — could be JSON or key=value pairs
        try:
            args = json.loads(args_raw) if args_raw else {}
        except (json.JSONDecodeError, ValueError):
            # Try wrapping in braces if model output bare key: value
            try:
                args = json.loads("{" + args_raw + "}")
            except (json.JSONDecodeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {"value": args} if args else {}
        found.append({
            "id": f"tc_{len(found)}",
            "name": name,
            "arguments": json.dumps(args),
        })
    if found:
        return found

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
                found.append(
                    {
                        "id": f"tc_{len(found)}",
                        "name": obj["name"],
                        "arguments": json.dumps(args),
                    }
                )
        if found:
            break  # first pattern that matches wins

    return found


# ---------------------------------------------------------------------------
# Tool definition helper
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


# ---------------------------------------------------------------------------
# Chat — with rate limiting, retry, and key rotation
# ---------------------------------------------------------------------------


async def chat(req: ChatRequest) -> ChatResponse:
    """Send ``req`` to the appropriate endpoint and return a normalised response.

    For cloud requests:
    - Respects the per-key 40 RPM rate limit via token bucket
    - Retries on 429/500/502/503 with exponential backoff + jitter
    - Rotates API keys on auth/rate-limit errors
    - Falls back to local model if all API keys are exhausted
    """
    client_factory, model, tier = _pick_client_and_model(req.model)
    max_retries = get_config().max_retries

    # For local models, skip sending `tools` / `tool_choice` so llama-server
    # does NOT build a PEG grammar.
    send_tools = bool(req.tools) and (tier != "local" or _local_tools_enabled())

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

    # Cloud requests: rate limit + retry logic
    if tier == "cloud":
        try:
            return await _chat_cloud_with_retry(
                client_factory, kwargs, model, tier, req, max_retries
            )
        except AllKeysExhausted as exc:
            logger.warning("All API keys exhausted (%s), falling back to local", exc)
            # Fall back to local model
            local_url, local_model = _resolve_local()
            local_client = AsyncOpenAI(base_url=local_url, api_key="not-needed", timeout=60.0)
            kwargs["model"] = local_model
            resp = await local_client.chat.completions.create(**kwargs)
            return _parse_response(resp, local_model, "local", req, send_tools=False)

    # Local requests: simple call, no rate limit needed
    client = client_factory()
    resp = await client.chat.completions.create(**kwargs)
    return _parse_response(resp, model, tier, req, send_tools=False)


async def _chat_cloud_with_retry(
    client_factory: Callable[[], AsyncOpenAI],
    kwargs: dict[str, Any],
    model: str,
    tier: Tier,
    req: ChatRequest,
    max_retries: int,
) -> ChatResponse:
    """Cloud chat with per-key rate limiting, exponential backoff + jitter, and key rotation."""
    from openai import (
        APIStatusError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )

    send_tools = bool(req.tools)
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        rotator = _ensure_rotator()
        limiters = get_per_key_rate_limiters()
        # Check if all keys are exhausted BEFORE attempting
        exhausted, _ = rotator.is_exhausted()
        if exhausted:
            raise AllKeysExhausted(
                len(rotator._keys), rotator.is_exhausted()[1]
            )

        # Get the current key and its index in the key list
        key = rotator.current
        key_index = rotator._index % len(rotator._keys)

        # Acquire a token for this specific key's rate limiter
        await limiters[key_index].acquire()

        # Use the current key from the rotator (which should be the one we just selected)
        client = _cloud_client(key=rotator.current)

        try:
            resp = await client.chat.completions.create(**kwargs)
            # Success: leave the rotator's index as-is (so next call starts from here)
            return _parse_response(resp, model, tier, req, send_tools=send_tools)
        except RateLimitError as exc:
            # 429 — rate limited; back off with jitter, rotate key
            last_exc = exc
            if attempt < max_retries:
                # Exponential backoff with full jitter
                backoff = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
                # Full jitter: random delay between 0 and backoff
                jittered_backoff = random.uniform(0, backoff)
                logger.warning(
                    "cloud 429 (attempt %d/%d), backoff %.1fs (jittered), rotating key",
                    attempt + 1,
                    max_retries + 1,
                    jittered_backoff,
                )
                new_key = rotator.mark_failure()
                if new_key:
                    logger.info("rotated to key ending ...%s", new_key[-4:])
                await asyncio.sleep(jittered_backoff)
                continue
        except (InternalServerError, APIConnectionError) as exc:
            # 500/502/503 or connection error — back off with jitter, same key
            last_exc = exc
            if attempt < max_retries:
                backoff = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
                jittered_backoff = random.uniform(0, backoff)
                logger.warning(
                    "cloud %s (attempt %d/%d), backoff %.1fs (jittered)",
                    type(exc).__name__,
                    attempt + 1,
                    max_retries + 1,
                    jittered_backoff,
                )
                await asyncio.sleep(jittered_backoff)
                continue
        except APIStatusError as exc:
            # 401/403 — auth error; rotate key with jittered backoff
            if exc.status_code in (401, 403):
                last_exc = exc
                if attempt < max_retries:
                    backoff = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
                    jittered_backoff = random.uniform(0, backoff)
                    logger.warning(
                        "cloud auth error %d (attempt %d/%d), backoff %.1fs (jittered), rotating key",
                        exc.status_code,
                        attempt + 1,
                        max_retries + 1,
                        jittered_backoff,
                    )
                    new_key = rotator.mark_failure()
                    if new_key:
                        logger.info("rotated to key ending ...%s", new_key[-4:])
                    await asyncio.sleep(jittered_backoff)
                    continue
            # Unrecoverable — re-raise
            raise
        finally:
            # Restore the original index (in case we changed it above)
            # Note: we did not change _key_rotator._index in this block,
            # but we leave this for safety if we ever modify it.
            pass

    # All retries exhausted
    raise last_exc or RuntimeError(
        f"cloud request failed after {max_retries + 1} attempts"
    )


def _parse_response(
    resp: Any,
    model: str,
    tier: Tier,
    req: ChatRequest,
    send_tools: bool,
) -> ChatResponse:
    """Parse an OpenAI response into a ChatResponse, handling tool calls."""
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

    return ChatResponse(
        content=content_text,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason or "stop",
        model=model,
        usage=dict(resp.usage) if resp.usage else {},
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "AllKeysExhausted",
    "RateLimiter",
    "get_key_rotator",
    "get_per_key_rate_limiters",
    "Tier",
    "tool_def",
    "chat",
    "chat_stream",
]


# ---------------------------------------------------------------------------
# Streaming support
# ---------------------------------------------------------------------------

async def chat_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    """Stream LLM response token-by-token.

    Returns an async generator that yields chunks (deltas) as they arrive.
    Works with both cloud (NIM) and local (llama-server) endpoints.

    Usage::

        async for chunk in chat_stream(req):
            print(chunk, end="", flush=True)

    For cloud requests, rate limiting and key rotation are applied
    the same as ``chat()``. If all keys are exhausted, falls back to
    local automatically.

    Yields
    ------
    str
        Text deltas as they arrive from the LLM.
    """
    client_factory, model, tier = _pick_client_and_model(req.model)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": True,
    }
    # Don't send tools to local unless explicitly enabled
    send_tools = bool(req.tools) and (tier != "local" or _local_tools_enabled())
    if send_tools:
        kwargs["tools"] = req.tools
        kwargs["tool_choice"] = req.tool_choice
    kwargs.update(req.extra)

    if tier == "cloud":
        max_retries = get_config().max_retries
        try:
            async for text in _chat_cloud_stream_with_retry(
                client_factory, kwargs, model, tier, req, max_retries
            ):
                yield text
        except AllKeysExhausted:
            logger.warning("All API keys exhausted during streaming, falling back to local")
            local_url, local_model = _resolve_local()
            local_client = AsyncOpenAI(base_url=local_url, api_key="not-needed", timeout=60.0)
            kwargs["model"] = local_model
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            stream = await local_client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
    else:
        client = client_factory()
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


async def _chat_cloud_stream_with_retry(
    client_factory: Callable[[], AsyncOpenAI],
    kwargs: dict[str, Any],
    model: str,
    tier: Tier,
    req: ChatRequest,
    max_retries: int,
) -> AsyncGenerator[str, None]:
    """Cloud streaming with per-key rate limiting, backoff, and key rotation."""
    from openai import (
        APIStatusError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )

    rotator = get_key_rotator()
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        key_idx = rotator.next_key() if rotator else 0
        per_key_limiters = get_per_key_rate_limiters()
        limiter = per_key_limiters[key_idx] if key_idx < len(per_key_limiters) else None

        if limiter:
            await limiter.acquire()

        try:
            client = client_factory()
            keys = rotator.get_keys() if rotator else []
            if keys and key_idx < len(keys):
                client = AsyncOpenAI(
                    base_url=kwargs.get("base_url", CLOUD_URL_DEFAULT),
                    api_key=keys[key_idx],
                    timeout=60.0,
                )

            stream = await client.chat.completions.create(**kwargs)
            collected = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    collected += delta
                    yield delta
            return

        except (RateLimitError, InternalServerError) as exc:
            last_exc = exc
            if rotator:
                rotator.mark_failed(key_idx)
            backoff = min(2 ** attempt, 16)
            jitter = random.uniform(0, backoff)
            await asyncio.sleep(jitter)
        except (APIConnectionError, APIStatusError) as exc:
            last_exc = exc
            await asyncio.sleep(min(2 ** attempt, 16))
        except AllKeysExhausted:
            raise
        except Exception as exc:
            last_exc = exc
            break

    if last_exc:
        raise last_exc


def get_per_key_rate_limiter() -> list[RateLimiter]:
    """Return the list of per-key rate limiters (for testing)."""
    return get_per_key_rate_limiters()


# Backward compat 
def get_rate_limiter() -> RateLimiter:
    """Return the first per-key rate limiter (for backward compat)."""
    limiters = get_per_key_rate_limiters()
    if limiters:
        return limiters[0]
    return RateLimiter()