"""hermes_lite.llm — OpenAI-compatible LLM client (cloud-first with local fallback).

Supports two endpoints via the same OpenAI Python SDK:
- Cloud NVIDIA NIM: ``https://integrate.api.nvidia.com/v1`` (default)
- Local llama.cpp server: ``http://127.0.0.1:8080/v1`` (fallback)

**Cloud-first (v0.4+):** The default model is now a cloud NIM endpoint.
Local mode is still available for offline/privacy use via the ``local:``
prefix or when no API key is configured, but new installs default to cloud.

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
- ``HERMES_LITE_LOCAL_MODEL``     (default ``qwen2.5-7b-instruct-q4_k_m.gguf``)
- ``HERMES_LITE_CLOUD_URL``       (default ``https://integrate.api.nvidia.com/v1``)
- ``HERMES_LITE_CLOUD_MODEL``     (default ``minimaxai/minimax-m3``)
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
from typing import Any, Callable, List, Literal, Optional, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

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
    # Single key from HERMES_LITE_NVIDIA_API_KEY, or fall back to NVIDIA_API_KEY
    api_key = os.environ.get(
        "HERMES_LITE_NVIDIA_API_KEY"
    ) or os.environ.get("NVIDIA_API_KEY", "")
    return url, model, api_key


def _resolve_api_key_pool() -> list[str]:
    """Return the list of available API keys for rotation.

    Reads ``HERMES_LITE_NVIDIA_API_KEYS`` (comma-separated) first.
    Falls back to ``HERMES_LITE_NVIDIA_API_KEY`` / ``NVIDIA_API_KEY``.
    Deduplicates and drops empty values.
    """
    raw = os.environ.get("HERMES_LITE_NVIDIA_API_KEYS", "").strip()
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
    else:
        single = os.environ.get("HERMES_LITE_NVIDIA_API_KEY") or os.environ.get(
            "NVIDIA_API_KEY", ""
        )
        keys = [single] if single.strip() else []
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


# ---------------------------------------------------------------------------
# Rate limiter — token bucket for NIM Free API (40 RPM default per key)
# ---------------------------------------------------------------------------

DEFAULT_RPM = 40
DEFAULT_MAX_RETRIES = 4
_BACKOFF_BASE = 1.0  # seconds
_BACKOFF_CAP = 16.0  # max single backoff


class RateLimiter:
    """Token-bucket rate limiter for cloud API calls.

    Allows burst up to the full RPM budget, then enforces the refill rate.
    Thread-safe for async use (single-threaded event loop assumed).
    """

    def __init__(self, rpm: int | None = None) -> None:
        rpm = rpm or int(os.environ.get("HERMES_LITE_RPM", str(DEFAULT_RPM)))
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
# Module-level singletons
# ---------------------------------------------------------------------------

# Initialize the key rotator and per-key rate limiters
_key_rotator = APIKeyRotator()
# One RateLimiter per API key in the pool (same order as _key_rotator._keys)
_per_key_rate_limiters: List[RateLimiter] = [
    RateLimiter() for _ in range(len(_key_rotator._keys))
]


def get_key_rotator() -> APIKeyRotator:
    """Return the module-level API key rotator (for testing/config)."""
    return _key_rotator


# For backward compatibility, we keep get_rate_limiter but it now returns
# the first per-key rate limiter if any keys are configured, otherwise a
# dummy RateLimiter.
def get_rate_limiter() -> RateLimiter:
    """Return the module-level cloud rate limiter (for testing/config)."""
    if _per_key_rate_limiters:
        return _per_key_rate_limiters[0]
    return RateLimiter()  # fallback


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
    api_key = key or _key_rotator.current
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
    max_retries = int(os.environ.get("HERMES_LITE_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

    # For local models, skip sending `tools` / `tool_choice` so llama-server
    # does NOT build a PEG grammar.
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
        # Check if all keys are exhausted BEFORE attempting
        exhausted, _ = _key_rotator.is_exhausted()
        if exhausted:
            raise AllKeysExhausted(
                len(_key_rotator._keys), _key_rotator.is_exhausted()[1]
            )

        # Get the current key and its index in the key list
        key = _key_rotator.current
        key_index = _key_rotator._index % len(_key_rotator._keys)

        # Acquire a token for this specific key's rate limiter
        await _per_key_rate_limiters[key_index].acquire()

        # Use the current key from the rotator (which should be the one we just selected)
        client = _cloud_client(key=_key_rotator.current)

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
                new_key = _key_rotator.mark_failure()
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
                    new_key = _key_rotator.mark_failure()
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
        finish_reason=msg.finish_reason or "stop",
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
    "get_rate_limiter",
    "get_key_rotator",
    "Tier",
    "tool_def",
    "chat",
    "get_per_key_rate_limiter",
]


def get_per_key_rate_limiter() -> list[RateLimiter]:
    """Return the list of per-key rate limiters (for testing)."""
    return _per_key_rate_limiters