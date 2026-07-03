"""hermes_lite.router — Tier routing controller (cloud-first with local fallback).

Decides which LLM tier (``cloud`` or ``local``) to use for a given prompt
based on a deterministic complexity score. **Cloud-first since v0.4:**
the default fallback chain starts with NIM cloud models; local is a fallback
for offline/privacy use.

Complexity score (0.0 - 1.0, clamped) — weighted sum of four signals:

  * prompt length (0.2 weight, normalised at 2_000 chars)
  * context token count (0.4 weight, normalised at 4_000 tokens)
  * history turns used (0.2 weight, normalised at 4 turns)
  * keyword heuristic (0.2 weight, 0.0-1.0): tokens like ``refactor``,
    ``architect``, ``debug``, ``multi-step`` each contribute up to 0.25.

A configurable threshold (env ``LITE_LOCAL_MAX_COMPLEXITY``, default 0.3)
splits local-vs-cloud: <= threshold ⇒ ``local``; above ⇒ ``cloud``.

With a cloud-first chain (the default) the threshold still works the same
way — *low* complexity stays on the preferred (cloud) model, high complexity
uses a heavier cloud model from the chain. The ``local`` tier only activates
when the chain explicitly starts with a ``local:`` entry.

Escalation
----------
``LiteRouter`` is *stateful* per request lifecycle. The caller registers
the route it took (``record_outcome``); on repeated local failures or
malformed ``tool_calls`` JSON, the next ``route`` call is forced to
``cloud`` until ``reset()`` or a successful call clears the escalation
counter.

Config via env (overridable at construction):

* ``LITE_LOCAL_MAX_COMPLEXITY``              (default ``0.3``)
* ``LITE_ESCALATE_AFTER_FAILURES``           (default ``2``)
* ``LITE_LARGE_PROMPT_CHARS``                (default ``2000``)
* ``LITE_LARGE_CONTEXT_TOKENS``              (default ``4000``)
* ``LITE_LARGE_HISTORY_TURNS``               (default ``4``)
* ``LITE_FALLBACK_CHAIN``                    (default
  ``"minimaxai/minimax-m3,moonshotai/kimi-k2.6,qwen/qwen3.5-397b-a17b,deepseek-ai/deepseek-v4-flash"``)

Public API:
* ``LiteRouter``
* ``RoutingDecision``  (dataclass returned by ``route``)
* ``parse_fallback_chain`` (helper)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterable

from hermes_lite.llm import Tier

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MAX_COMPLEXITY = 0.3
DEFAULT_ESCALATE_AFTER_FAILURES = 2
DEFAULT_LARGE_PROMPT_CHARS = 2_000
DEFAULT_LARGE_CONTEXT_TOKENS = 4_000
DEFAULT_LARGE_HISTORY_TURNS = 4

# Each keyword contributes up to 1/4 of the 0.2 keyword weight.
_KEYWORDS: tuple[str, ...] = (
    "refactor",
    "architect",
    "debug",
    "multi-step",
    "redesign",
    "optimize",
    "explain in detail",
)
_KEYWORD_HIT_WEIGHT = 0.25  # per keyword match; 4 hits → 1.0 → 0.2 weight fully consumed

# Lead-in intent patterns that always escalate regardless of the score.
# These represent user requests that clearly warrant a stronger model
# even when the literal prompt is short — matches the spec's "boost"
# intent and the acceptance criterion "refactor this 200-line script →
# cloud". Matched case-insensitively at message start (after stripping
# a single leading verb like "please" / "can you").
_INTENT_PREFIX: tuple[str, ...] = (
    "refactor",
    "rewrite",
    "redesign",
    "architect ",
    "rearchitect",
    "migrate ",
    "port ",
    "build me a",
    "implement a",
    "ship a",
    "design a",
    "from scratch",
    "end-to-end",
)

# Cloud-first NIM fallback chain (v0.6+):
# 1. z-ai/glm-5.2                  — best general-purpose (new primary)
# 2. minimaxai/minimax-m3          — strong general-purpose
# 3. moonshotai/kimi-k2.6          — strong reasoning
# 4. qwen/qwen3.5-397b-a17b        — MoE, efficient
# 5. deepseek-ai/deepseek-v4-flash — fast fallback
DEFAULT_FALLBACK_CHAIN = (
    "z-ai/glm-5.2,"
    "minimaxai/minimax-m3,"
    "moonshotai/kimi-k2.6,"
    "qwen/qwen3.5-397b-a17b,"
    "deepseek-ai/deepseek-v4-flash"
)


# ---------------------------------------------------------------------------
# Fallback chain parsing
# ---------------------------------------------------------------------------


def parse_fallback_chain(raw: str | Iterable[str]) -> list[str]:
    """Parse the ``LITE_FALLBACK_CHAIN`` env value into a clean list.

    Accepts either a single comma-separated string or any iterable of
    strings. Whitespace is stripped; empty fragments are dropped; the
    first entry is treated as the *preferred* model.

    >>> parse_fallback_chain("a,b , c")
    ['a', 'b', 'c']
    """
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = list(raw)
    return [p.strip() for p in parts if p and p.strip()]

# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingDecision:
    """One decision returned by :meth:`LiteRouter.route`.

    The orchestrator uses ``model_id`` to build a :class:`ChatRequest`
    and ``tier`` to pick the client endpoint.
    """

    model_id: str
    tier: Tier
    complexity_score: float
    reason: str
    fell_back: bool = False  # True when the decision was forced by escalation


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class LiteRouter:
    """Deterministic complexity-based tier router.

    The router is intentionally simple — heuristics, not ML. It computes
    a 0.0-1.0 score from four observable signals and compares against a
    configurable threshold. State (failure counters, last failure reason)
    is held until ``reset()`` is called.

    With a cloud-first chain (the v0.4+ default), the preferred entry is
    a cloud model. Low-complexity requests stay on the preferred cloud
    model; high-complexity ones try the next heavier model in the chain.
    Local entries in the chain (``local:…``) are only used when the chain
    explicitly includes them.
    """

    def __init__(
        self,
        *,
        local_max_complexity: float | None = None,
        escalate_after_failures: int | None = None,
        large_prompt_chars: int | None = None,
        large_context_tokens: int | None = None,
        large_history_turns: int | None = None,
        fallback_chain: list[str] | None = None,
    ) -> None:
        # Defaults overridable via env then explicit kwargs.
        self.local_max_complexity = float(
            local_max_complexity
            if local_max_complexity is not None
            else os.environ.get("LITE_LOCAL_MAX_COMPLEXITY", DEFAULT_LOCAL_MAX_COMPLEXITY)
        )
        self.escalate_after_failures = int(
            escalate_after_failures
            if escalate_after_failures is not None
            else os.environ.get(
                "LITE_ESCALATE_AFTER_FAILURES", DEFAULT_ESCALATE_AFTER_FAILURES
            )
        )
        self.large_prompt_chars = int(
            large_prompt_chars
            if large_prompt_chars is not None
            else os.environ.get("LITE_LARGE_PROMPT_CHARS", DEFAULT_LARGE_PROMPT_CHARS)
        )
        self.large_context_tokens = int(
            large_context_tokens
            if large_context_tokens is not None
            else os.environ.get("LITE_LARGE_CONTEXT_TOKENS", DEFAULT_LARGE_CONTEXT_TOKENS)
        )
        self.large_history_turns = int(
            large_history_turns
            if large_history_turns is not None
            else os.environ.get("LITE_LARGE_HISTORY_TURNS", DEFAULT_LARGE_HISTORY_TURNS)
        )

        chain_raw = (
            fallback_chain
            if fallback_chain is not None
            else parse_fallback_chain(
                os.environ.get("LITE_FALLBACK_CHAIN", DEFAULT_FALLBACK_CHAIN)
            )
        )
        # If the user only supplied one entry, we still allow it — the
        # engine will surface an error if a forced cloud call is later
        # made without credentials.
        if not chain_raw:
            chain_raw = list(parse_fallback_chain(DEFAULT_FALLBACK_CHAIN))
        self.fallback_chain: list[str] = chain_raw

        # Mutable state — survives across route() calls within a single
        # orchestrator session. Cleared by reset().
        self._consecutive_local_failures: int = 0
        # Cloud escalation tracking (mirrors local tracking for cloud failures).
        self._consecutive_cloud_failures: int = 0
        self._fallback_index: int = 0  # position in chain for cloud fallback

    # -- complexity --------------------------------------------------------

    def complexity(self, prompt: str, context_tokens: int, history_turns: int) -> float:
        """Return the 0.0-1.0 complexity score for a single request.

        Each component is normalised against the corresponding "large"
        threshold so a request hitting all three ceilings maxes out at
        1.0 before the keyword term is added. We *clamp* the keyword
        contribution at 1.0 for the same reason.

        Public for testing.
        """
        if not isinstance(prompt, str):
            prompt = str(prompt)
        prompt_len = len(prompt)

        prompt_norm = min(prompt_len / max(self.large_prompt_chars, 1), 1.0)
        context_norm = min(max(context_tokens, 0) / max(self.large_context_tokens, 1), 1.0)
        history_norm = min(max(history_turns, 0) / max(self.large_history_turns, 1), 1.0)
        keyword_score = self._keyword_score(prompt)

        score = (
            0.2 * prompt_norm
            + 0.4 * context_norm
            + 0.2 * history_norm
            + 0.2 * keyword_score
        )
        # Final clamp — guards against rounding edge cases where
        # kwargs.weights sum to slightly more than 1.0 in pathological
        # normalisation regimes.
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _keyword_score(prompt: str) -> float:
        """Return a 0.0-1.0 keyword heuristic score.

        Looks for any of the configured keywords as case-insensitive whole
        words (``multi-step`` matches via literal hyphen). Each match
        adds ``_KEYWORD_HIT_WEIGHT``; clamps at 1.0.
        """
        if not prompt:
            return 0.0
        lowered = prompt.lower()
        hits = 0
        for kw in _KEYWORDS:
            # Use word-boundary regex for short tokens; allow hyphenated
            # tokens like ``multi-step`` to match via substring.
            if "-" in kw or " " in kw:
                if kw in lowered:
                    hits += 1
            else:
                if re.search(rf"\b{re.escape(kw)}\b", lowered):
                    hits += 1
        return min(hits * _KEYWORD_HIT_WEIGHT, 1.0)

    @staticmethod
    def _strip_hedging_prefix(prompt: str) -> str:
        """Strip a single leading conversational hedge like "please" or
        "can you" so the intent-prefix check sees the actual first verb.
        """
        hedgers = (
            "please ",
            "pls ",
            "can you ",
            "could you ",
            "i need to ",
            "i want to ",
            "let's ",
            "let us ",
            "i'd like to ",
            "would you ",
        )
        lowered = prompt.lower().lstrip()
        for h in hedgers:
            if lowered.startswith(h):
                return lowered[len(h):]
        return lowered

    @classmethod
    def _is_complex_intent(cls, prompt: str) -> bool:
        """True when the prompt opens with a verb that warrants a heavier
        model in the fallback chain.

        Examples that return True: ``"refactor …"``, ``"please rewrite …"``,
        ``"Architect a new …"``, ``"rearchitect …"``, ``"redesign …"``.
        Examples that return False: ``"find X"``, ``"show me …"``, ``"what
        is …"``.
        """
        if not prompt:
            return False
        body = cls._strip_hedging_prefix(prompt)
        for prefix in _INTENT_PREFIX:
            if body.startswith(prefix):
                return True
        return False

    # -- tier selection ----------------------------------------------------

    def _pick_preferred(self) -> tuple[str, Tier]:
        """Return the (model_id, tier) of the *preferred* (first) chain entry."""
        first = self.fallback_chain[0]
        tier: Tier = "local" if first.startswith("local:") or "/" not in first else "cloud"
        return first, tier

    def _pick_escalation(self) -> tuple[str, Tier]:
        """Pick the first cloud entry in the chain; fall back to the
        preferred entry if no cloud candidate exists."""
        for entry in self.fallback_chain:
            tier: Tier = "local" if entry.startswith("local:") or "/" not in entry else "cloud"
            if tier == "cloud":
                return entry, tier
        # All-local chain — return preferred as-is with a synthesised
        # cloud label so the orchestrator knows to surface the issue.
        entry, _ = self._pick_preferred()
        return entry, "cloud"

    def _pick_next_cloud_fallback(self) -> tuple[str, Tier] | None:
        """Pick the next cloud model in the chain after _fallback_index.

        Returns None if we've exhausted the cloud entries in the chain.
        Advances _fallback_index so repeated calls walk the chain.
        """
        cloud_entries = [
            (i, e) for i, e in enumerate(self.fallback_chain)
            if not e.startswith("local:") and "/" in e
        ]
        if not cloud_entries:
            return None
        # Start from current fallback index, find next
        for idx, entry in cloud_entries:
            if idx >= self._fallback_index:
                self._fallback_index = idx + 1
                return entry, "cloud"
        # Exhausted — wrap around to first cloud entry
        self._fallback_index = cloud_entries[0][0] + 1
        return cloud_entries[0][1], "cloud"

    # -- public ------------------------------------------------------------

    def route(
        self,
        prompt: str,
        context_tokens: int,
        history_turns: int,
    ) -> RoutingDecision:
        """Decide which tier + model handles this request.

        Cloud-first logic (v0.4+):

        1. **Escalation override** — if consecutive failures exceed the
           threshold, advance to the next model in the chain.
        2. **Score + intent** — for cloud-first chains, the preferred
           model handles most requests. High complexity or complex-intent
           picks the next heavier cloud model in the chain.
        3. **Local fallback** — if the chain starts with a local entry,
           the old behaviour applies (score > threshold → cloud escalation).
        """
        # 1. Escalation override (highest priority)
        if self._consecutive_cloud_failures >= self.escalate_after_failures:
            next_model = self._pick_next_cloud_fallback()
            if next_model:
                model_id, tier = next_model
            else:
                model_id, tier = self._pick_escalation()
            score = self.complexity(prompt, context_tokens, history_turns)
            return RoutingDecision(
                model_id=model_id,
                tier=tier,
                complexity_score=score,
                reason=(
                    f"escalated: {self._consecutive_cloud_failures} consecutive "
                    f"failures >= threshold {self.escalate_after_failures}"
                ),
                fell_back=True,
            )

        if self._consecutive_local_failures >= self.escalate_after_failures:
            model_id, tier = self._pick_escalation()
            score = self.complexity(prompt, context_tokens, history_turns)
            return RoutingDecision(
                model_id=model_id,
                tier=tier,
                complexity_score=score,
                reason=(
                    f"escalated: {self._consecutive_local_failures} consecutive "
                    f"local failures >= threshold {self.escalate_after_failures}"
                ),
                fell_back=True,
            )

        # 2. Score-based decision (preferred model from chain head)
        score = self.complexity(prompt, context_tokens, history_turns)
        model_id, tier = self._pick_preferred()

        if tier == "local":
            # If the preferred model is local and the score exceeds the
            # threshold we escalate to the first cloud entry in the chain.
            if score > self.local_max_complexity:
                cloud_model, cloud_tier = self._pick_escalation()
                return RoutingDecision(
                    model_id=cloud_model,
                    tier=cloud_tier,
                    complexity_score=score,
                    reason=(
                        f"complexity {score:.3f} > threshold {self.local_max_complexity} "
                        f"\u2192 cloud (preferred was local)"
                    ),
                    fell_back=True,
                )

            # 3. Intent-prefix override (still local-preferred + still below
            #    threshold). Complex verbs escalate to cloud.
            if self._is_complex_intent(prompt):
                cloud_model, cloud_tier = self._pick_escalation()
                return RoutingDecision(
                    model_id=cloud_model,
                    tier=cloud_tier,
                    complexity_score=score,
                    reason=(
                        f"complex intent prefix detected "
                        f"\u2192 cloud (preferred was local)"
                    ),
                    fell_back=True,
                )

            return RoutingDecision(
                model_id=model_id,
                tier="local",
                complexity_score=score,
                reason=f"complexity {score:.3f} <= threshold {self.local_max_complexity}",
                fell_back=False,
            )

        else:
            # Preferred is cloud (cloud-first chain).
            # High complexity → advance to next cloud model in chain.
            if score > self.local_max_complexity or self._is_complex_intent(prompt):
                next_model = self._pick_next_cloud_fallback()
                if next_model:
                    next_id, next_tier = next_model
                    return RoutingDecision(
                        model_id=next_id,
                        tier=next_tier,
                        complexity_score=score,
                        reason=(
                            f"complexity {score:.3f} > threshold {self.local_max_complexity} "
                            f"\u2192 next cloud fallback"
                        ),
                        fell_back=True,
                    )
                # Exhausted chain — stay on preferred
                return RoutingDecision(
                    model_id=model_id,
                    tier="cloud",
                    complexity_score=score,
                    reason=(
                        f"complexity {score:.3f} > threshold but no more "
                        f"cloud fallbacks; staying on preferred"
                    ),
                    fell_back=False,
                )

            return RoutingDecision(
                model_id=model_id,
                tier="cloud",
                complexity_score=score,
                reason=(
                    f"cloud-first chain; complexity {score:.3f}"
                ),
                fell_back=False,
            )

    # -- bookkeeping -------------------------------------------------------

    def record_outcome(
        self,
        decision: RoutingDecision,
        *,
        succeeded: bool,
        tool_calls_malformed: bool = False,
    ) -> None:
        """Update escalation state after a routed call returns.

        A successful call resets the relevant counter. A failure on the
        matching tier increments it. Cloud failures now also trigger
        fallback along the NIM chain.
        """
        if decision.tier == "local":
            if succeeded and not tool_calls_malformed:
                self._consecutive_local_failures = 0
            else:
                self._consecutive_local_failures += 1
            return

        # Cloud tier
        if succeeded and not tool_calls_malformed:
            self._consecutive_cloud_failures = 0
        else:
            self._consecutive_cloud_failures += 1

    def reset(self) -> None:
        """Clear escalation state — call at the start of a new task/session."""
        self._consecutive_local_failures = 0
        self._consecutive_cloud_failures = 0
        self._fallback_index = 0

    # -- introspection -----------------------------------------------------

    @property
    def consecutive_local_failures(self) -> int:
        """Number of consecutive local-tier failures since last reset."""
        return self._consecutive_local_failures

    @property
    def consecutive_cloud_failures(self) -> int:
        """Number of consecutive cloud-tier failures since last reset."""
        return self._consecutive_cloud_failures

    def explain(self) -> dict[str, object]:
        """Return a JSON-serialisable snapshot of current router state —
        useful for ``/status`` style user-facing commands and tests."""
        return {
            "local_max_complexity": self.local_max_complexity,
            "escalate_after_failures": self.escalate_after_failures,
            "large_prompt_chars": self.large_prompt_chars,
            "large_context_tokens": self.large_context_tokens,
            "large_history_turns": self.large_history_turns,
            "fallback_chain": list(self.fallback_chain),
            "consecutive_local_failures": self._consecutive_local_failures,
            "consecutive_cloud_failures": self._consecutive_cloud_failures,
            "fallback_index": self._fallback_index,
        }


__all__ = [
    "LiteRouter",
    "RoutingDecision",
    "parse_fallback_chain",
    "DEFAULT_LOCAL_MAX_COMPLEXITY",
    "DEFAULT_ESCALATE_AFTER_FAILURES",
    "DEFAULT_LARGE_PROMPT_CHARS",
    "DEFAULT_LARGE_CONTEXT_TOKENS",
    "DEFAULT_LARGE_HISTORY_TURNS",
    "DEFAULT_FALLBACK_CHAIN",
]
