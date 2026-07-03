"""tests/test_router.py — Unit tests for the LiteRouter complexity-based router.

These tests exercise the four weighted complexity signals plus the
escalation state machine. They are pure-Python and do not touch the
network — no LLM client, no SQLite.

Boundary cases covered (per acceptance):
* Simple "find X" prompts always route to local
* Long / keyword-heavy prompts route to cloud
* Forced escalation flips the tier after >= N local failures
* Malformed ``tool_calls`` count as a local failure
* A successful call resets the escalation counter
* Env vars override defaults
* Fallback chain parses cleanly from a comma-separated env string
"""

from __future__ import annotations

import pytest

from hermes_lite.router import (
    DEFAULT_ESCALATE_AFTER_FAILURES,
    DEFAULT_LOCAL_MAX_COMPLEXITY,
    LiteRouter,
    RoutingDecision,
    parse_fallback_chain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def router() -> LiteRouter:
    """Plain router with defaults; deterministic for tests."""
    return LiteRouter(
        fallback_chain=[
            "local:qwen2.5-3b",
            "minimaxai/minimax-m3",
        ],
    )


@pytest.fixture
def cloud_first_router() -> LiteRouter:
    """Router with a cloud-first preferred model and a local entry second."""
    return LiteRouter(
        fallback_chain=[
            "minimaxai/minimax-m3",
            "local:qwen2.5-3b",
        ],
    )


# ---------------------------------------------------------------------------
# parse_fallback_chain
# ---------------------------------------------------------------------------


class TestParseFallbackChain:
    def test_basic_string(self):
        assert parse_fallback_chain("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert parse_fallback_chain(" a , b ,  c  ") == ["a", "b", "c"]

    def test_drops_empty_fragments(self):
        assert parse_fallback_chain("a,,b,, ,c") == ["a", "b", "c"]

    def test_accepts_iterable(self):
        assert parse_fallback_chain(["x", "y", "z"]) == ["x", "y", "z"]

    def test_accepts_generator(self):
        assert parse_fallback_chain(f"m{i}" for i in range(3)) == ["m0", "m1", "m2"]

    def test_empty_string(self):
        assert parse_fallback_chain("") == []


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------


class TestComplexityScore:
    def test_simple_prompt_is_low_score(self, router: LiteRouter):
        """'find X' style prompts should sit well below the local threshold."""
        score = router.complexity("find customer_id for order 42", 100, 0)
        assert score < router.local_max_complexity

    def test_empty_prompt_is_zero(self, router: LiteRouter):
        """No signals → zero score."""
        score = router.complexity("", 0, 0)
        assert score == 0.0

    def test_long_prompt_pushes_score_high(self, router: LiteRouter):
        """A 5k-char prompt should max out the prompt length component."""
        prompt = "x" * 5000
        score = router.complexity(prompt, 0, 0)
        # 0.2 (prompt) + 0.0 + 0.0 + 0.0 = 0.2
        assert score == pytest.approx(0.2, abs=1e-9)

    def test_keywords_push_score_high(self, router: LiteRouter):
        """All keyword hits should consume the 0.2 keyword weight."""
        # Pad both prompts to identical length so the length contribution
        # cancels when we take the delta.
        wordy = (" refactor architect debug multi-step " + " ").ljust(60)
        neutral = ("hello world friend" + " ").ljust(60)
        delta = router.complexity(wordy, 0, 0) - router.complexity(neutral, 0, 0)
        # 4 hits × 0.25 × 0.2 = 0.2 contribution.
        assert delta == pytest.approx(0.2, abs=1e-9)

    def test_history_turns_push_score_up(self, router: LiteRouter):
        """history_turns=large should add the 0.2 history weight."""
        score = router.complexity("", 0, router.large_history_turns)
        # 0.0 (prompt) + 0.0 (context) + 0.2 (history) + 0.0 (keyword) = 0.2
        assert score == pytest.approx(0.2, abs=1e-9)

    def test_context_tokens_dominate(self, router: LiteRouter):
        """context_tokens=large should contribute 0.4 (vs 0.2 each)."""
        no_ctx = router.complexity("hello", 0, 0)
        with_ctx = router.complexity("hello", router.large_context_tokens, 0)
        delta = with_ctx - no_ctx
        assert delta == pytest.approx(0.4, abs=1e-9)

    def test_score_clamps_to_one(self, router: LiteRouter):
        """All signals maxed should still clamp at 1.0."""
        prompt = (
            "refactor architect debug multi-step redesign optimize 'explain in detail'"
            + " x" * 5000
        )
        score = router.complexity(prompt, router.large_context_tokens, router.large_history_turns)
        assert score == pytest.approx(1.0)

    def test_score_is_deterministic(self, router: LiteRouter):
        """Two identical calls must produce the same score."""
        prompt = "Please refactor the model loader"
        s1 = router.complexity(prompt, 200, 1)
        s2 = router.complexity(prompt, 200, 1)
        assert s1 == s2

    def test_non_string_prompt_handled(self, router: LiteRouter):
        """Robustness — non-string input should not crash."""
        # Cast path means the int's str() repr contributes a tiny prompt score.
        score = router.complexity(12345, 0, 0)  # type: ignore[arg-type]
        assert 0.0 <= score <= 0.01


# ---------------------------------------------------------------------------
# Routing decisions — acceptance criteria
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_simple_find_x_is_local_100_percent(self, router: LiteRouter):
        """Acceptance: 'find X' should always be local."""
        d = router.route("find invoice id 42", 50, 0)
        assert d.tier == "local"
        assert d.fell_back is False

    def test_simple_find_x_local_across_sizes(self, router: LiteRouter):
        """Across several natural phrasings the tier stays local."""
        for prompt in [
            "find me the customer",
            "Find the latest invoice",
            "show ticket 17",
            "where is the README",
        ]:
            d = router.route(prompt, 100, 0)
            assert d.tier == "local", prompt

    def test_refactor_long_script_routes_to_cloud(self, router: LiteRouter):
        """Acceptance: 'refactor this 200-line script' should always be cloud."""
        d = router.route("refactor this 200-line script", context_tokens=0, history_turns=0)
        assert d.tier == "cloud"
        assert d.fell_back is True  # complexity > threshold flipped tier

    def test_above_threshold_score_routes_to_cloud(self, router: LiteRouter):
        """Score > threshold forces cloud even on local-preferred chain."""
        # Push key word heaviness only.
        d = router.route(
            "refactor architect debug",
            context_tokens=0,
            history_turns=0,
        )
        assert d.tier == "cloud"

    def test_cloud_first_chain_is_cloud_simple(self, cloud_first_router: LiteRouter):
        """When preferred is cloud, simple prompts stay cloud (operator's policy)."""
        d = cloud_first_router.route("hi", 10, 0)
        assert d.tier == "cloud"

    def test_decision_fields_populated(self, router: LiteRouter):
        """Every RoutingDecision must carry model_id, tier, score, reason."""
        d = router.route("echo", 0, 0)
        assert isinstance(d, RoutingDecision)
        assert d.model_id  # non-empty
        assert d.tier in {"local", "cloud"}
        assert 0.0 <= d.complexity_score <= 1.0
        assert d.reason

    def test_repeated_routes_for_simple_prompt_stay_local(self, router: LiteRouter):
        """100 simple prompts in a row — none should escalate."""
        for _ in range(100):
            d = router.route("find product sku a", 0, 0)
            assert d.tier == "local"


# ---------------------------------------------------------------------------
# Escalation state machine
# ---------------------------------------------------------------------------


class TestEscalation:
    def test_escalation_triggers_after_n_local_failures(self, router: LiteRouter):
        """>= escalate_after_failures consecutive local failures → forced cloud."""
        d0 = router.route("hi", 0, 0)
        assert d0.tier == "local"

        # Record enough failures to trigger escalation.
        for _ in range(router.escalate_after_failures):
            router.record_outcome(d0, succeeded=False)

        # Next decision must be cloud, regardless of complexity.
        d1 = router.route("hi", 0, 0)
        assert d1.tier == "cloud"
        assert d1.fell_back is True

    def test_one_failure_below_threshold_no_escalation(self, router: LiteRouter):
        """Single failure shouldn't flip tier."""
        d0 = router.route("hi", 0, 0)
        router.record_outcome(d0, succeeded=False)
        d1 = router.route("hi", 0, 0)
        assert d1.tier == "local"

    def test_successful_call_resets_counter(self, router: LiteRouter):
        """Success on local empties the failure streak."""
        d0 = router.route("hi", 0, 0)
        router.record_outcome(d0, succeeded=False)
        router.record_outcome(d0, succeeded=False)
        # Counter is at 2 (== threshold) — one more would flip but success resets.
        router.record_outcome(d0, succeeded=True)
        assert router.consecutive_local_failures == 0

    def test_malformed_tool_calls_counts_as_failure(self, router: LiteRouter):
        """Malformed JSON tool_calls on local tier must increment the counter."""
        d0 = router.route("hi", 0, 0)
        router.record_outcome(d0, succeeded=True, tool_calls_malformed=True)
        # tool_calls_malformed=True → treated as failure, not success.
        for _ in range(router.escalate_after_failures - 1):
            router.record_outcome(d0, succeeded=False)
        d_next = router.route("hi", 0, 0)
        assert d_next.tier == "cloud"

    def test_cloud_failure_does_not_increment(self, router: LiteRouter):
        """Failures on cloud tier are tracked separately — don't loop escalation."""
        # Build a decision whose tier is "cloud". We do this by forcing an
        # escalated decision directly and feeding it back.
        router._consecutive_local_failures = router.escalate_after_failures
        d = router.route("hi", 0, 0)
        assert d.tier == "cloud"
        # Now feeding a cloud failure should NOT keep incrementing locally
        # (state machine ignores it).
        router.record_outcome(d, succeeded=False)
        # The next route is still under escalation (counter not touched by
        # cloud failure) — but determining that requires same counter logic.
        # So force a reset and check the cloud-failure is truly a no-op:
        router.reset()
        router.record_outcome(d, succeeded=False)
        # No increments; counter stays at zero.
        d_after = router.route("hi", 0, 0)
        assert d_after.tier == "local"

    def test_reset_clears_state(self, router: LiteRouter):
        """reset() must wipe the failure counter."""
        d = router.route("hi", 0, 0)
        router.record_outcome(d, succeeded=False)
        router.record_outcome(d, succeeded=False)
        assert router.consecutive_local_failures == 2
        router.reset()
        assert router.consecutive_local_failures == 0

    def test_escalated_decision_picks_cloud_from_chain(self, router: LiteRouter):
        """The escalated model_id must come from the *cloud* entries, not local."""
        router._consecutive_local_failures = router.escalate_after_failures
        d = router.route("hi", 0, 0)
        assert d.tier == "cloud"
        # fallback_chain[1] is the cloud entry in the test fixture.
        assert d.model_id == router.fallback_chain[1]


# ---------------------------------------------------------------------------
# Configuration / env-var overrides
# ---------------------------------------------------------------------------


class TestEnvConfig:
    def test_defaults_from_env_when_no_kwargs(self, monkeypatch):
        monkeypatch.setenv("LITE_LOCAL_MAX_COMPLEXITY", "0.7")
        monkeypatch.setenv("LITE_ESCALATE_AFTER_FAILURES", "5")
        monkeypatch.setenv(
            "LITE_FALLBACK_CHAIN",
            "local:tiny,minimaxai/minimax-m3",
        )
        r = LiteRouter()
        assert r.local_max_complexity == 0.7
        assert r.escalate_after_failures == 5
        assert r.fallback_chain == ["local:tiny", "minimaxai/minimax-m3"]

    def test_kwargs_override_env(self, monkeypatch):
        monkeypatch.setenv("LITE_LOCAL_MAX_COMPLEXITY", "0.9")
        r = LiteRouter(local_max_complexity=0.1)
        assert r.local_max_complexity == 0.1

    def test_explain_returns_serialisable_state(self, router: LiteRouter):
        snapshot = router.explain()
        assert snapshot["local_max_complexity"] == DEFAULT_LOCAL_MAX_COMPLEXITY
        assert snapshot["escalate_after_failures"] == DEFAULT_ESCALATE_AFTER_FAILURES
        assert snapshot["consecutive_local_failures"] == 0
        assert isinstance(snapshot["fallback_chain"], list)
        # All values must be JSON-friendly primitives.
        for v in snapshot.values():
            assert isinstance(v, (int, float, str, list))

    def test_explain_reflects_failures(self, router: LiteRouter):
        d = router.route("hi", 0, 0)
        router.record_outcome(d, succeeded=False)
        router.record_outcome(d, succeeded=False)
        snap = router.explain()
        assert snap["consecutive_local_failures"] == 2


# ---------------------------------------------------------------------------
# Tier boundary edge cases
# ---------------------------------------------------------------------------


class TestTierBoundaries:
    def test_score_exactly_at_threshold_goes_local(self, router: LiteRouter):
        """Threshold comparison is strict '>' for the cloud flip → == stays local."""
        # Construct a prompt that yields exactly the threshold.
        target = router.local_max_complexity
        # 0.2 prompt + 0.4 context + 0.2 history + 0.2 keyword = 1.0
        # We need the weighted sum == target. Solve for context_norm:
        # 0.2 + 0.4*c + 0.0 + 0.0 == target  →  c = (target - 0.2) / 0.4
        needed = (target - 0.2) / 0.4
        tokens = int(needed * router.large_context_tokens)
        d = router.route("x" * router.large_prompt_chars, tokens, 0)
        assert d.tier == "local", f"score {d.complexity_score} should sit at or below threshold {target}"

    def test_score_just_above_threshold_goes_cloud(self, router: LiteRouter):
        """Score just above the threshold should flip to cloud."""
        target = router.local_max_complexity
        # With local-first v0.7, threshold is 0.7. Need prompt (0.2) + context (0.4) +
        # history (0.2) to exceed the threshold. Add extra tokens to push above exactly.
        needed_context = (target - 0.2 - 0.2) / 0.4
        tokens = int(needed_context * router.large_context_tokens) + 10  # +10 to exceed
        history_turns = router.large_history_turns  # full history weight
        d = router.route("x" * router.large_prompt_chars, tokens, history_turns)
        assert d.tier == "cloud", f"score {d.complexity_score:.4f} should exceed threshold {target}"
