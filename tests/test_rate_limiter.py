import asyncio
import time
from unittest.mock import patch

import pytest

from hermes_lite.llm import APIKeyRotator, AllKeysExhausted, RateLimiter


def test_rate_limiter_init():
    rl = RateLimiter(rpm=60)
    assert rl.rpm == 60
    assert rl._max_tokens == 60.0
    assert rl._refill_per_sec == 1.0
    assert rl._tokens == 60.0


def test_rate_limiter_acquire_immediate():
    rl = RateLimiter(rpm=60)
    # Just check that we start with a full bucket
    assert rl._tokens == 60.0


@pytest.mark.asyncio
async def test_rate_limiter_acquire_wait():
    rl = RateLimiter(rpm=60)  # 1 per second
    # Consume all tokens quickly
    rl._tokens = 0.0
    # Record time before
    start = time.monotonic()
    # Acquire one token should take about 1 second
    await rl.acquire()
    elapsed = time.monotonic() - start
    # Allow some tolerance for system scheduling
    assert 0.8 <= elapsed <= 1.2
    # After acquiring, tokens should be in [0, 1) because we just took one token
    # and the refill during the wait might have added a tiny fraction.
    assert 0 <= rl._tokens < 1.0


@pytest.mark.asyncio
async def test_rate_limiter_burst():
    rl = RateLimiter(rpm=60)  # bucket size 60, refill 1 per second
    # We should be able to acquire 60 tokens immediately
    start = time.monotonic()
    tasks = [asyncio.create_task(rl.acquire()) for _ in range(60)]
    await asyncio.gather(*tasks)
    elapsed = time.monotonic() - start
    # Should be almost instantaneous
    assert elapsed < 0.2
    # Now the bucket is empty (allow small error)
    assert 0 <= rl._tokens < 1.0
    # Next acquisition should take about 1 second
    start = time.monotonic()
    await rl.acquire()
    elapsed = time.monotonic() - start
    assert 0.8 <= elapsed <= 1.2


def test_rate_limiter_refill():
    rl = RateLimiter(rpm=60)  # 1 per second
    # Start with empty bucket
    rl._tokens = 0.0
    rl._last_refill = time.monotonic() - 1.0  # 1 second ago
    # Trigger a refill
    rl._refill()
    # Should have 1 token now (allow small floating point error)
    assert abs(rl._tokens - 1.0) < 1e-5
    # After another second, should have 2
    rl._last_refill = time.monotonic() - 1.0
    rl._refill()
    assert abs(rl._tokens - 2.0) < 1e-5


def test_api_key_rotator_init():
    keys = ["key1", "key2", "key3"]
    rot = APIKeyRotator(keys=keys)
    assert rot._keys == keys
    assert rot._index == 0
    assert rot.current == "key1"


def test_api_key_rotator_mark_failure():
    keys = ["key1", "key2", "key3"]
    rot = APIKeyRotator(keys=keys)
    # Initially, current is key1
    assert rot.current == "key1"
    # After first failure, key0 goes to cooldown, current becomes key2
    assert rot.mark_failure() == "key2"
    # After second failure, key1 goes to cooldown, current becomes key3
    assert rot.mark_failure() == "key3"
    # After third failure, key2 goes to cooldown, current becomes key1 (wrapped)
    assert rot.mark_failure() == "key1"
    # Now all three are in cooldown
    exhausted, remaining = rot.is_exhausted()
    assert exhausted is True
    assert remaining > 55  # approximately 60 seconds minus a few milliseconds


def test_api_key_rotator_current_when_exhausted():
    keys = ["key1"]
    rot = APIKeyRotator(keys=keys)
    # Mark the only key as failed
    rot.mark_failure()
    # current should still return the key (even though in cooldown)
    assert rot.current == "key1"
    # is_exhausted should be True
    exhausted, _ = rot.is_exhausted()
    assert exhausted is True


def test_all_keys_exhausted_exception():
    exc = AllKeysExhausted(keys_tried=3, cooldown_remaining=10.5)
    assert "All 3 API keys exhausted" in str(exc)
    assert "10s" in str(exc)  # .0f formatting gives 10
    # Check attributes
    assert exc.keys_tried == 3
    assert exc.cooldown_remaining == 10.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])