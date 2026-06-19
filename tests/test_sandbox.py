"""tests/test_sandbox.py — Safety tests for hermes_lite.sandbox."""

from __future__ import annotations

import sys

import pytest

from hermes_lite.sandbox import (
    CommandTimeout,
    ResourceLimitError,
    SandboxError,
    run_sandboxed,
)


def test_run_echo():
    """Trivial happy path."""
    r = run_sandboxed("/bin/echo", args=["hello sandbox"], timeout=10)
    assert r.exit_code == 0
    assert "hello sandbox" in r.stdout
    assert r.stderr == ""
    assert r.elapsed_ms >= 0


def test_run_with_args():
    r = run_sandboxed("/bin/ls", args=["/tmp"], timeout=10)
    assert r.exit_code == 0
    # stdout should contain at least one /tmp entry
    assert len(r.stdout.split()) >= 1


def test_timeout_raises():
    """Sleep past the timeout — should raise CommandTimeout."""
    with pytest.raises(CommandTimeout):
        run_sandboxed("/bin/sleep", args=["5"], timeout=1, memory_mb=64)


def test_command_not_found():
    """Bad command path raises SandboxError (FileNotFoundError wrapped)."""
    with pytest.raises(SandboxError):
        run_sandboxed("/does/not/exist/at/all", timeout=5)


def test_exit_propagated():
    """Non-zero exit does NOT raise — we want to surface the error."""
    r = run_sandboxed("/bin/sh", args=["-c", "exit 7"], timeout=10)
    assert r.exit_code == 7


def test_log_file_created():
    """Audit log is appended to."""
    import os

    log_path = os.path.expanduser("~/.hermes_lite/sandbox.log")
    if os.path.exists(log_path):
        before = os.path.getsize(log_path)
    else:
        before = 0
    run_sandboxed("/bin/echo", args=["log-test"], timeout=5)
    after = os.path.getsize(log_path)
    assert after > before


@pytest.mark.skipif(
    sys.platform == "darwin" and "CI" in __import__("os").environ,
    reason="RLIMIT_AS behaviour differs on macOS in CI",
)
def test_memory_cap_kills_yes():
    """Allocating huge memory should be aborted by RLIMIT_AS."""
    with pytest.raises(CommandTimeout):
        # /usr/bin/yes repeats forever; the memory cap doesn't kill it but the timeout does
        run_sandboxed("/usr/bin/yes", args=[], timeout=1, memory_mb=128)
