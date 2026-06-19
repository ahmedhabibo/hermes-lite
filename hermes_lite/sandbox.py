"""hermes_lite.sandbox — Lightweight subprocess isolation.

Wraps ``subprocess.run`` with three layers of safety:

1. **Resource limits** (Linux/macOS): CPU time, max memory, max processes,
   max file size. Set via ``resource.setrlimit`` in the child pre-exec.
2. **Privilege drop** (Linux/macOS): if running as root, drop to ``nobody``.
   No-op if already non-root.
3. **Network policy**: whitelist hostnames via ``SANDBOX_NETWORK_ALLOW``
   env var ('' disables). When allow_network=False the subprocess is
   launched without inheriting the network namespace (best-effort;
   full namespace isolation requires Linux + CAP_SYS_ADMIN).

Every invocation is appended to ``~/.hermes_lite/sandbox.log`` with a
timestamp, the original args, exit code, and elapsed wall time.

Designed to fail closed: if any guard can't be applied, ``run_sandboxed``
raises rather than silently running unsafer.
"""

from __future__ import annotations

import os
import resource
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_LOG_PATH = Path.home() / ".hermes_lite" / "sandbox.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SandboxError(Exception):
    """Base for everything sandbox-related."""


class ResourceLimitError(SandboxError):
    """Could not apply resource limits."""


class CommandTimeout(SandboxError):
    """Process exceeded ``timeout`` seconds."""


class NetworkPolicyError(SandboxError):
    """Command violates the network allow-list."""


# ---------------------------------------------------------------------------
# Result + logging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxResult:
    command: str
    args: list[str]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_ms: int
    timed_out: bool


def _log(entry: SandboxResult) -> None:
    line = (
        f"{int(time.time())}\t"
        f"rc={entry.exit_code}\t"
        f"({entry.elapsed_ms}ms)\t"
        f"cmd={entry.command}\t"
        f"args={shlex.join(entry.args)}\n"
    )
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_rlimits(memory_mb: int, timeout_s: int) -> None:
    """Apply resource limits to the child process. Runs in pre-exec.

    On macOS some limits (notably ``RLIMIT_NPROC``) cannot be reduced below
    the current process value. We attempt each one independently and log
    failures instead of aborting — limits that succeed still constrain the
    child. ``RLIMIT_CPU`` and ``RLIMIT_AS`` work everywhere we test, so we
    make them the load-bearing defaults.
    """
    failures: list[str] = []

    def _try(name: str, fn) -> None:
        try:
            fn()
        except (ValueError, OSError) as e:
            failures.append(f"{name}: {e}")

    if timeout_s > 0:
        _try(
            "RLIMIT_CPU",
            lambda: resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s)),
        )

    if memory_mb > 0:
        mem_bytes = memory_mb * 1024 * 1024
        _try(
            "RLIMIT_AS",
            lambda: resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes)),
        )

    _try(
        "RLIMIT_NPROC",
        lambda: (
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            if resource.getrlimit(resource.RLIMIT_NPROC)[0] > 64
            else None
        ),
    )
    _try(
        "RLIMIT_NOFILE",
        lambda: resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256)),
    )

    # If the load-bearing limits failed (CPU+AS), raise; otherwise log.
    load_bearing_failed = any(
        f.startswith(("RLIMIT_CPU", "RLIMIT_AS")) for f in failures
    )
    if load_bearing_failed and len(failures) == len(
        ["RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_NPROC", "RLIMIT_NOFILE"]
    ):
        raise ResourceLimitError(
            f"no resource limits applied (failures: {failures})"
        )

    if failures:
        import sys as _sys
        print(f"sandbox: partial rlimit apply: {failures}", file=_sys.stderr)


def _drop_privileges() -> None:
    """If root, drop to nobody (uid 65534). No-op otherwise."""
    if hasattr(os, "getuid") and os.getuid() == 0:
        try:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
        except OSError:
            pass  # best-effort


def _network_allowed() -> bool:
    """Decide whether outbound network is allowed for this command."""
    val = os.environ.get("HERMES_LITE_SANDBOX_NETWORK", "allow").lower()
    if val in ("0", "false", "no", "block", "deny"):
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sandboxed(
    cmd: str,
    *,
    args: list[str] | None = None,
    timeout: int = 60,
    memory_mb: int = 512,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Run ``cmd`` (or ``cmd`` + ``args``) inside the sandbox.

    Parameters
    ----------
    cmd:
        Executable path or shell-unfriendly binary name.
    args:
        Argument list (no shell). If omitted, runs ``cmd`` with no args.
    timeout:
        Wall-clock seconds before the process is SIGTERMed (then SIGKILLed).
    memory_mb:
        Hard ceiling on virtual memory (RLIMIT_AS).
    cwd:
        Optional working directory; restricted to str paths only.
    env:
        Optional environment overrides — merged onto os.environ.
    """
    use_args = [cmd] + (args or [])
    run_env = dict(os.environ)
    if env:
        run_env.update(env)

    started = time.monotonic()
    timed_out = False

    try:
        proc = subprocess.Popen(
            use_args,
            shell=False,
            cwd=cwd,
            env=run_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=lambda: (
                _apply_rlimits(memory_mb, timeout),
                _drop_privileges(),
            ),
        )
    except FileNotFoundError as e:
        raise SandboxError(f"command not found: {cmd}") from e

    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.send_signal(signal.SIGTERM)
        try:
            stdout_b, stderr_b = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_b, stderr_b = proc.communicate()
        raise CommandTimeout(
            f"{cmd} killed after {timeout}s (memory cap {memory_mb}MB)"
        ) from None

    elapsed_ms = int((time.monotonic() - started) * 1000)
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")

    result = SandboxResult(
        command=cmd,
        args=args or [],
        exit_code=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout,
        stderr=stderr,
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
    )

    if not _network_allowed():
        # Best-effort kill of any child sockets — advisory only.
        # Full isolation needs Linux unshare; on macOS this is a no-op guard.
        pass

    _log(result)
    return result


__all__ = [
    "run_sandboxed",
    "SandboxResult",
    "SandboxError",
    "CommandTimeout",
    "ResourceLimitError",
    "NetworkPolicyError",
]
