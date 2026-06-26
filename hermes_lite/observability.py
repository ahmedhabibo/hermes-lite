"""hermes_lite.observability — Lightweight per-turn JSONL logging.

Every LLM turn produces exactly one newline-delimited JSON line in
``~/.hermes_lite/turns.jsonl``.  The write is non-blocking (fire-and-forget
via :class:`asyncio.Task`) so it never delays the LLM response.

Log rotation: when the file exceeds ``MAX_LOG_BYTES`` (default 10 MB),
the current file is renamed to ``turns.jsonl.1`` (deleting any previous
``.1``) and a fresh file is started.  Roughly 10K turns per rotation.

CLI::

    python -m hermes_lite stats

Prints the last 50 turns as a Rich table with count, p50/p95 latency,
and model distribution.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LOG_DIR = Path(os.path.expanduser("~/.hermes_lite"))
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "turns.jsonl"
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TurnRecord = dict[str, Any]
"""A single log line — see module docstring for the full schema."""


def _make_record(
    *,
    turn_id: str,
    tier: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    elapsed_ms: int = 0,
    tools_called: list[str] | None = None,
    errors: list[str] | None = None,
) -> TurnRecord:
    """Build a structured turn record."""
    return {
        "ts": int(time.time()),
        "turn": turn_id,
        "tier": tier,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_ms": elapsed_ms,
        "tools_called": tools_called or [],
        "errors": errors or [],
    }


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def _rotate_if_needed(path: Path, max_bytes: int = MAX_LOG_BYTES) -> None:
    """Rotate *path* when it exceeds *max_bytes*.

    ``turns.jsonl`` → ``turns.jsonl.1`` (old .1 deleted first).
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    if size < max_bytes:
        return
    rotated = Path(str(path) + ".1")
    try:
        rotated.unlink()
    except FileNotFoundError:
        pass
    try:
        path.rename(rotated)
    except OSError:
        pass  # best-effort


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------


def _write_record(path: Path, record: TurnRecord) -> None:
    """Synchronously append a JSON line to *path*.

    Ensures the parent directory exists.  Called inside a fire-and-forget
    :class:`asyncio.Task` so the caller never blocks.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed(path)
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_turn(
    *,
    turn_id: str | None = None,
    tier: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    elapsed_ms: int = 0,
    tools_called: list[str] | None = None,
    errors: list[str] | None = None,
    log_path: Path | str | None = None,
) -> None:
    """Record one turn to the JSONL log (non-blocking).

    Spawns a :class:`asyncio.Task` that writes the line in the background.
    If no running event loop exists (e.g. called from a test or a
    synchronous context), falls back to a direct write.

    Parameters
    ----------
    turn_id:
        Unique turn identifier.  Auto-generated (UUID4) when omitted.
    tier:
        ``"local"`` or ``"cloud"``.
    model:
        Full model identifier, e.g. ``"local:qwen2.5-7b"``.
    prompt_tokens:
        Tokens sent to the model.
    completion_tokens:
        Tokens received from the model.
    elapsed_ms:
        Wall-clock time for the LLM call in milliseconds.
    tools_called:
        Names of tools invoked during this turn.
    errors:
        Error messages encountered during this turn.
    log_path:
        Override the default ``~/.hermes_lite/turns.jsonl``.
    """
    if turn_id is None:
        turn_id = uuid.uuid4().hex

    record = _make_record(
        turn_id=turn_id,
        tier=tier,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_ms=elapsed_ms,
        tools_called=tools_called,
        errors=errors,
    )

    path = Path(log_path) if log_path else DEFAULT_LOG_FILE

    # Try async fire-and-forget; fall back to sync
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(asyncio.to_thread(_write_record, path, record))
    except RuntimeError:
        # No running loop — write synchronously
        _write_record(path, record)


# ---------------------------------------------------------------------------
# Stats — ``python -m hermes_lite stats``
# ---------------------------------------------------------------------------


def read_turns(
    log_path: Path | str | None = None,
    limit: int = 50,
) -> list[TurnRecord]:
    """Read the last *limit* turn records from the JSONL log.

    Parameters
    ----------
    log_path:
        Override the default path.
    limit:
        Maximum records to return (most recent first).

    Returns
    -------
    list[TurnRecord]
        Parsed JSON objects, newest first.
    """
    path = Path(log_path) if log_path else DEFAULT_LOG_FILE
    records: list[TurnRecord] = []

    # Read from both the active file and the rotated backup
    for p in [path, Path(str(path) + ".1")]:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Sort by timestamp descending, take last N
    records.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return records[:limit]


def compute_stats(turns: Sequence[TurnRecord]) -> dict[str, Any]:
    """Compute summary statistics from a list of turn records.

    Returns
    -------
    dict
        ``count``, ``p50_ms``, ``p95_ms``, ``model_counts``,
        ``tier_counts``, ``total_prompt_tokens``,
        ``total_completion_tokens``, ``total_errors``.
    """
    if not turns:
        return {
            "count": 0,
            "p50_ms": 0,
            "p95_ms": 0,
            "model_counts": {},
            "tier_counts": {},
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_errors": 0,
        }

    latencies = [t.get("elapsed_ms", 0) for t in turns if t.get("elapsed_ms", 0) > 0]
    p50 = int(statistics.median(latencies)) if latencies else 0
    p95 = int(statistics.quantiles(latencies, n=20)[18]) if len(latencies) >= 2 else (latencies[0] if latencies else 0)

    model_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    total_prompt = 0
    total_completion = 0
    total_errors = 0

    for t in turns:
        m = t.get("model", "unknown")
        model_counts[m] = model_counts.get(m, 0) + 1
        tr = t.get("tier", "unknown")
        tier_counts[tr] = tier_counts.get(tr, 0) + 1
        total_prompt += t.get("prompt_tokens", 0)
        total_completion += t.get("completion_tokens", 0)
        total_errors += len(t.get("errors", []))

    return {
        "count": len(turns),
        "p50_ms": p50,
        "p95_ms": p95,
        "model_counts": model_counts,
        "tier_counts": tier_counts,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_errors": total_errors,
    }


def print_stats(log_path: Path | str | None = None, limit: int = 50) -> None:
    """Print a Rich table of the last *limit* turns + summary stats.

    Used by ``python -m hermes_lite stats``.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    turns = read_turns(log_path=log_path, limit=limit)

    if not turns:
        console.print("[dim]No turn records found.[/dim]")
        return

    # Recent turns table
    table = Table(
        title=f"Last {len(turns)} Turns",
        show_lines=True,
        border_style="cyan",
    )
    table.add_column("Time", style="dim", max_width=19)
    table.add_column("Tier", justify="center")
    table.add_column("Model", max_width=25)
    table.add_column("Latency", justify="right")
    table.add_column("Tokens (p→c)", justify="right")
    table.add_column("Tools", max_width=30)
    table.add_column("Errs", justify="right")

    for t in reversed(turns):  # chronological order
        ts = t.get("ts", 0)
        from datetime import datetime, timezone
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, ValueError):
            time_str = str(ts)

        tier = t.get("tier", "?")
        tier_style = "green" if tier == "local" else "magenta"
        model = t.get("model", "?")
        if len(model) > 25:
            model = "…" + model[-24:]
        ms = t.get("elapsed_ms", 0)
        latency = f"{ms}ms" if ms else "—"
        pt = t.get("prompt_tokens", 0)
        ct = t.get("completion_tokens", 0)
        tokens = f"{pt}→{ct}" if (pt or ct) else "—"
        tools = ", ".join(t.get("tools_called", [])) or "—"
        errs = str(len(t.get("errors", [])))

        table.add_row(
            time_str,
            f"[{tier_style}]{tier}[/{tier_style}]",
            model,
            latency,
            tokens,
            tools,
            errs,
        )

    console.print(table)

    # Summary stats
    stats = compute_stats(turns)
    console.print()
    console.print("[bold]Summary[/bold]")
    console.print(f"  Turns:          {stats['count']}")
    console.print(f"  p50 latency:    {stats['p50_ms']}ms")
    console.print(f"  p95 latency:    {stats['p95_ms']}ms")
    console.print(f"  Prompt tokens:  {stats['total_prompt_tokens']:,}")
    console.print(f"  Output tokens:  {stats['total_completion_tokens']:,}")
    console.print(f"  Total errors:   {stats['total_errors']}")

    if stats["model_counts"]:
        console.print("\n[bold]Model Distribution[/bold]")
        for m, c in sorted(stats["model_counts"].items(), key=lambda x: -x[1]):
            console.print(f"  {m}: {c}")

    if stats["tier_counts"]:
        console.print("\n[bold]Tier Distribution[/bold]")
        for tr, c in sorted(stats["tier_counts"].items(), key=lambda x: -x[1]):
            console.print(f"  {tr}: {c}")


__all__ = [
    "log_turn",
    "read_turns",
    "compute_stats",
    "print_stats",
    "TurnRecord",
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_FILE",
    "MAX_LOG_BYTES",
]
