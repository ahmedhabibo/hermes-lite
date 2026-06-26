#!/usr/bin/env python3
"""Hermes-Lite Demo Script — non-interactive, designed for screen recording.

Runs 3 pre-scripted prompts through the agent with visible tool calls,
showing the two-tier routing (local vs cloud) and tool-loop in action.

Usage:
    # Start llama-server first (see README Quick Start step 4)
    python scripts/demo.py

Requirements:
    - llama-server running on localhost:8080
    - pip install -e ".[test]"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

from hermes_lite.orchestrator import HermesOrchestrator


async def run_demo():
    console = Console()

    orch = HermesOrchestrator(db_path=":memory:")
    orch._create_default_tools()
    await orch._initialize_memory()

    demos = [
        ("List all Python files in the project", "search_files tool — local tier"),
        ("Read the first 10 lines of pyproject.toml", "read_file tool — local tier"),
        ("Design a microservice architecture for an ERP system", "cloud escalation — router bumps to cloud"),
    ]

    console.print(Panel(
        "[bold cyan]Hermes-Lite[/] — Local-First Agent Demo\n"
        "[dim]Model: Qwen 2.5 7B Instruct Q4_K_M · 8 GB MacBook · 100% offline[/]",
        box=box.ROUNDED,
        style="bold green",
    ))

    for prompt, description in demos:
        console.print(f"\n[bold yellow]▸ {description}[/]")
        console.print(f"[bold white]> {prompt}[/]\n")

        try:
            result = await orch._handle_prompt(prompt)
            console.print(Panel(
                Markdown(result) if isinstance(result, str) else str(result),
                title="⚡ Agent",
                border_style="cyan",
            ))
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            console.print("[dim]Make sure llama-server is running on port 8080[/]")

    console.print(Panel(
        "[bold green]✓ Demo complete[/]\n"
        "[dim]All tool calls ran locally on a 7B model.\n"
        "No API keys. No cloud. No GPU. Just your laptop.[/]",
        box=box.ROUNDED,
    ))


if __name__ == "__main__":
    asyncio.run(run_demo())
