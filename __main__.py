"""Hermes-Lite — CLI entry point.

Run with::

    python -m hermes_lite
"""

from hermes_lite.cli import run_cli


def _welcome_handler(prompt: str) -> str:
    """Placeholder handler — wired to echo until the orchestrator
    integration (kanban task t_4830e31a) connects the real LLM loop.
    """
    return f"[echo mode — orchestrator not yet connected]\nYou said: {prompt}"


if __name__ == "__main__":
    run_cli(
        on_prompt=_welcome_handler,
        welcome_message="Hermes-Lite v0.1 — CLI shell ready.  "
        "Prompt handler is in echo mode until the orchestrator "
        "engine is integrated (task t_4830e31a).",
    )