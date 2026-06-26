"""hermes_lite.cli — CLI Shell & Rich Interface Loop

Terminal interface loop built with prompt_toolkit and Rich.
Captures raw prompts, renders styled panels, handles Ctrl+C/D,
and passes input to a callback for LLM generation.

Usage::

    from hermes_lite.cli import run_cli

    def my_handler(prompt: str) -> str:
        return f"You said: {prompt}"

    run_cli(
        on_prompt=my_handler,
        welcome_message="Hermes-Lite v0.1 — type your prompt below.",
    )
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PtStyle
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
from rich.text import Text

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

# Rich console — our primary output renderer
_console = Console(
    color_system="auto",
    highlight=True,
)

# prompt_toolkit style overrides
_PT_STYLE = PtStyle.from_dict({
    "prompt": "bold cyan",
    "status": "bold green",
    "error": "bold red",
    "trailing_input": "dim",
})


def _style() -> PtStyle:
    return _PT_STYLE


# ---------------------------------------------------------------------------
# Key bindings
# ---------------------------------------------------------------------------


def _make_bindings(on_exit: Callable[[], Any] | None = None) -> KeyBindings:
    """Build key bindings for the prompt session.

    - Ctrl+C / Ctrl+D: graceful exit
    """
    bindings = KeyBindings()

    @bindings.add("c-c", eager=True)
    def _ctrl_c(event: Any) -> None:
        """Graceful exit on Ctrl+C."""
        event.app.exit(result="<EXIT>")
        if on_exit:
            on_exit()

    @bindings.add("c-d", eager=True)
    def _ctrl_d(event: Any) -> None:
        """Graceful exit on Ctrl+D (EOF)."""
        event.app.exit(result="<EXIT>")
        if on_exit:
            on_exit()

    return bindings


# ---------------------------------------------------------------------------
# Prompt handler type
# ---------------------------------------------------------------------------

PromptHandler = Callable[[str], str] | Callable[[str], Awaitable[str]]
"""Signature for the LLM generation callback.

Receives the user's raw prompt string.  May be sync or async.
Should return the LLM response text.
"""

# ---------------------------------------------------------------------------
# Status rendering helpers
# ---------------------------------------------------------------------------


def _render_welcome(header: str) -> None:
    """Print a welcome panel."""
    _console.print()
    _console.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]{header}[/bold cyan]\n\n"
                "[dim]Type your prompt below.  "
                "Ctrl+C or Ctrl+D to exit.[/dim]"
            ),
            box=box.HEAVY,
            border_style="cyan",
            title="[bold]🐚 Hermes-Lite[/bold]",
            title_align="left",
        )
    )
    _console.print()


def _render_response(prompt: str, response: str) -> None:
    """Render a single exchange in a panel."""
    _console.print()

    # User prompt panel
    _console.print(
        Panel(
            Text.from_markup(f"[bold white]❯ {prompt}[/bold white]"),
            box=box.ROUNDED,
            border_style="blue",
            title="[blue]You[/blue]",
            title_align="left",
            padding=(0, 1),
        )
    )

    # Response panel — try Markdown, fall back to plain text
    try:
        content: Markdown | Text = Markdown(response)
    except Exception:
        content = Text(response)

    _console.print(
        Panel(
            content,
            box=box.ROUNDED,
            border_style="green",
            title="[green]Hermes[/green]",
            title_align="left",
            padding=(0, 1),
        )
    )

    _console.print()


def _render_error(error: str) -> None:
    """Render an error message."""
    _console.print()
    _console.print(
        Panel(
            Text.from_markup(f"[bold red]⚠ {error}[/bold red]"),
            box=box.HEAVY,
            border_style="red",
            title="[red]Error[/red]",
            title_align="left",
        )
    )
    _console.print()


def _render_status(text: str) -> None:
    """Print a status line (no panel)."""
    _console.print(f"[dim]⚡ {text}[/dim]")


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


async def _run_async(
    on_prompt: PromptHandler,
    *,
    welcome_message: str = "Welcome to Hermes-Lite.",
    system_prompt: str | None = None,
    prompt_text: str = "❯ ",
) -> None:
    """Core async CLI loop.

    Parameters
    ----------
    on_prompt:
        Callback invoked for each user prompt.  May be sync or async.
    welcome_message:
        Text shown in the welcome panel.
    system_prompt:
        Optional system-level message shown on startup (not sent to the
        generation loop — just displayed).
    prompt_text:
        Text shown as the input prompt.
    """
    _render_welcome(welcome_message)

    if system_prompt:
        _render_response("System", system_prompt)

    # Build prompt session
    history = InMemoryHistory()
    bindings = _make_bindings()

    session: PromptSession[str] = PromptSession(
        history=history,
        key_bindings=bindings,
        style=_style(),
        enable_history_search=True,
        complete_while_typing=False,
    )

    # Main input loop
    while True:
        try:
            raw = await session.prompt_async(
                message=[
                    ("class:prompt", prompt_text),
                ],
            )
        except (EOFError, KeyboardInterrupt):
            # These can also bubble from the prompt; handle gracefully
            _render_status("Goodbye!")
            break

        prompt = raw.strip()

        # Skip empty
        if not prompt:
            continue

        # Special: exit commands
        if prompt.lower() in ("/exit", "/quit", "/q"):
            _render_status("Goodbye!")
            break

        # Render the user's prompt
        _render_status("Thinking ...")

        try:
            # Call the handler (sync or async)
            if asyncio.iscoroutinefunction(on_prompt):
                response = await on_prompt(prompt)
            else:
                response = on_prompt(prompt)
        except Exception as exc:
            _render_error(f"{type(exc).__name__}: {exc}")
            continue

        # Render the response
        _render_response(prompt, str(response))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_cli(
    on_prompt: PromptHandler,
    *,
    welcome_message: str = "Welcome to Hermes-Lite.",
    system_prompt: str | None = None,
    prompt_text: str = "❯ ",
) -> None:
    """Launch the Hermes-Lite CLI shell.

    Parameters
    ----------
    on_prompt:
        Callback invoked for each user prompt.  Receives the raw prompt
        string; must return (or resolve to) the LLM response text.
        May be ``async def`` or a regular ``def``.
    welcome_message:
        Text shown in the welcome panel on startup.
    system_prompt:
        Optional system-level message displayed on startup.
    prompt_text:
        Text shown as the input prompt cue.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop and loop.is_running():
            # Already inside an event loop (e.g. PTY wrapper) — run
            # the async CLI in a separate thread with its own loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(
                    asyncio.run,
                    _run_async(
                        on_prompt=on_prompt,
                        welcome_message=welcome_message,
                        system_prompt=system_prompt,
                        prompt_text=prompt_text,
                    ),
                ).result()
        else:
            asyncio.run(
                _run_async(
                    on_prompt=on_prompt,
                    welcome_message=welcome_message,
                    system_prompt=system_prompt,
                    prompt_text=prompt_text,
                )
            )
    except KeyboardInterrupt:
        pass
    _render_status("Goodbye!")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------


def _echo_handler(prompt: str) -> str:
    """Simple echo handler for testing the CLI."""
    return f"Echo: {prompt}"


if __name__ == "__main__":
    run_cli(
        on_prompt=_echo_handler,
        welcome_message="Hermes-Lite v0.2 — type anything to chat. Use /help for commands.",
    )