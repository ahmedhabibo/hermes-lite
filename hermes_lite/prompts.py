"""hermes_lite.prompts — System prompt + persona loader.

Loads `prompts/system.md` and the optional persona overlay into a single
system message string. All I/O is read-only and synchronous; the loaded
content is cached on the module for the lifetime of the process.

Config (env vars):
- HERMES_LITE_PROMPT_OVERRIDE: absolute path to a markdown file to use
  instead of the bundled `system.md`. Useful for hot-reloading prompts
  during dev.
- HERMES_LITE_PERSONA: "concise" | "balanced" | "verbose" — appends the
  persona section to the system prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_SYSTEM = _PROMPTS_DIR / "system.md"
_DEFAULT_PERSONAS = _PROMPTS_DIR / "personas.md"

_ALLOWED_PERSONAS = {"concise", "balanced", "verbose"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _config_persona() -> str:
    """Return persona name from canonical config (single source of truth).

    Falls back to env if config has not been initialized yet, so callers
    like the CLI that pre-import ``prompts`` still work during cold start.
    """
    try:
        from hermes_lite.config import get_config
        return get_config().persona or "balanced"
    except Exception:
        import os
        return os.environ.get("HERMES_LITE_PERSONA") or "balanced"


def _config_override() -> str | None:
    """Return prompt override path from canonical config (env fallback)."""
    try:
        from hermes_lite.config import get_config
        return get_config().prompt_override or None
    except Exception:
        import os
        return os.environ.get("HERMES_LITE_PROMPT_OVERRIDE") or None


@lru_cache(maxsize=8)
def _load_system(override: str | None) -> str:
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"HERMES_LITE_PROMPT_OVERRIDE not found: {p}")
        return _read(p)
    return _read(_DEFAULT_SYSTEM)


@lru_cache(maxsize=4)
def _load_personas() -> dict[str, str]:
    """Parse personas.md into a {name: section} map."""
    full = _read(_DEFAULT_PERSONAS)
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in full.splitlines():
        if line.startswith("## "):
            if current:
                sections[current.lower()] = "\n".join(buf).strip()
            current = line[3:].strip().split(" ")[0].lower()
            buf = []
        else:
            buf.append(line)
    if current:
        sections[current.lower()] = "\n".join(buf).strip()
    return sections


def build_system_prompt(
    persona: str | None = None,
    *,
    extra: str | None = None,
) -> str:
    """Return the full system prompt string.

    Order: identity + tools + loop + style + persona overlay + extra.
    """
    override = _config_override()
    base = _load_system(override)
    parts = [base]

    p = (persona or _config_persona()).lower()
    if p not in _ALLOWED_PERSONAS:
        raise ValueError(f"Unknown persona: {p}. Allowed: {_ALLOWED_PERSONAS}")

    sections = _load_personas()
    overlay = sections.get(p)
    if overlay and p != "balanced":
        parts.append(f"\n## Persona: {p}\n{overlay}")

    if extra:
        parts.append(f"\n## Extra\n{extra.strip()}")

    return "\n".join(parts)


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English markdown)."""
    return len(text) // 4


__all__ = ["build_system_prompt", "approx_tokens"]
