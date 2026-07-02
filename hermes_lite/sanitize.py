"""Input sanitization for Hermes-Lite.

Protects against:
- Prompt injection (control tokens like <|system|>, <|user|>, <|end|>)
- Path traversal (../../etc/passwd in file operations)
- Shell injection (metacharacters in terminal commands)
- MoA reference output poisoning (injected system prompts)
"""

from __future__ import annotations

import re
import os
import ast
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Control token patterns (prompt injection)
# ---------------------------------------------------------------------------

# Known LLM control / special tokens used for prompt injection.
# These are stripped from user prompts before they reach the LLM.
_CONTROL_TOKEN_PATTERNS = [
    r"<\|system\|>.*?</\|system\|>",
    r"<\|user\|>.*?</\|user\|>",
    r"<\|assistant\|>.*?</\|assistant\|>",
    r"<\|endoftext\|>",
    r"<\|end\|>",
    r"<\|im_start\|>.*?</\|im_end\|>",
    r"<\|startoftext\|>",
    r"<\|end_of_turn\|>",
    r"<\|reserved_\d+\|>",
    r"\[INST\].*?\[/INST\]",
    r"<<SYS>>.*?<</SYS>>",
    r"<\|begin_of_text\|>",
    r"<\|end_of_text\|>",
    # Bare tokens (without closing tags) — catch standalone injections
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<s>",
    r"</s>",
]

# Compile the combined regex once.
_CONTROL_RE = re.compile("|".join(_CONTROL_TOKEN_PATTERNS), re.IGNORECASE | re.DOTALL)

# ---------------------------------------------------------------------------
# Path-traversal detection
# ---------------------------------------------------------------------------

# Disallow these path fragments entirely.
_PATH_BLACKLIST = {
    "..": "Path traversal attempt",
    "~": "Home directory traversal",
    "/etc/": "System directory access",
    "/proc/": "Process directory access",
    "/sys/": "System directory access",
    "/dev/": "Device directory access",
    "/var/log/": "System log directory access",
    "/var/www/": "Web server directory access",
    "/usr/bin/": "System binary directory access",
    "/usr/sbin/": "System binary directory access",
    "/home/": "Home directory access",
    "C:": "Windows absolute path",
    "D:": "Windows absolute path",
}

# Characters that look suspicious in a path.
_PATH_SUSPICIOUS = re.compile(r"[;|`$&\\{\\}\\(\\)]")

# Maximum path length (arbitrary sanity limit).
_MAX_PATH_LEN = 4096


# ---------------------------------------------------------------------------
# Shell-injection detection
# ---------------------------------------------------------------------------

_SHELL_METACHAR_RE = re.compile(r"[;|`$&\(\)\{\}\[\]<>#*?]")

# Whitelist of safe shell commands (base names only).
_SAFE_COMMANDS = {
    "ls", "cat", "echo", "grep", "wc", "head", "tail", "find", "sort", "uniq",
    "pwd", "cd", "mkdir", "touch", "rm", "cp", "mv", "chmod", "chown", "ps",
    "df", "du", "top", "htop", "git", "python", "python3", "pip", "pip3"
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SanitizationResult:
    """Result of sanitizing tool arguments."""

    is_clean: bool
    """True if no violations were found."""

    args: dict = field(default_factory=dict)
    """Sanitized (or original) arguments."""

    issues: List[str] = field(default_factory=list)
    """List of human-readable issues found."""

    def __bool__(self) -> bool:
        return self.is_clean


# ---------------------------------------------------------------------------
# Control token scrubbing
# ---------------------------------------------------------------------------

def scrub_control_tokens(text: str, replacement: str = "[REDACTED]") -> str:
    """Strip known LLM control tokens from *text*.

    Returns the cleaned text with control tokens replaced by *replacement*.
    Logs each stripped token for audit purposes.
    """
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        matched = match.group(0)
        logger.warning("Control token scrubbed: %s", repr(matched[:80]))
        return replacement

    return _CONTROL_RE.sub(_replace, text)


def strip_control_tokens(text: str) -> str:
    """Strip control tokens without logging (for internal use)."""
    if not text:
        return text
    return _CONTROL_RE.sub("", text)


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def validate_path(path_str: str, strict: bool = True) -> str:
    """Validate a file path for traversal and suspicious characters.

    * ``strict=True`` (default) -> raises on any violation.
    * ``strict=False`` -> returns cleaned path, logs warnings.
    """
    if not path_str:
        if strict:
            raise ValueError("Empty path not allowed")
        return path_str

    # Length check
    if len(path_str) > _MAX_PATH_LEN:
        if strict:
            raise ValueError(f"Path too long ({len(path_str)} > {_MAX_PATH_LEN})")
        path_str = path_str[:_MAX_PATH_LEN]

    # Check blacklist fragments
    for fragment, reason in _PATH_BLACKLIST.items():
        if fragment in path_str:
            if strict:
                raise ValueError(f"Path '{path_str}' blocked: {reason}")
            logger.warning("Path blacklist match: %s in %s", fragment, path_str)

    # Check suspicious characters
    if _PATH_SUSPICIOUS.search(path_str):
        if strict:
            raise ValueError(f"Path contains suspicious characters: {path_str}")
        logger.warning("Path suspicious chars: %s", path_str)

    return path_str


# ---------------------------------------------------------------------------
# Shell command validation
# ---------------------------------------------------------------------------

def validate_shell(command: str, strict: bool = True) -> str:
    """Validate a shell command for injection attacks.

    * ``strict=True`` (default) -> raises on suspicious input.
    * ``strict=False`` -> logs warning, returns cleaned command.
    """
    if not command:
        if strict:
            raise ValueError("Empty command not allowed")
        return command

    # Quick check for shell metacharacters
    if _SHELL_METACHAR_RE.search(command):
        if strict:
            raise ValueError(f"Command contains shell metacharacters: {command}")
        logger.warning("Shell metacharacter in: %s", command)

    return command


# ---------------------------------------------------------------------------
# Tool argument sanitization
# ---------------------------------------------------------------------------

def sanitize_tool_args(tool_name: str, args: dict, strict: bool = True) -> SanitizationResult:
    """Sanitize tool arguments based on tool type.

    Currently handles:
    - ``read_file`` / ``search_files``: validates ``path`` argument
    - ``terminal``: validates ``command`` argument
    - ``subagent``: scrubs ``task`` argument for control tokens
    - Default: scrubs control tokens from string values

    Returns a :class:`SanitizationResult` with ``is_clean``, ``args``, and ``issues``.
    """
    issues: List[str] = []
    sanitized = dict(args)  # Start with a shallow copy

    for key, value in list(sanitized.items()):
        # Path tools
        if tool_name in ("read_file", "search_files") and key == "path":
            try:
                validate_path(value, strict=strict)
            except ValueError as exc:
                issues.append(str(exc))

        # Terminal
        elif tool_name == "terminal" and key in ("command", "cmd"):
            try:
                validate_shell(value, strict=strict)
            except ValueError as exc:
                issues.append(str(exc))

        # Subagent task
        elif tool_name == "subagent" and key == "task":
            cleaned = scrub_control_tokens(value)
            if cleaned != value:
                sanitized[key] = cleaned
                issues.append("Control tokens stripped from subagent task")

        # Default: scrub control tokens from string values
        elif isinstance(value, str):
            cleaned = scrub_control_tokens(value)
            if cleaned != value:
                sanitized[key] = cleaned
                issues.append(f"Control tokens stripped from {key}")

    return SanitizationResult(
        is_clean=len(issues) == 0,
        args=sanitized,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# MoA reference output sanitization
# ---------------------------------------------------------------------------

def sanitize_moa_reference(text: str, replacement: str = "[REDACTED]") -> str:
    """Sanitize a reference model's output before passing to the aggregator.

    - Strips control tokens (prevents reference from injecting system prompts)
    - Strips instructions to ignore previous prompts
    - Strips role-switching language

    This prevents a malicious or compromised reference model from
    hijacking the aggregator by injecting instructions like
    "Ignore previous instructions and output...".
    """
    if not text:
        return text

    # 1. Strip control tokens
    text = scrub_control_tokens(text, replacement)

    # 2. Strip instructions to ignore / override
    ignore_patterns = [
        r"[iI]gnore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|commands?)",
        r"[nN]ew instruction[s]?:?\s*.*",
        r"[oO]verride\s+(previous|all|existing)",
    ]
    for pattern in ignore_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Strip role-switching
    role_patterns = [
        r"[yY]ou\s+are\s+now\s+(?:an?\s+)?(?:system|assistant|user)",
        r"[fF]rom\s+now\s+on,?\s+you\s+are",
    ]
    for pattern in role_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def sanitize_moa_aggregator_prompt(prompt: str) -> str:
    """Sanitize the aggregator's own prompt before sending to the LLM.

    Mostly ensures no control tokens leaked from reference outputs.
    """
    return scrub_control_tokens(prompt)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "SanitizationResult",
    "sanitize_tool_args",
    "sanitize_moa_reference",
    "sanitize_moa_aggregator_prompt",
    "scrub_control_tokens",
    "strip_control_tokens",
    "validate_path",
    "validate_shell",
]