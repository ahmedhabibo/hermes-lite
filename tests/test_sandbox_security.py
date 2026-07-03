"""Tests for sandbox security hardening: secret scrubbing, command allowlist, log redaction."""

import os
import pytest
from unittest import mock

from hermes_lite.sandbox import (
    run_sandboxed,
    SandboxError,
    CommandBlockedError,
    _is_secret_env,
    _sanitize_env,
    _redact_in_text,
    _collect_env_secrets,
)


class TestSecretEnvDetection:
    """Tests for _is_secret_env — identifies env vars containing secrets."""

    @pytest.mark.parametrize("name", [
        "API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HERMES_LITE_NVIDIA_API_KEY",
        "AUTH_TOKEN",
        "ACCESS_TOKEN",
        "MY_SECRET",
        "PASSWORD",
        "PRIVATE_KEY",
        "SOME_CREDENTIAL",
    ])
    def test_detects_secret_patterns(self, name):
        assert _is_secret_env(name) is True

    @pytest.mark.parametrize("name", [
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "TERM",
        "MY_API_KEY_URL",  # contains API_KEY
    ])
    def test_detects_non_secret(self, name):
        # MY_API_KEY_URL contains "API_KEY" so it should be detected
        if "API_KEY" in name.upper():
            assert _is_secret_env(name) is True
        else:
            assert _is_secret_env(name) is False


class TestSanitizeEnv:
    """Tests for _sanitize_env — strips secrets, passes allowlisted vars."""

    def test_strips_secrets_from_env(self):
        env = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "OPENAI_API_KEY": "sk-secret123",
            "HERMES_LITE_NVIDIA_API_KEY": "nv-secret",
            "TERM": "xterm",
        }
        result = _sanitize_env(env)
        assert "OPENAI_API_KEY" not in result
        assert "HERMES_LITE_NVIDIA_API_KEY" not in result
        assert result.get("PATH") == "/usr/bin"
        assert result.get("HOME") == "/home/user"
        assert result.get("TERM") == "xterm"

    def test_only_passes_allowlisted_vars(self):
        env = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "UNKNOWN_VAR": "should-not-pass",
        }
        result = _sanitize_env(env)
        assert "UNKNOWN_VAR" not in result
        assert "PATH" in result
        assert "HOME" in result

    def test_custom_passthrough(self):
        env = {"PATH": "/usr/bin", "HOME": "/home", "CUSTOM": "val"}
        result = _sanitize_env(env, passthrough=("PATH",))
        assert "PATH" in result
        assert "HOME" not in result
        assert "CUSTOM" not in result


class TestRedactInText:
    """Tests for _redact_in_text — replaces secret values with [REDACTED]."""

    def test_redacts_known_secret(self):
        text = "my key is sk-secret123 and token is abc456"
        result = _redact_in_text(text, ("sk-secret123", "abc456"))
        assert "sk-secret123" not in result
        assert "abc456" not in result
        assert "[REDACTED]" in result

    def test_no_secrets_no_change(self):
        text = "no secrets here"
        result = _redact_in_text(text, ("sk-secret",))
        assert result == text

    def test_empty_secrets_no_change(self):
        text = "some text"
        result = _redact_in_text(text, ())
        assert result == text


class TestCollectEnvSecrets:
    """Tests for _collect_env_secrets — extracts actual secret values."""

    def test_collects_secret_values(self):
        env = {
            "OPENAI_API_KEY": "sk-abc123",
            "PATH": "/usr/bin",
            "HERMES_LITE_NVIDIA_API_KEY": "nv-secret",
        }
        secrets = _collect_env_secrets(env)
        assert "sk-abc123" in secrets
        assert "nv-secret" in secrets
        assert "/usr/bin" not in secrets

    def test_empty_env_no_secrets(self):
        secrets = _collect_env_secrets({})
        assert secrets == ()

    def test_no_secret_values_in_env(self):
        env = {"PATH": "/usr/bin", "HOME": "/home"}
        secrets = _collect_env_secrets(env)
        assert secrets == ()


class TestCommandAllowlist:
    """Tests for HERMES_LITE_SANDBOX_ALLOWLIST."""

    def test_blocked_command_raises(self, monkeypatch):
        monkeypatch.setenv("HERMES_LITE_SANDBOX_ALLOWLIST", "echo:true")
        with pytest.raises(CommandBlockedError):
            run_sandboxed("ls", args=["-la"], timeout=5)

    def test_allowed_command_passes(self, monkeypatch):
        monkeypatch.setenv("HERMES_LITE_SANDBOX_ALLOWLIST", "echo:true")
        result = run_sandboxed("echo", args=["hello"], timeout=10)
        assert result.exit_code == 0
        assert "hello" in result.stdout


class TestCommandBlocklist:
    """Tests for HERMES_LITE_SANDBOX_BLOCKLIST."""

    def test_blocklist_blocks_command(self, monkeypatch):
        monkeypatch.setenv("HERMES_LITE_SANDBOX_BLOCKLIST", "rm:dd:mkfs")
        with pytest.raises(CommandBlockedError):
            run_sandboxed("rm", args=["-rf", "/"], timeout=5)

    def test_blocklist_takes_precedence_over_allowlist(self, monkeypatch):
        monkeypatch.setenv("HERMES_LITE_SANDBOX_ALLOWLIST", "echo:rm:true")
        monkeypatch.setenv("HERMES_LITE_SANDBOX_BLOCKLIST", "rm")
        with pytest.raises(CommandBlockedError):
            run_sandboxed("rm", args=["-rf", "/"], timeout=5)


class TestSecretScrubbingInRunSandboxed:
    """Integration tests: verify secrets are stripped from child env."""

    def test_secret_not_in_child_env(self, monkeypatch):
        """Ensure API keys don't leak to child process environment."""
        monkeypatch.setenv("MY_API_KEY", "sk-test-secret-12345")
        monkeypatch.setenv("HERMES_LITE_SANDBOX_ALLOWLIST", "env:printenv")
        result = run_sandboxed(
            "/usr/bin/env",
            args=[],
            timeout=10,
        )
        assert "sk-test-secret-12345" not in result.stdout
