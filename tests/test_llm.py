"""tests/test_llm.py — Routing + ChatRequest tests for hermes_lite.llm."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from hermes_lite.llm import (
    ChatRequest,
    ChatResponse,
    Tier,
    _pick_client_and_model,
    tool_def,
)


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


def test_route_local_prefix():
    """``local:<m>`` always resolves to local tier."""
    client, model, tier = _pick_client_and_model("local:qwen.gguf")
    # We can't easily assert on the AsyncOpenAI instance, but we can on tier+model
    assert tier == "local"
    assert model == "qwen.gguf"


def test_route_local_bare():
    """Bare qwen-style names default to local."""
    _, model, tier = _pick_client_and_model("qwen2.5-3b-instruct-q4_k_m.gguf")
    assert tier == "local"
    assert model == "qwen2.5-3b-instruct-q4_k_m.gguf"


@pytest.mark.parametrize("prefix", ["nvidia/", "minimaxai/", "moonshotai/", "qwen/", "deepseek-ai/"])
def test_route_cloud_prefix(prefix):
    _, model, tier = _pick_client_and_model(f"{prefix}mymodel")
    assert tier == "cloud"
    assert model == f"{prefix}mymodel"


def test_route_cloud_requires_api_key(monkeypatch):
    monkeypatch.delenv("HERMES_LITE_NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    # Patching the OpenAI function is heavy; we just check the helper raises clearly
    from hermes_lite.llm import _cloud_client

    with pytest.raises(RuntimeError, match="HERMES_LITE_NVIDIA_API_KEY"):
        _cloud_client()


# ---------------------------------------------------------------------------
# ChatRequest dataclass
# ---------------------------------------------------------------------------


def test_chat_request_defaults():
    req = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="local:test",
    )
    assert req.tier is None  # auto-resolved
    assert req.temperature == 0.2
    assert req.max_tokens == 512
    assert req.tools == []
    assert req.tool_choice == "auto"


def test_chat_request_frozen():
    req = ChatRequest(messages=[], model="local:x")
    with pytest.raises(Exception):
        req.model = "local:y"  # frozen dataclass rejects


# ---------------------------------------------------------------------------
# tool_def helper
# ---------------------------------------------------------------------------


def test_tool_def_structure():
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    td = tool_def("read_file", "Read a file.", schema)
    assert td["type"] == "function"
    assert td["function"]["name"] == "read_file"
    assert td["function"]["description"] == "Read a file."
    assert td["function"]["parameters"] == schema


# ---------------------------------------------------------------------------
# Env-overrides (resolution)
# ---------------------------------------------------------------------------


def test_local_endpoint_env_override(monkeypatch):
    monkeypatch.setenv("HERMES_LITE_LOCAL_URL", "http://192.168.1.10:9090/v1")
    from hermes_lite.llm import _resolve_local

    url, _ = _resolve_local()
    assert url == "http://192.168.1.10:9090/v1"


def test_cloud_endpoint_env_override(monkeypatch):
    monkeypatch.setenv("HERMES_LITE_CLOUD_MODEL", "qwen/qwen3.5-397b-a17b")
    monkeypatch.setenv("HERMES_LITE_NVIDIA_API_KEY", "test-key")
    from hermes_lite.llm import _resolve_cloud

    url, model, key = _resolve_cloud()
    assert model == "qwen/qwen3.5-397b-a17b"
    assert key == "test-key"
