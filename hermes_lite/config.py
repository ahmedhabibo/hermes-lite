"""hermes_lite.config — Standalone configuration system (v0.8.0+).

Replaces the previous pattern of scattering os.environ.get() calls across
every module.  Config is loaded once from (in priority order):

1. Environment variables (always take precedence)
2. ``~/.hermes_lite/config.yaml`` (optional YAML file, if present)
3. Built-in defaults

This module has **zero dependency on Hermes Agent** — no imports from
``hermes_tools``, no references to ``~/.hermes/config.yaml``, and no
MCP server configuration.  Hermes-Lite is fully standalone.

Usage
-----
    from hermes_lite.config import get_config

    cfg = get_config()               # singleton, cached
    print(cfg.local_model)           # → "Qwen2.5-Coder-7B-Instruct-IQ3_XS.gguf"
    print(cfg.cloud_model)           # → "z-ai/glm-5.2"

    # Force reload (after changing env or config file at runtime)
    from hermes_lite.config import reload_config
    cfg = reload_config()

Config file format (~/.hermes_lite/config.yaml)
-----------------------------------------------

    local:
      url: http://127.0.0.1:8080/v1
      model: Qwen2.5-Coder-7B-Instruct-IQ3_XS.gguf
    cloud:
      url: https://integrate.api.nvidia.com/v1
      model: z-ai/glm-5.2
    router:
      local_max_complexity: 0.3
    moa:
      preset: null
      timeout_s: 60
    memory:
      max_chars: 800
    observability:
      max_log_bytes: 10485760
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Defaults — the source of truth for all configurable values
# ---------------------------------------------------------------------------

LOCAL_URL_DEFAULT = "http://127.0.0.1:8080/v1"
LOCAL_MODEL_DEFAULT = "Qwen2.5-Coder-7B-Instruct-IQ3_XS.gguf"

CLOUD_URL_DEFAULT = "https://integrate.api.nvidia.com/v1"
CLOUD_MODEL_DEFAULT = "z-ai/glm-5.2"

# Local-first fallback chain (preferred model first)
DEFAULT_FALLBACK_CHAIN = [
    f"local:{LOCAL_MODEL_DEFAULT}",
    "z-ai/glm-5.2",
    "minimaxai/minimax-m3",
    "moonshotai/kimi-k2.6",
    "qwen/qwen3.5-397b-a17b",
    "deepseek-ai/deepseek-v4-flash",
]

# Rate limiting
DEFAULT_RPM = 40
DEFAULT_MAX_RETRIES = 4

# Router (local-first: stay local until score exceeds threshold)
DEFAULT_LOCAL_MAX_COMPLEXITY = 0.7
DEFAULT_ESCALATE_AFTER_FAILURES = 2
DEFAULT_LARGE_PROMPT_CHARS = 2_000
DEFAULT_LARGE_CONTEXT_TOKENS = 4_000
DEFAULT_LARGE_HISTORY_TURNS = 4

# MoA
DEFAULT_MOA_TIMEOUT_S = 60.0
DEFAULT_MOA_REF_TEMP = 0.4
DEFAULT_MOA_AGG_TEMP = 0.2
DEFAULT_MOA_MAX_TOKENS = 4096

# Memory
DEFAULT_MEMORY_MAX_CHARS = 800

# Observability
DEFAULT_MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB

# Subagent
DEFAULT_SUBAGENT_TIMEOUT_S = 180.0
DEFAULT_MAX_ITERATIONS = 12

# Sandbox
DEFAULT_SANDBOX_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class HermesLiteConfig:
    """Central configuration for all hermes_lite modules.

    Every field has a default, so a freshly constructed instance is
    always usable.  The :func:`get_config` factory applies env-var
    overrides and optional YAML config on top of these defaults.
    """

    # Local LLM
    local_url: str = LOCAL_URL_DEFAULT
    local_model: str = LOCAL_MODEL_DEFAULT

    # Cloud LLM
    cloud_url: str = CLOUD_URL_DEFAULT
    cloud_model: str = CLOUD_MODEL_DEFAULT
    cloud_api_key: str = ""
    cloud_api_keys: list[str] = field(default_factory=list)

    # Fallback chain
    fallback_chain: list[str] = field(default_factory=lambda: list(DEFAULT_FALLBACK_CHAIN))

    # Rate limiting
    rpm: int = DEFAULT_RPM
    max_retries: int = DEFAULT_MAX_RETRIES

    # Router
    local_max_complexity: float = DEFAULT_LOCAL_MAX_COMPLEXITY
    escalate_after_failures: int = DEFAULT_ESCALATE_AFTER_FAILURES
    large_prompt_chars: int = DEFAULT_LARGE_PROMPT_CHARS
    large_context_tokens: int = DEFAULT_LARGE_CONTEXT_TOKENS
    large_history_turns: int = DEFAULT_LARGE_HISTORY_TURNS

    # MoA
    moa_preset: Optional[str] = None
    moa_timeout_s: float = DEFAULT_MOA_TIMEOUT_S
    moa_ref_temperature: float = DEFAULT_MOA_REF_TEMP
    moa_agg_temperature: float = DEFAULT_MOA_AGG_TEMP
    moa_max_tokens: int = DEFAULT_MOA_MAX_TOKENS

    # Memory
    memory_max_chars: int = DEFAULT_MEMORY_MAX_CHARS

    # Observability
    max_log_bytes: int = DEFAULT_MAX_LOG_BYTES

    # Subagent
    subagent_timeout_s: float = DEFAULT_SUBAGENT_TIMEOUT_S
    subagent_model: str = "local:Qwen2.5-Coder-7B-Instruct-IQ3_XS.gguf"
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    # Sandbox
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    sandbox_allowlist: Optional[str] = None
    sandbox_blocklist: Optional[str] = None
    sandbox_network: str = "allow"

    # Local tools (send tool_choice to local endpoint)
    local_tools_enabled: bool = False

    # Web tools
    web_search_disabled: bool = False
    web_fetch_disabled: bool = False

    # Prompts / persona
    prompt_override: str = ""
    persona: str = "balanced"

    # WebUI server
    webui_port: int = 3007
    webui_host: str = "0.0.0.0"

    # Auth
    auth_token: str = ""

    # Paths
    home_dir: Path = field(default_factory=lambda: Path.home() / ".hermes_lite")

    # --- derived helpers ---

    @property
    def memory_db_path(self) -> Path:
        return self.home_dir / "memory.db"

    @property
    def log_dir(self) -> Path:
        return self.home_dir / "logs"

    @property
    def config_file_path(self) -> Path:
        return self.home_dir / "config.yaml"

    @property
    def fallback_chain_csv(self) -> str:
        """Comma-joined fallback chain (legacy env shape)."""
        return ",".join(self.fallback_chain)


# ---------------------------------------------------------------------------
# YAML loading (optional, pure-stdlib if pyyaml is missing)
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict on any error."""
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except ImportError:
        # PyYAML not installed — config.yaml is optional
        return {}
    except Exception:
        return {}


def _deep_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Traverse nested dict, returning *default* if any key is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur if cur is not None else default


# ---------------------------------------------------------------------------
# Env-var override helpers
# ---------------------------------------------------------------------------


def _env_first(*names: str, default: str = "") -> str:
    """Return the first non-empty env value among *names*, else *default*."""
    for name in names:
        val = os.environ.get(name)
        if val is not None and str(val).strip() != "":
            return str(val)
    return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "")
    if val:
        return val in ("1", "true", "True", "yes", "YES")
    return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "")
    if not raw:
        return list(default)
    for sep in ("|", ",", ";"):
        if sep in raw:
            return [k.strip() for k in raw.split(sep) if k.strip()]
    return [raw.strip()] if raw.strip() else list(default)


def _parse_csv_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p and p.strip()]


# ---------------------------------------------------------------------------
# Factory: build config from env + YAML + defaults
# ---------------------------------------------------------------------------


def _build_config() -> HermesLiteConfig:
    """Build a HermesLiteConfig from env vars, optional YAML, and defaults."""
    cfg = HermesLiteConfig()

    home_env = os.environ.get("HERMES_LITE_HOME", "")
    if home_env:
        cfg.home_dir = Path(home_env).expanduser()

    yaml_data = _load_yaml(cfg.config_file_path)

    # --- Local LLM ---
    cfg.local_url = _env_str(
        "HERMES_LITE_LOCAL_URL",
        _deep_get(yaml_data, "local", "url", default=LOCAL_URL_DEFAULT),
    )
    cfg.local_model = _env_str(
        "HERMES_LITE_LOCAL_MODEL",
        _deep_get(yaml_data, "local", "model", default=LOCAL_MODEL_DEFAULT),
    )

    # --- Cloud LLM ---
    cfg.cloud_url = _env_str(
        "HERMES_LITE_CLOUD_URL",
        _deep_get(yaml_data, "cloud", "url", default=CLOUD_URL_DEFAULT),
    )
    cfg.cloud_model = _env_str(
        "HERMES_LITE_CLOUD_MODEL",
        _deep_get(yaml_data, "cloud", "model", default=CLOUD_MODEL_DEFAULT),
    )

    single_key = os.environ.get(
        "HERMES_LITE_NVIDIA_API_KEY", ""
    ) or os.environ.get("NVIDIA_API_KEY", "")
    keys_pool = os.environ.get("HERMES_LITE_NVIDIA_API_KEYS", "")
    if keys_pool:
        cfg.cloud_api_keys = _env_list("HERMES_LITE_NVIDIA_API_KEYS", [])
        cfg.cloud_api_key = cfg.cloud_api_keys[0] if cfg.cloud_api_keys else single_key
    elif single_key:
        cfg.cloud_api_key = single_key
        cfg.cloud_api_keys = [single_key]
    else:
        yaml_keys = _deep_get(yaml_data, "cloud", "api_keys", default=[])
        if isinstance(yaml_keys, list) and yaml_keys:
            cfg.cloud_api_key = str(yaml_keys[0])
            cfg.cloud_api_keys = [str(k) for k in yaml_keys]

    # Prefer HERMES_LITE_FALLBACK_CHAIN; accept legacy LITE_FALLBACK_CHAIN
    chain_env = _env_first("HERMES_LITE_FALLBACK_CHAIN", "LITE_FALLBACK_CHAIN", default="")
    if chain_env:
        cfg.fallback_chain = _parse_csv_list(chain_env)
    else:
        yaml_chain = _deep_get(yaml_data, "cloud", "fallback_chain", default=None)
        if isinstance(yaml_chain, list) and yaml_chain:
            cfg.fallback_chain = [str(m) for m in yaml_chain]
        else:
            # Prepend local model to match historical router default chain shape
            local_entry = f"local:{cfg.local_model}"
            if cfg.fallback_chain and not cfg.fallback_chain[0].startswith("local:"):
                cfg.fallback_chain = [local_entry] + list(cfg.fallback_chain)
            elif not cfg.fallback_chain:
                cfg.fallback_chain = [local_entry, cfg.cloud_model]

    cfg.rpm = _env_int(
        "HERMES_LITE_RPM",
        _deep_get(yaml_data, "rate_limit", "rpm", default=DEFAULT_RPM),
    )
    cfg.max_retries = _env_int(
        "HERMES_LITE_MAX_RETRIES",
        _deep_get(yaml_data, "rate_limit", "max_retries", default=DEFAULT_MAX_RETRIES),
    )

    # Router — HERMES_LITE_* preferred, LITE_* accepted for backward compat
    cfg.local_max_complexity = float(
        _env_first(
            "HERMES_LITE_LOCAL_MAX_COMPLEXITY",
            "LITE_LOCAL_MAX_COMPLEXITY",
            default=str(
                _deep_get(
                    yaml_data,
                    "router",
                    "local_max_complexity",
                    default=DEFAULT_LOCAL_MAX_COMPLEXITY,
                )
            ),
        )
    )
    cfg.escalate_after_failures = int(
        _env_first(
            "HERMES_LITE_ESCALATE_AFTER_FAILURES",
            "LITE_ESCALATE_AFTER_FAILURES",
            default=str(
                _deep_get(
                    yaml_data,
                    "router",
                    "escalate_after_failures",
                    default=DEFAULT_ESCALATE_AFTER_FAILURES,
                )
            ),
        )
    )
    cfg.large_prompt_chars = int(
        _env_first(
            "HERMES_LITE_LARGE_PROMPT_CHARS",
            "LITE_LARGE_PROMPT_CHARS",
            default=str(
                _deep_get(
                    yaml_data,
                    "router",
                    "large_prompt_chars",
                    default=DEFAULT_LARGE_PROMPT_CHARS,
                )
            ),
        )
    )
    cfg.large_context_tokens = int(
        _env_first(
            "HERMES_LITE_LARGE_CONTEXT_TOKENS",
            "LITE_LARGE_CONTEXT_TOKENS",
            default=str(
                _deep_get(
                    yaml_data,
                    "router",
                    "large_context_tokens",
                    default=DEFAULT_LARGE_CONTEXT_TOKENS,
                )
            ),
        )
    )
    cfg.large_history_turns = int(
        _env_first(
            "HERMES_LITE_LARGE_HISTORY_TURNS",
            "LITE_LARGE_HISTORY_TURNS",
            default=str(
                _deep_get(
                    yaml_data,
                    "router",
                    "large_history_turns",
                    default=DEFAULT_LARGE_HISTORY_TURNS,
                )
            ),
        )
    )

    moa_preset_env = os.environ.get("HERMES_LITE_MOA_PRESET", "")
    yaml_moa_preset = _deep_get(yaml_data, "moa", "preset", default=None)
    cfg.moa_preset = moa_preset_env or yaml_moa_preset or None

    cfg.moa_timeout_s = _env_float(
        "HERMES_LITE_MOA_TIMEOUT_S",
        _deep_get(yaml_data, "moa", "timeout_s", default=DEFAULT_MOA_TIMEOUT_S),
    )
    cfg.moa_ref_temperature = _env_float(
        "HERMES_LITE_MOA_REF_TEMPERATURE",
        _deep_get(yaml_data, "moa", "ref_temperature", default=DEFAULT_MOA_REF_TEMP),
    )
    cfg.moa_agg_temperature = _env_float(
        "HERMES_LITE_MOA_AGG_TEMPERATURE",
        _deep_get(yaml_data, "moa", "agg_temperature", default=DEFAULT_MOA_AGG_TEMP),
    )
    cfg.moa_max_tokens = _env_int(
        "HERMES_LITE_MOA_MAX_TOKENS",
        _deep_get(yaml_data, "moa", "max_tokens", default=DEFAULT_MOA_MAX_TOKENS),
    )

    cfg.memory_max_chars = int(
        _env_first(
            "HERMES_LITE_MEMORY_MAX_CHARS",
            "LITE_MEMORY_MAX_CHARS",
            default=str(
                _deep_get(
                    yaml_data, "memory", "max_chars", default=DEFAULT_MEMORY_MAX_CHARS
                )
            ),
        )
    )

    cfg.max_log_bytes = _env_int(
        "HERMES_LITE_MAX_LOG_BYTES",
        _deep_get(yaml_data, "observability", "max_log_bytes", default=DEFAULT_MAX_LOG_BYTES),
    )

    cfg.subagent_timeout_s = _env_float(
        "HERMES_LITE_SUBAGENT_TIMEOUT",
        _deep_get(yaml_data, "subagent", "timeout_s", default=DEFAULT_SUBAGENT_TIMEOUT_S),
    )
    cfg.subagent_model = _env_str(
        "HERMES_LITE_SUBAGENT_MODEL",
        _deep_get(yaml_data, "subagent", "model", default=cfg.subagent_model),
    )
    cfg.max_iterations = _env_int(
        "HERMES_LITE_MAX_ITERATIONS",
        _deep_get(yaml_data, "orchestrator", "max_iterations", default=DEFAULT_MAX_ITERATIONS),
    )

    cfg.sandbox_timeout = _env_int(
        "HERMES_LITE_SANDBOX_TIMEOUT",
        _deep_get(yaml_data, "sandbox", "timeout", default=DEFAULT_SANDBOX_TIMEOUT),
    )
    allow = os.environ.get("HERMES_LITE_SANDBOX_ALLOWLIST")
    cfg.sandbox_allowlist = allow if allow else None
    block = os.environ.get("HERMES_LITE_SANDBOX_BLOCKLIST")
    cfg.sandbox_blocklist = block if block else None
    cfg.sandbox_network = _env_str(
        "HERMES_LITE_SANDBOX_NETWORK",
        _deep_get(yaml_data, "sandbox", "network", default="allow"),
    )

    cfg.local_tools_enabled = _env_bool(
        "HERMES_LITE_LOCAL_TOOLS",
        _deep_get(yaml_data, "local", "tools_enabled", default=False),
    )

    # Web tools
    cfg.web_search_disabled = _env_bool(
        "HERMES_LITE_WEB_SEARCH_DISABLED",
        _deep_get(yaml_data, "tools", "web_search_disabled", default=False),
    )
    cfg.web_fetch_disabled = _env_bool(
        "HERMES_LITE_WEB_FETCH_DISABLED",
        _deep_get(yaml_data, "tools", "web_fetch_disabled", default=False),
    )

    # Prompts / persona
    cfg.prompt_override = _env_str(
        "HERMES_LITE_PROMPT_OVERRIDE",
        _deep_get(yaml_data, "prompts", "override", default=""),
    )
    cfg.persona = _env_str(
        "HERMES_LITE_PERSONA",
        _deep_get(yaml_data, "prompts", "persona", default="balanced"),
    )

    # WebUI
    cfg.webui_port = _env_int(
        "HERMES_LITE_WEBUI_PORT",
        _deep_get(yaml_data, "webui", "port", default=3007),
    )
    cfg.webui_host = _env_str(
        "HERMES_LITE_WEBUI_HOST",
        _deep_get(yaml_data, "webui", "host", default="0.0.0.0"),
    )

    # Auth token
    cfg.auth_token = _env_str(
        "HERMES_LITE_AUTH_TOKEN",
        _deep_get(yaml_data, "auth", "token", default=""),
    )

    return cfg


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[HermesLiteConfig] = None
_config_lock = threading.Lock()


def get_config() -> HermesLiteConfig:
    """Return the config singleton, building it on first call."""
    global _config_instance
    with _config_lock:
        if _config_instance is None:
            _config_instance = _build_config()
        return _config_instance


def reload_config() -> HermesLiteConfig:
    """Discard the cached config and build a fresh instance.

    Useful when env vars change at runtime (e.g. after a .env reload).
    """
    global _config_instance
    with _config_lock:
        _config_instance = _build_config()
        return _config_instance


__all__ = [
    "HermesLiteConfig",
    "get_config",
    "reload_config",
    # Defaults (re-exported so other modules can import from one place)
    "LOCAL_URL_DEFAULT",
    "LOCAL_MODEL_DEFAULT",
    "CLOUD_URL_DEFAULT",
    "CLOUD_MODEL_DEFAULT",
    "DEFAULT_FALLBACK_CHAIN",
    "DEFAULT_RPM",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_LOCAL_MAX_COMPLEXITY",
    "DEFAULT_ESCALATE_AFTER_FAILURES",
    "DEFAULT_LARGE_PROMPT_CHARS",
    "DEFAULT_LARGE_CONTEXT_TOKENS",
    "DEFAULT_LARGE_HISTORY_TURNS",
    "DEFAULT_MOA_TIMEOUT_S",
    "DEFAULT_MOA_REF_TEMP",
    "DEFAULT_MOA_AGG_TEMP",
    "DEFAULT_MOA_MAX_TOKENS",
    "DEFAULT_MEMORY_MAX_CHARS",
    "DEFAULT_MAX_LOG_BYTES",
    "DEFAULT_SUBAGENT_TIMEOUT_S",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_SANDBOX_TIMEOUT",
]
