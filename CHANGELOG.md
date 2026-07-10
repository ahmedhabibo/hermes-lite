# CHANGELOG

## 0.9.0 — Release-quality packaging polish

### Highlights
- Hermes-Lite ships as a clean standalone PyPI package — zero Hermes Agent runtime dependency
- Test suite green: **467 collected, 464 passed (3 slow deselected), 0 failed** — verified post-bump
- Fresh `dist/` build validated with `twine check` (wheel + sdist)
- Clean working tree: `egg-info/`, `dist/`, `build/` and transient `smoke/latency.json` moved to `.gitignore`
- `pyproject.toml` metadata hardened: `Development Status :: 5 - Production/Stable`, standalone keyword + long description

### Changed
- `pyproject.toml`: `version` → `0.9.0`; Production/Stable classifier; standalone/ddgs keywords
- `README.md`: test-count badge refreshed to reflect 467-collection baseline
- `.gitignore`: build artefacts stripped from tracking (no more churn from `pip install -e .` / `python -m build`)

### Verified
- `pytest tests/ -q -m "not slow"` → 464 passed, 3 deselected (slow), 0 failed
- `python -m build` → wheel + sdist produced in `dist/`
- `python -m twine check dist/*` → PASSED for both artefacts
- Smoke run against `z-ai/glm-5.2` via NVIDIA NIM Free API (see `smoke/latency.json`, gitignored)

## 0.8.0 — Decouple from Hermes Agent (Standalone Local-First Agent)

### Summary
Hermes-Lite is now fully standalone — zero dependency on Hermes Agent runtime.
Config, memory, routing, and web tool backends are all self-contained.

### Config (standalone since v0.8.0)
- `~/.hermes_lite/config.yaml` replaces `~/.hermes/config.yaml`
- `HERMES_LITE_*` env vars replace `HERMES_*` vars
- `HermesLiteConfig` dataclass with singleton accessor (`get_config()`, `reload_config()`)
- Optional YAML config (PyYAML optional, pure-stdlib fallback)
- `is_standalone` property always returns `True`

### Memory (standalone since v0.8.0)
- `~/.hermes_lite/memory.db` replaces `~/.hermes/memory.db`
- `MemoryBridge` SQLite layer for cross-session memory (memory + user targets)
- Async SQLite pool for sessions/messages (separate from Hermes Agent state.db)
- No Hindsight integration — standalone only

### Routing (standalone since v0.7+)
- Local-first fallback chain: `local:Qwen2.5-Coder-7B-Instruct-IQ3_XS.gguf` as preferred
- Cloud escalation via NIM API when complexity > threshold or intent prefix matches
- `LiteRouter` with complexity scoring (prompt length, context tokens, history, keywords)
- `LITE_*` env vars for router tuning (separate from Hermes Agent)

### Web Tools (standalone since v0.8.0)
- `web_search`: DuckDuckGo via `ddgs` (no `hermes_tools` dependency)
- `web_fetch`: `trafilatura` (preferred) + `httpx`/`html2text` fallback
- Graceful degradation when optional deps missing — helpful message, no fake data
- Disable via `HERMES_LITE_WEB_SEARCH_DISABLED=1` / `HERMES_LITE_WEB_FETCH_DISABLED=1`

### Tests
- Updated `TestWebSearch`, `TestWebFetch`, `TestToolResultContract` in
  `test_tools_essentials.py` to mock new standalone backends (ddgs/trafilatura)
  instead of legacy `_ht_web_search`/`_ht_web_extract` (hermes_tools)
- **467 passed, 0 failures** (2 pre-existing aiosqlite cleanup warnings)

## 0.6.0 — Security Hardening + Streaming + Docker + CI
**Released**: 2026-07-03

### Added — Security (Phase 1, Items 1-7)
- **#1 API Key Exhaustion**: `AllKeysExhausted` exception with `is_exhausted()` cooldown math, cloud→local fallback when all keys exhausted
- **#2 Auth/Authorization**: `ToolAuthError`, `dangerous` flag on tools, `--auth-token` CLI, built-in tools flagged
- **#3 Input Sanitization**: `sanitize.py` — control-token scrubbing, path-traversal blocking, shell-injection heuristics, MoA sanitization
- **#4 Rate-limit Hardening**: Per-key token buckets, full-jitter exponential backoff, `z-ai/glm-5.2` default, 5-model fallback chain
- **#5 Sandbox Tightening**: Command allowlist (`HERMES_LITE_SANDBOX_ALLOWLIST`), blocklist, `CommandBlockedError`, `_check_command_allowed()` enforced in `run_sandboxed`
- **#6 Secret Redaction**: `_sanitize_env()` strips secrets from child env, `_redact_in_text()` redacts secrets from stdout/stderr/audit log
- **#7 Subagent Isolation**: Child `os.environ` sanitized (secrets stripped), parent env restored after subagent run

### Added — Features
- **Streaming**: `chat_stream()` async generator — token-by-token streaming for both cloud and local endpoints
- **Docker**: Dockerfile (Python 3.11-slim), docker-compose.yml (interactive + test services)
- **CI**: GitHub Actions workflow — Python 3.9/3.11/3.12 matrix, pytest + coverage + ruff lint, concurrency cancel
- **.env loading**: `python-dotenv` auto-load on package import for multi-key rotation
- **PyPI metadata**: Authors, classifiers, keywords, project URLs added to `pyproject.toml`

### Changed
- **Default cloud model**: `z-ai/glm-5.2` (was `minimaxai/minimax-m3`)
- **Default local model**: `Qwen2.5-Coder-7B-Instruct-IQ3_XS.gguf` (Bartowski quant, 3.1GB — was `gemma-4-E2B-it-abliterated-Q4_K_M.gguf`)
- **Fallback chain**: 5 models — `z-ai/glm-5.2 → minimaxai/minimax-m3 → moonshotai/kimi-k2.6 → qwen/qwen3.5-397b-a17b → deepseek-ai/deepseek-v4-flash`
- **launchd plist**: Updated to `ngl=28`, `ctx=65536`, Q8_0 KV cache for 8GB M1 safety

### Tests
- **432 → 467** (+30 sandbox security, +1 subagent env isolation, +4 streaming)
- New files: `test_sandbox_security.py`, `test_streaming.py`, `conftest.py`
- 20 test files, 16 source modules

## 0.5.0 — MoA Orchestration + CLI Entry Point

### Added
- **Mixture-of-Agents (MoA) engine** (`hermes_lite/moa.py`): Run 3-5 diverse LLMs in parallel, then aggregate with a synthesis model
- **5 built-in MoA presets**: `council`, `speed`, `verification`, `coding`, `creative` — all using verified NIM free-tier models
- **CLI commands**: `/moa` (status), `/moa <preset>` (activate), `/moa off` (deactivate)
- **Auto-activation**: `HERMES_LITE_MOA_PRESET` env var to enable MoA on startup
- **CLI entry point**: `pip install -e .` → `hermes-lite` command with `--version` and `stats` subcommands
- **Version display**: Dynamic version from `pyproject.toml` shown in welcome banner

### Changed
- **Default timeout**: MoA reference timeout increased from 30s → 60s for large models
- **Preset models**: Removed non-existent NIM models (nemotron-3-super, stepfun-3.7, mistral-medium-3.5); all presets now use only verified free-tier models
- **Aggregator fallback**: If aggregator fails, returns best reference response instead of retrying same model

### Fixed
- **MoA crash**: `log_turn()` signature mismatch (passed invalid kwargs)
- **Duplicate header**: MoA path no longer renders `☁️ cloud · 1 turn(s)` twice
## 0.4.0 — Cloud-First NIM Pivot
**Released**: 2026-06-30

### Added
- **Rate limiting**: Token bucket (40 RPM default, configurable via `HERMES_LITE_RPM`)
- **API key rotation**: Round-robin pool from `HERMES_LITE_NVIDIA_API_KEYS` (comma-separated), 60s cooldown per key
- **Exponential backoff**: Retries on 429/500/502/503 with delays 1s/2s/4s/8s (max 16s)
- **Cloud-first routing**: NIM models as default provider; `local:` prefix preserved for backward compatibility
- **Fallback chain**: `minimaxai/minimax-m3 → moonshotai/kimi-k2.6 → qwen/qwen3.5-397b-a17b → deepseek-ai/deepseek-v4-flash`
- **Subagent default**: Changed to `deepseek-ai/deepseek-v4-flash` (cloud NIM model)

### Changed
- **Router**: `DEFAULT_FALLBACK_CHAIN` now cloud-first; tier selection derives from chain head
- **LLM layer**: Added `stepfun-ai/` cloud prefix support
- **Tests**: Updated assertions for cloud-first behavior
## 0.3.0 — Hackathon-Ready Polish
**Released**: 2026-06-19

### Added
- **Local model upgrade**: Qwen 2.5 7B Instruct Q4_K_M (from 3B)
- **Qwen tool-call parser**: Native support for Qwen's tool calling format
- **LICENSE**: MIT license
- **CONTRIBUTING.md**: Contribution guidelines
- **Badges**: Tests, coverage, version badges in README
- **Demo script**: `scripts/demo.py` for quick showcase

### Changed
- **Default tools**: 6 essentials (`read_file`, `search_files`, `terminal`, `memory`, `web_search`, `web_fetch`)
- **Version**: Bumped to 0.3.0
## 0.2.0 — Local Qwen 3B + 6 Essential Tools + Router + Sandbox
**Released**: 2026-06-19

### Added
- **Tool registry** with 6 essential built-in tools: `read_file`, `search_files`, `terminal`, `memory`, `web_search`, `web_fetch`
- **LiteRouter**: Prompt complexity classifier for automatic local/cloud tier routing
- **ToolLoop**: Two-tier tool-calling loop with termination guards (max 4 iterations, repeated-error detection, malformed-JSON handling)
- **Sandboxed terminal**: `terminal` tool runs via macOS `sandbox-exec` with timeout-safe process lifecycle
- **Memory Bridge**: Cross-session persistent facts stored in SQLite, injected into prompts (800 char cap)
- **Subagent**: Parallel `delegate_task` spawning with isolated context per child
- **Observability**: Per-turn JSONL logging with 10 MB rotation, `python -m hermes_lite stats` CLI command
- **Test suite**: 304 tests covering registry, memory, orchestrator, tools, router, sandbox, subagent

### Changed
- Default tools changed from `echo`/`calculator`/`save_note` to 6 essentials
- Version bumped to 0.2.0

### Removed
- Retired `echo`, `calculator`, `save_note` default tools
## 0.1.0 — Initial Demo
**Released**: 2026-06-13

- Basic CLI with `echo` and `calculator` tools
- SQLite memory layer
- Plugin registry with Pydantic validation