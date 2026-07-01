# CHANGELOG

## 0.5.0 — MoA Orchestration + CLI Entry Point
**Released**: 2026-07-01

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

---

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

---

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

---

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

---

## 0.1.0 — Initial Demo
**Released**: 2026-06-13

- Basic CLI with `echo` and `calculator` tools
- SQLite memory layer
- Plugin registry with Pydantic validation