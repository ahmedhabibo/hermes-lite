# CHANGELOG

## 0.6.0 — Rate-Limit Hardening + GLM-5.2 Model Support
**Released**: 2026-07-02

### Added
- **Per-key rate limiting**: Each API key gets its own token-bucket RateLimiter (40 RPM default) so that exhaustion of one key doesn't block others.
- **Jittered exponential backoff**: Full jitter on backoff delays (0 to backoff seconds) to reduce thundering herd during rate-limit or API errors.
- **Key-aware rotation**: When a key hits a 429/401/403, it's marked as failed and the next key in the pool is used, with its own rate limiter.
- **Exhaustion detection**: `APIKeyRotator.is_exhausted()` now correctly reports when all keys are in cooldown.
- **GLM-5.2 model support**: `z-ai/glm-5.2` added as the primary default cloud model, replacing `minimaxai/minimax-m3`.
- **Cloud prefix recognition**: Added `z-ai/` to `_CLOUD_PREFIXES` so GLM-5.2 routes to the cloud endpoint correctly.

### Changed
- **Default cloud model**: Changed from `minimaxai/minimax-m3` to `z-ai/glm-5.2` (CLOUD_MODEL_DEFAULT, DEFAULT_FALLBACK_CHAIN).
- **Fallback chain**: Now 5 models: `z-ai/glm-5.2 → minimaxai/minimax-m3 → moonshotai/kimi-k2.6 → qwen/qwen3.5-397b-a17b → deepseek-ai/deepseek-v4-flash`.
- **MoA presets**: All 5 presets (council, speed, verification, creative, coding) updated to use `z-ai/glm-5.2` as primary reference and aggregator where applicable.
- **RateLimiter scope**: Changed from a single module-level singleton to a list of per-key instances, indexed by the key rotator.
- **Backoff algorithm**: Added jitter (full jitter) to exponential backoff for both rate-limit and server errors.
- **Logging**: Enhanced debug and warning logs to show jittered backoff values and key rotation details.
- **Version fallbacks**: Updated hardcoded `0.5.0` fallbacks to `0.6.0` in `orchestrator.py` and `__main__.py`.

### Fixed
- **`finish_reason` AttributeError**: `ChatCompletionMessage` has no `finish_reason` — now correctly accessed via `choice.finish_reason` instead of `msg.finish_reason`.
- **`web_search`/`web_fetch` crash in standalone mode**: When `hermes_tools` is not importable, handlers now return `_ok()` with an informative message instead of `_err()`, preventing the orchestrator's `repeated_error` loop from triggering.
- **Key exhaustion edge case**: When all keys are exhausted, the `AllKeysExhausted` exception now reports the correct cooldown time (time until the earliest key recovers).
- **Test compatibility**: Updated existing rate-limiter tests to work with the new per-key design (tests still pass).

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