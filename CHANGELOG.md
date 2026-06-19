# CHANGELOG

## 0.2.0 — Local Qwen 3B + 6 essential tools + router + sandbox

**Released**: 2026-06-19

### Added
- **Tool registry** with 6 essential built-in tools: `read_file`, `search_files`, `terminal`, `memory`, `web_search`, `web_fetch`
- **LiteRouter**: prompt complexity classifier for automatic local/cloud tier routing
- **ToolLoop**: two-tier tool-calling loop with termination guards (max 4 iterations, repeated-error detection, malformed-JSON handling)
- **Sandboxed terminal**: `terminal` tool runs via macOS `sandbox-exec` with timeout-safe process lifecycle
- **Memory Bridge**: cross-session persistent facts stored in SQLite, injected into prompts (800 char cap)
- **Subagent**: parallel `delegate_task` spawning with isolated context per child
- **Observability**: per-turn JSONL logging with 10 MB rotation, `python -m hermes_lite stats` CLI command
- **Test suite**: 304 tests covering registry, memory, orchestrator, tools, router, sandbox, subagent

### Changed
- Default tools changed from `echo`/`calculator`/`save_note` to 6 essentials
- Version bumped to 0.2.0

### Removed
- Retired `echo`, `calculator`, `save_note` default tools

---

## 0.1.0 — Initial demo

**Released**: 2026-06-13

- Basic CLI with `echo` and `calculator` tools
- SQLite memory layer
- Plugin registry with Pydantic validation