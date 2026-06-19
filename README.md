# Hermes-Lite v0.2

> Lightweight local-first agent framework — Qwen 3B LLM, 6 essential tools, routing controller, sandboxed execution, persistent memory, and observability.

Designed for macOS (Apple Silicon, 8 GB). Runs fully local. No GPU required.

---

## Quick Start

```bash
# 1. Install llama.cpp (LLM server)
brew install llama.cpp

# 2. Download the model
huggingface-cli download Qwen/Qwen3.5-3B-GGUF qwen3.5-3b-q4_k_m.gguf --local-dir ~/.cache/llama.cpp

# 3. Start the LLM server
llama-server -m ~/.cache/llama.cpp/qwen3.5-3b-q4_k_m.gguf \
    --port 8081 --n-gpu-layers -1 --ctx-size 8192

# 4. Install hermes-lite
git clone https://github.com/your-org/hermes-lite.git
cd hermes-lite
pip install -e ".[test]"

# 5. Run the CLI
python -m hermes_lite
```

That's it. Start chatting in under 10 minutes.

---

## Examples

```
$ python -m hermes_lite
⚡ _local_ · 1 turn(s)
I can help with file reads, searches, and more. What would you like?

> find odoo modules
⚡ search_files('*.py')
⚡ _local_ · 2 turn(s) · tools: `search_files`
I found 43 Python modules in the hermes_lite directory.

> summarize README
⚡ read_file('README.md')
⚡ _local_ · 2 turn(s) · tools: `read_file`
Hermes-Lite is a local agent framework with 6 built-in tools...

> refactor this function
⚡ _local_ · 1 turn(s)
☁️ _cloud_ · routed to llm for complex reasoning
```

---

## Features

| Area | What it does |
|------|-------------|
| **Tool registry** | 6 built-in essentials: `read_file`, `search_files`, `terminal`, `memory`, `web_search`, `web_fetch`. Pydantic-validated dispatch. Extensible via `ToolDefinition`. |
| **LLM layer** | OpenAI-compatible chat API. Default: local Qwen 3B GGUF via llama.cpp. Supports remote fallback. |
| **Tool loop** | Two-tier loop: LLM calls tools → results fed back → LLM responds. Max 4 iterations, repeated-error and malformed-JSON guards. |
| **Router** | LiteRouter classifies prompts by complexity. Simple queries → `_local_` tier (1–3 B model). Complex reasoning → `_cloud_` tier. Consecutive-failure escalation with linear backoff. |
| **Sandbox** | `terminal` tool runs commands in a macOS sandbox (`sandbox-exec`). Timeout-safe with process lifecycle management. |
| **Memory** | SQLite bridge for cross-session facts. `add`, `replace`, `remove`, `list` — unique-match semantics. Loaded into every prompt (800 char cap). |
| **Sub-agent** | Spawn parallel `delegate_task` sub-agents. Isolated context + toolset per child. Nested orchestration (max 2 levels). |
| **Observability** | Per-turn JSONL logging, rotation at 10 MB, `python -m hermes_lite stats` for session summary. |
| **CLI** | prompt_toolkit + Rich terminal. Ctrl+C/D, `!tool {args}` direct invocation, `/tools`, `/history`, `/help`. |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    Router                       │
│  LiteRouter: classify by complexity → local/cloud│
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│                  LLM Layer                       │
│  Qwen 3B (local)  │  Remote (cloud fallback)     │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│               Tool Loop                          │
│  2-tier: LLM → tool → result → LLM → response   │
│  Max 4 iterations, repeated-error, malformed-JSON│
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│           Tool Layer (PluginRegistry)            │
│ read_file │ search_files │ terminal │ memory │   │
│ web_search │ web_fetch │ subagent                │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│              Sandbox / Backend                   │
│  sandbox-exec   │  Hermes MCP   │  curl   │      │
└──────────────────────────────────────────────────┘
```

---

## Configuration

Set these environment variables (or edit `hermes_lite/router.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `LITE_LOCAL_MAX_COMPLEXITY` | `0.6` | Max complexity score for local routing |
| `LITE_LOCAL_MODEL` | `local:qwen-3b` | Model ID for local tier |
| `LITE_CLOUD_MODEL` | `openai:gpt-4o-mini` | Model ID for cloud tier |
| `LITE_LLM_BASE_URL` | `http://localhost:8081/v1` | LLM server endpoint |
| `LITE_MEMORY_MAX_CHARS` | `800` | Max chars of memory injected into prompt |

---

## Test Suite

304 tests, all passing. Run with:

```bash
cd hermes-lite
pip install -e ".[test]"
python -m pytest tests/ -v
```

Tests cover: registry (48), memory (47), orchestrator (31), tool loop (15), tools-essentials (55), LLM (5), router (37), sandbox (30), sub-agent (18), memory bridge (10), observability (6), e2e smoke (5).

---

## CHANGELOG

### 0.2.0 — Local Qwen 3B + 6 essential tools + router + sandbox
- Tool registry with 6 essentials (read_file, search_files, terminal, memory, web_search, web_fetch)
- LiteRouter: prompt complexity classifier for local/cloud tier routing
- ToolLoop: two-tier tool-calling loop with termination guards
- Sandboxed terminal execution (macOS sandbox-exec)
- Memory Bridge: cross-session persistent facts (SQLite)
- Subagent: parallel tool-spawning with isolated context
- Observability: per-turn JSONL logging + stats CLI
- 304 tests, all passing on macOS + 8 GB