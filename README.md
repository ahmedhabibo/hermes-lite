# Hermes-Lite ⚡

[![Tests](https://img.shields.io/badge/tests-312%20passing-brightgreen)](./tests)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](./pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![Model: Qwen 2.5 7B](https://img.shields.io/badge/model-Qwen%202.5%207B%20Instruct-orange)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)

> **Run a full AI agent on any laptop — no GPU, no cloud API key, no cost.**

Hermes-Lite is a local-first agent framework that proves you don't need expensive hardware or cloud subscriptions to run an agentic AI. A 7B quantized model + 6 essential tools + intelligent routing = a fully functional agent on an 8 GB MacBook.

**Built for the Hermes Agent Accelerated Business Hackathon** (NVIDIA × Stripe × Nous Research).

---

## Why Hermes-Lite?

Most agent frameworks assume you have a GPU cluster or an OpenAI API key. That leaves out:

- 💻 Developers on old laptops (8 GB RAM, no discrete GPU)
- 🌍 Users in regions with expensive or unreliable internet
- 🔒 Privacy-conscious teams that need fully offline agents
- 💰 Students and indie hackers who can't justify $100+/month in API costs

**Hermes-Lite runs 100% local.** The 7B Qwen model (4.4 GB Q4_K_M) fits in RAM alongside your IDE. Six built-in tools let it read files, search code, run commands, remember facts, and fetch web pages — all without a single outbound API call. When you *do* need heavy lifting, the router escalates to your cloud endpoint of choice (NVIDIA NIM, OpenAI, anything OpenAI-compatible).

---

## Quick Start

```bash
# 1. Install llama.cpp (LLM server)
brew install llama.cpp

# 2. Download the model (Qwen 2.5 7B Instruct, Q4_K_M — 4.4 GB)
hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf \
    --local-dir ~/.hermes_lite/models/

# 3. Merge the split files
llama-gguf-split --merge \
    ~/.hermes_lite/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    ~/.hermes_lite/models/qwen2.5-7b-instruct-q4_k_m.gguf

# 4. Start the LLM server
llama-server \
    -m ~/.hermes_lite/models/qwen2.5-7b-instruct-q4_k_m.gguf \
    --port 8080 --temp 0.3 --repeat-penalty 1.1 \
    -ngl 99 -c 4096

# 5. Install hermes-lite
git clone https://github.com/ahmedhabibo/hermes-lite.git
cd hermes-lite
pip install -e ".[test]"

# 6. Run the CLI
python -m hermes_lite
```

That's it. Start chatting in under 15 minutes.

---

## How It Works

```
┌─────────────────────────────────────────────────┐
│                    Router                       │
│  LiteRouter: classify by complexity → local/cloud│
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│                  LLM Layer                       │
│  Qwen 2.5 7B (local)  │  NVIDIA NIM (cloud)     │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│               Tool Loop                          │
│  2-tier: LLM → tool → result → LLM → response   │
│  Max 4 iterations, repeated-error, malformed-JSON│
│  Tool calls parsed from text (no PEG grammar)   │
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
│  sandbox-exec   │  Hermes MCP   │  curl          │
└──────────────────────────────────────────────────┘
```

**Key insight:** Local models can't reliably parse structured `tools` JSON from the OpenAI API spec. Hermes-Lite solves this by *not sending* `tools`/`tool_choice` to the local endpoint — instead, the model outputs tool calls as natural text, and a 4-pattern regex parser extracts them. This works even on models that never saw tool-calling in training.

---

## Demo

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
| **LLM layer** | OpenAI-compatible chat API. Default: local Qwen 2.5 7B Instruct GGUF via llama.cpp. Supports remote fallback (NVIDIA NIM). |
| **Tool loop** | Two-tier loop: LLM calls tools → results fed back → LLM responds. Max 4 iterations, repeated-error and malformed-JSON guards. Tool calls parsed from free-form text (4 regex patterns). |
| **Router** | LiteRouter classifies prompts by complexity. Simple queries → `_local_` tier (7B model). Complex reasoning → `_cloud_` tier. Consecutive-failure escalation with linear backoff. |
| **Sandbox** | `terminal` tool runs commands in a macOS sandbox (`sandbox-exec`). Timeout-safe with process lifecycle management. |
| **Memory** | SQLite bridge for cross-session facts. `add`, `replace`, `remove`, `list` — unique-match semantics. Loaded into every prompt (800 char cap). |
| **Sub-agent** | Spawn parallel `delegate_task` sub-agents. Isolated context + toolset per child. Nested orchestration (max 2 levels). |
| **Observability** | Per-turn JSONL logging, rotation at 10 MB, `python -m hermes_lite stats` for session summary. |
| **CLI** | prompt_toolkit + Rich terminal. Ctrl+C/D, `!tool {args}` direct invocation, `/tools`, `/history`, `/help`. |

---

## Configuration

Set these environment variables (or edit `hermes_lite/llm.py` / `router.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `HERMES_LITE_LOCAL_URL` | `http://127.0.0.1:8080/v1` | Local llama.cpp server endpoint |
| `HERMES_LITE_LOCAL_MODEL` | `qwen2.5-7b-instruct-q4_k_m.gguf` | Model file for local tier |
| `HERMES_LITE_CLOUD_URL` | `https://integrate.api.nvidia.com/v1` | Cloud endpoint (NVIDIA NIM) |
| `HERMES_LITE_CLOUD_MODEL` | `minimaxai/minimax-m3` | Model for cloud tier |
| `HERMES_LITE_NVIDIA_API_KEY` | — | NVIDIA NIM API key (required for cloud) |
| `HERMES_LITE_LOCAL_TOOLS` | unset | Set `1` to send `tools`/`tool_choice` to local endpoint |
| `LITE_LOCAL_MAX_COMPLEXITY` | `0.6` | Max complexity score for local routing |

---

## Test Suite

312 tests, all passing. Run with:

```bash
cd hermes-lite
pip install -e ".[test]"
python -m pytest tests/ -v
```

Tests cover: registry (48), memory (47), orchestrator (31), tool loop (15), tools-essentials (55), LLM (5), router (37), sandbox (30), sub-agent (18), memory bridge (10), observability (6), e2e smoke (5).

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) — PRs welcome, TDD preferred.

---

## License

[MIT](./LICENSE) — use it, fork it, ship it.

---

## CHANGELOG

### 0.3.0 — Qwen 2.5 7B Instruct + text-based tool-call parser
- Upgraded local model: Qwen 2.5 3B → Qwen 2.5 7B Instruct Q4_K_M (4.4 GB)
- Text-based tool-call parser: 4 regex patterns (Qwen blank-line JSON, fenced `tool_call`, fenced JSON, bare JSON)
- Skip `tools`/`tool_choice` for local endpoint — avoids PEG grammar 500 errors on small models
- Cloud fallback via NVIDIA NIM (configurable model chain)
- MIT license + CONTRIBUTING guide added

### 0.2.0 — Local Qwen 3B + 6 essential tools + router + sandbox
- Tool registry with 6 essentials (read_file, search_files, terminal, memory, web_search, web_fetch)
- LiteRouter: prompt complexity classifier for local/cloud tier routing
- ToolLoop: two-tier tool-calling loop with termination guards
- Sandboxed terminal execution (macOS sandbox-exec)
- Memory Bridge: cross-session persistent facts (SQLite)
- Subagent: parallel tool-spawning with isolated context
- Observability: per-turn JSONL logging + stats CLI
- 304 tests, all passing on macOS + 8 GB
