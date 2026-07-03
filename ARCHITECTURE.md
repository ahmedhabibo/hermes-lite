# Hermes-Lite Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│  prompt_toolkit + Rich │ /tools /history /help /moa │ !tool {args} │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     HermesOrchestrator                           │
│  Wires: registry + memory + router + tool_loop + moa + CLI       │
└────┬───────────────────────────────────────────────────────────┘
     │
     ├───► LiteRouter ─────► tier: local / cloud
     │                        complexity threshold: 0.3
     │                        escalation on consecutive failures
     │
     ├───► MoAEngine ──────► parallel refs → aggregator (when active)
     │                        5 presets: council, speed, verification, coding, creative
     │
     ├───► ToolLoop ───────► LLM round-trips (max 4)
     │                        tool validation + dispatch
     │                        termination guards
     │
     ├───► PluginRegistry ─► ToolDefinition + Pydantic schema
     │                        built-in tools (6 essentials)
     │
     ├───► AsyncSQLitePool ─► sessions / messages / metadata
     │                        cross-session memory bridge
     │
     └───► Observability ──► JSONL logging
                              per-turn metrics
                              stats CLI
```

## Component Details

### 1. Router (`hermes_lite/router.py`)
- **Class**: `LiteRouter`
- **Responsibility**: Classify prompt complexity before LLM call
- **Logic**:
  - Simple heuristics: keyword detection, length, entity count
  - Threshold: `local_max_complexity` (default 0.3)
  - Consecutive local failures → escalate threshold (linear backoff)
  - **Cloud-first fallback chain**: `minimaxai/minimax-m3 → moonshotai/kimi-k2.6 → qwen/qwen3.5-397b-a17b → deepseek-ai/deepseek-v4-flash`
- **Output**: `RoutingDecision { tier, model_id, reasoning }`

### 2. LLM Layer (`hermes_lite/llm.py`)
- **Function**: `chat(req: ChatRequest) -> ChatResponse`
- **Protocol**: OpenAI-compatible `/v1/chat/completions`
- **Cloud**: NVIDIA NIM Free API (`https://integrate.api.nvidia.com/v1`)
- **Local**: llama.cpp server (`http://localhost:8080/v1`) — `local:` prefix
- **Rate limiting**: Token bucket (40 RPM default, `HERMES_LITE_RPM`)
- **Key rotation**: Round-robin from `HERMES_LITE_NVIDIA_API_KEYS` (comma-separated), 60s cooldown
- **Exponential backoff**: Retries on 429/500/502/503 with delays 1s/2s/4s/8s (max 16s)
- **Tool calling**: `tools`, `tool_choice`, `tool_calls` fields

### 3. MoA Engine (`hermes_lite/moa.py`)
- **Class**: `MoAEngine`
- **Pipeline**:
  1. **Reference models** — N diverse LLMs generate independent responses (parallel via `asyncio.gather` with semaphore limit of 2)
  2. **Aggregator model** — Stronger synthesis model receives all reference outputs + original prompt, produces final response
  3. **Fallback** — If reference fails, continue with remaining. If all refs fail → direct single-model call. If aggregator fails → best reference response (longest)
- **Presets** (5 built-in, all verified NIM free-tier models):
  | Preset | Refs | Aggregator | Use case |
  |--------|------|------------|----------|
  | `council` | minimax-m3, kimi-k2.6, qwen3.5-397b | kimi-k2.6 | General reasoning |
  | `speed` | minimax-m3, deepseek-v4-flash, qwen3.5-122b | minimax-m3 | Fast answers |
  | `verification` | kimi-k2.6, qwen3.5-397b, deepseek-v4-pro | minimax-m3 | Fact-checking |
  | `coding` | deepseek-v4-pro, qwen3.5-397b | deepseek-v4-pro | Code generation |
  | `creative` | minimax-m3, kimi-k2.6, qwen3.5-122b | minimax-m3 | Creative writing |
- **Env config**: `HERMES_LITE_MOA_PRESET`, `HERMES_LITE_MOA_TIMEOUT_S` (default 60s), `HERMES_LITE_MOA_REF_TEMPERATURE`, `HERMES_LITE_MOA_AGG_TEMPERATURE`, `HERMES_LITE_MOA_MAX_TOKENS`

### 4. Tool Loop (`hermes_lite/orchestrator.py`)
- **Class**: `ToolLoop`
- **Flow**:
  1. LLM receives messages + tool defs
  2. If `tool_calls` → validate JSON → dispatch via registry
  3. Append tool results to history
  4. Re-invoke LLM
  5. Repeat until: LLM responds plain text OR termination fires
- **Termination**:
  - `complete`: LLM returned no tool calls
  - `max_iterations`: exceeded 4 rounds
  - `repeated_error`: same tool+error twice
  - `malformed_tool_call`: bad JSON twice

### 5. Tool Registry (`hermes_lite/registry.py`)
- **Class**: `PluginRegistry(strict_validation=True)`
- **Operations**: `add_tool`, `call_tool`, `list_tools`, `tool_descriptions`
- **Validation**: Pydantic schema at dispatch time
- **Built-ins** (`hermes_lite/tools_builtins.py`):
  - `read_file`: read text file with pagination
  - `search_files`: regex/grep + file glob
  - `terminal`: sandboxed shell command
  - `memory`: persistent cross-session facts
  - `web_search`: Hermes MCP web search
  - `web_fetch`: URL content extraction

### 6. Memory (`hermes_lite/memory.py`, `hermes_lite/memory_bridge.py`)
- **Schema**: `sessions`, `messages`, `metadata` tables
- **Bridge**: `MemoryBridge` singleton with `add/replace/remove/list/load_into_prompt`
- **Uniqueness**: strict single-match semantics (error on duplicates)
- **Loading**: injected at orchestrator startup (800 char cap)

### 7. Sandbox (`hermes_lite/sandbox.py`)
- **Backend**: macOS `sandbox-exec` with custom profile
- **Allowed**: file read/write within workspace, network (configurable)
- **Timeout**: enforced via subprocess + thread monitor
- **Exit handling**: process cleanup on timeout/kill

### 8. Subagent (`hermes_lite/subagent.py`)
- **Tool**: `delegate_task` via Hermes MCP
- **Parameters**: `goal`, `context`, `toolsets`, `role`
- **Limits**: max 4 concurrent, max 2 nested levels
- **Isolation**: per-child session + terminal

### 9. Observability (`hermes_lite/observability.py`)
- **Log**: JSONL to `~/.hermes_lite/logs/turns.jsonl`
- **Rotation**: 10 MB cap, oldest first
- **Metrics**: `elapsed_ms`, `prompt_tokens`, `completion_tokens`, `tool_names`, `tier`, `errors`
- **CLI**: `python -m hermes_lite stats` prints session summary

## Data Flow

```
User prompt
    │
    ▼
┌─────────────────┐
│  LiteRouter     │ ───► tier = local/cloud
└────────┬────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  MoA Active?                        │
    │  (HERMES_LITE_MOA_PRESET or /moa)   │
    └──────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    ┌─────────┐         ┌─────────┐
    │ MoA     │         │ ToolLoop│
    │ Engine  │         │         │
    └────┬────┘         └────┬────┘
         │                   │
         ▼                   ▼
    ┌─────────┐         ┌─────────┐
    │ Refs    │         │ LLM     │
    │ (parallel)        │ + tools │
    └────┬────┘         └────┬────┘
         │                   │
         ▼                   ▼
    ┌─────────┐         ┌─────────┐
    │Aggregator│        │ Loop    │
    │         │         │ (max 4) │
    └────┬────┘         └────┬────┘
         │                   │
         └─────────┬─────────┘
                   ▼
         ┌─────────────────┐
         │  Save + Display │
         │  Response       │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Router Feedback│
         │  (escalation)   │
         └─────────────────┘
```

## Configuration

Environment variables (or edit defaults in `router.py`, `llm.py`, `moa.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HERMES_LITE_NVIDIA_API_KEY` | — | Single NIM API key |
| `HERMES_LITE_NVIDIA_API_KEYS` | — | Comma-separated key pool for rotation |
| `HERMES_LITE_RPM` | `40` | Requests per minute (token bucket) |
| `HERMES_LITE_MOA_PRESET` | — | Auto-activate MoA preset on startup |
| `HERMES_LITE_MOA_TIMEOUT_S` | `60` | Per-reference timeout (seconds) |
| `HERMES_LITE_MOA_REF_TEMPERATURE` | `0.4` | Default reference model temperature |
| `HERMES_LITE_MOA_AGG_TEMPERATURE` | `0.2` | Default aggregator temperature |
| `HERMES_LITE_MOA_MAX_TOKENS` | `4096` | Per-call token budget |
| `LITE_LOCAL_MAX_COMPLEXITY` | `0.3` | Max complexity for local routing |
| `LITE_LOCAL_MODEL` | `local:Qwen2.5-Coder-7B-Instruct-IQ3_XS` | Local model ID |
| `LITE_LLM_BASE_URL` | `http://localhost:8080/v1` | Local LLM server endpoint |
| `LITE_MEMORY_MAX_CHARS` | `800` | Max memory chars in prompt |
| `HERMES_MCP_CONFIG` | `~/.hermes/config.yaml` | MCP server config path |

## File Structure

```
hermes_lite/
├── __init__.py          # Public exports (incl. MoA)
├── __main__.py          # Entry point: python -m hermes_lite
├── cli.py               # prompt_toolkit + Rich CLI loop
├── llm.py               # ChatRequest/Response, chat(), rate limiting, key rotation
├── memory.py            # AsyncSQLitePool, CRUD ops
├── memory_bridge.py     # MemoryBridge singleton
├── moa.py               # MoAEngine, MoAPreset, MoAModelConfig, BUILTIN_PRESETS
├── observability.py     # JSONL logging, stats
├── orchestrator.py      # HermesOrchestrator, ToolLoop, MoA integration
├── prompts.py           # System prompts, templates
├── registry.py          # PluginRegistry, ToolDefinition
├── router.py            # LiteRouter, RoutingDecision, cloud-first fallback chain
├── sandbox.py           # sandbox-exec wrapper
├── subagent.py          # delegate_task tool
├── tools_builtins.py    # 6 essential tools
└── tests/
    ├── test_*.py        # 463 tests
```

## MoA Deep Dive

### Parallel Execution with Semaphore
```python
# In MoAEngine.__init__
self._ref_semaphore = asyncio.Semaphore(2)  # Limit concurrent calls

# In _call_reference
async with self._ref_semaphore:
    resp = await asyncio.wait_for(chat_fn(...), timeout=self.timeout_s)
```
This prevents 3-5 parallel reference calls from exhausting the 40 RPM token bucket.

### Aggregator Fallback Strategy
1. **All refs fail** → direct single-model call with aggregator model
2. **Aggregator fails** → return longest reference response (best effort)
3. **Partial ref failures** → aggregator synthesizes from successful ones

### Preset Selection Guide
- **`council`** — General questions, balanced depth/speed
- **`speed`** — Quick answers, low latency priority
- **`verification`** — Fact-checking, high accuracy priority
- **`coding`** — Code generation, debugging, refactoring
- **`creative`** — Writing, brainstorming, open-ended tasks