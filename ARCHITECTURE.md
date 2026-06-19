# Hermes-Lite Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│  prompt_toolkit + Rich │ /tools /history /help │ !tool {args}   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     HermesOrchestrator                           │
│  Wires: registry + memory + router + tool_loop + CLI             │
└────┬───────────────────────────────────────────────────────────┘
     │
     ├───► LiteRouter ─────► tier: local / cloud
     │                        complexity threshold: 0.6
     │                        escalation on consecutive failures
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
  - Threshold: `local_max_complexity` (default 0.6)
  - Consecutive local failures → escalate threshold (linear backoff)
  - Fallback chain: `["local:qwen-3b", "openai:gpt-4o-mini"]`
- **Output**: `RoutingDecision { tier, model_id, reasoning }`

### 2. LLM Layer (`hermes_lite/llm.py`)
- **Function**: `chat(req: ChatRequest) -> ChatResponse`
- **Protocol**: OpenAI-compatible `/v1/chat/completions`
- **Local**: llama.cpp server (`http://localhost:8081/v1`)
- **Cloud**: OpenAI API (configurable)
- **Tool calling**: `tools`, `tool_choice`, `tool_calls` fields

### 3. Tool Loop (`hermes_lite/orchestrator.py`)
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

### 4. Tool Registry (`hermes_lite/registry.py`)
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

### 5. Memory (`hermes_lite/memory.py`, `hermes_lite/memory_bridge.py`)
- **Schema**: `sessions`, `messages`, `metadata` tables
- **Bridge**: `MemoryBridge` singleton with `add/replace/remove/list/load_into_prompt`
- **Uniqueness**: strict single-match semantics (error on duplicates)
- **Loading**: injected at orchestrator startup (800 char cap)

### 6. Sandbox (`hermes_lite/sandbox.py`)
- **Backend**: macOS `sandbox-exec` with custom profile
- **Allowed**: file read/write within workspace, network (configurable)
- **Timeout**: enforced via subprocess + thread monitor
- **Exit handling**: process cleanup on timeout/kill

### 7. Subagent (`hermes_lite/subagent.py`)
- **Tool**: `delegate_task` via Hermes MCP
- **Parameters**: `goal`, `context`, `toolsets`, `role`
- **Limits**: max 4 concurrent, max 2 nested levels
- **Isolation**: per-child session + terminal

### 8. Observability (`hermes_lite/observability.py`)
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
┌─────────────────┐
│  LLM (chat)     │ ───► tool_calls[] or response
└────────┬────────┘
         │
         ├───► No tool_calls ───► Save + display response ───► Done
         │
         ▼
    Has tool_calls
         │
         ▼
┌─────────────────┐
│  Registry       │ ───► validate schema ──X──► ToolValidationError
│  call_tool()    │                              (append error to history)
└────────┬────────┘
         │
         ✓
         │
         ▼
┌─────────────────┐
│  Tool Backend   │ ───► read_file / search_files / terminal / ...
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Append result  │
│  to history     │
└────────┬────────┘
         │
         ▼
    Loop back to LLM (max 4 iterations)
```

## Configuration

Environment variables (or edit defaults in `router.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LITE_LOCAL_MAX_COMPLEXITY` | `0.6` | Max complexity for local routing |
| `LITE_LOCAL_MODEL` | `local:qwen-3b` | Model ID for local tier |
| `LITE_CLOUD_MODEL` | `openai:gpt-4o-mini` | Model ID for cloud tier |
| `LITE_LLM_BASE_URL` | `http://localhost:8081/v1` | LLM server endpoint |
| `LITE_MEMORY_MAX_CHARS` | `800` | Max memory chars in prompt |
| `HERMES_MCP_CONFIG` | `~/.hermes/config.yaml` | MCP server config path |

## File Structure

```
hermes_lite/
├── __init__.py          # Public exports
├── __main__.py          # Entry point: python -m hermes_lite
├── cli.py               # prompt_toolkit + Rich CLI loop
├── llm.py               # ChatRequest/Response, chat()
├── memory.py            # AsyncSQLitePool, CRUD ops
├── memory_bridge.py     # MemoryBridge singleton
├── observability.py     # JSONL logging, stats
├── orchestrator.py      # HermesOrchestrator, ToolLoop
├── prompts.py           # System prompts, templates
├── registry.py          # PluginRegistry, ToolDefinition
├── router.py            # LiteRouter, RoutingDecision
├── sandbox.py           # sandbox-exec wrapper
├── subagent.py          # delegate_task tool
├── tools_builtins.py    # 6 essential tools
└── tests/
    ├── test_*.py        # 304 tests
```