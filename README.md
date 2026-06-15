# Hermes-Lite

Lightweight agent framework with tool registry, SQLite memory, and orchestrator loop.

## Architecture

```
hermes_lite/
├── registry.py      — PluginRegistry: tool registration, validation, dispatch
├── memory.py        — AsyncSQLitePool: session/message/metadata persistence
├── orchestrator.py  — HermesOrchestrator: wires registry + memory + CLI
├── cli.py           — prompt_toolkit + Rich interface loop
└── __main__.py      — Entry point: `python -m hermes_lite`
```

### Components

| Module | Role | Tests |
|---|---|---|
| `registry.py` | Tool registry with Pydantic schema validation at dispatch. Supports strict (rejects schema-free tools) and non-strict modes. | 48 |
| `memory.py` | Async SQLite connection pool with WAL journaling. CRUD for sessions, messages (auto-sequence), and JSON metadata. | 47 |
| `orchestrator.py` | Coordinates all three layers: registers built-in tools (echo, calculator), persists conversation history, handles `!tool {...}` direct invocation, and `/tools` / `/history` / `/help` commands. | 25 |
| `cli.py` | Rich terminal loop with styled panels, Ctrl+C/D exit, async handler support. | — |

**Total: 120 tests, all passing.**

## Quick Start

```bash
# Install
pip install -e .

# Run the CLI
python -m hermes_lite
```

Commands inside the CLI:
- Type any message for a general response (shows available tools)
- `!echo {"message": "hello"}` — invoke the echo tool directly
- `!calculator {"expression": "2 + 3"}` — evaluate arithmetic
- `/tools` — list registered tools with their schemas
- `/history` — view recent conversation history
- `/help` — show help
- `/exit`, `/quit`, `/q`, or Ctrl+C/D — exit

## Configuration

Database path defaults to `~/.hermes_lite/sessions.db`. Override when creating the orchestrator:

```python
from hermes_lite import HermesOrchestrator
orch = HermesOrchestrator(db_path="/path/to/sessions.db")
orch.start()
```

## Extending

Add custom tools by creating a `ToolDefinition` with a Pydantic schema:

```python
from pydantic import BaseModel, Field
from hermes_lite import PluginRegistry, ToolDefinition

class GreetArgs(BaseModel):
    name: str = Field(..., description="Name to greet")

def greet(args: GreetArgs) -> str:
    return f"Hello, {args.name}!"

tool = ToolDefinition(
    name="greet",
    description="Greet someone by name.",
    schema_model=GreetArgs,
    handler=greet,
)
orch.registry.add_tool(tool)
```

## Development

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```