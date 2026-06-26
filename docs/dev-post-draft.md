# Hermes-Lite: A Full AI Agent That Runs on Your Old Laptop — No GPU, No API Key, No Cost

> Submission for the **Build with Hermes Agent** prompt — Hermes Agent Challenge

---

## The Problem

Most agent frameworks assume you have a GPU cluster or an OpenAI API key. That leaves out:

- Developers on 8 GB laptops with no discrete GPU
- Users in regions with expensive or unreliable internet
- Privacy-conscious teams that need fully offline agents
- Students and indie hackers who can't justify $100+/month in API costs

**What if you could run a fully agentic AI — tool calls, routing, memory, the works — on any MacBook from the last 5 years, with zero cloud dependency?**

That's Hermes-Lite.

---

## What I Built

**Hermes-Lite** is a local-first agent framework that runs entirely on a 7B quantized model (Qwen 2.5 7B Instruct Q4_K_M, 4.4 GB) via llama.cpp. It includes:

| Feature | How it works |
|---------|-------------|
| **6 built-in tools** | `read_file`, `search_files`, `terminal`, `memory`, `web_search`, `web_fetch` — Pydantic-validated dispatch |
| **Tool-calling without grammar** | 4-pattern regex parser extracts tool calls from free-form LLM text — no structured `tools` JSON needed |
| **Two-tier routing** | LiteRouter classifies prompts: simple → local 7B, complex → cloud fallback (NVIDIA NIM) |
| **Persistent memory** | SQLite cross-session facts, loaded into every prompt (800 char cap) |
| **Sandboxed execution** | `terminal` tool runs in macOS `sandbox-exec` |
| **Observability** | Per-turn JSONL logging, rotation, stats CLI |

**312 tests passing.** Install → download model → start llama-server → chat. Under 15 minutes.

---

## The Key Technical Insight

Local quantized models can't reliably parse the OpenAI-style `tools` JSON schema — they hallucinate formats, break JSON syntax, or crash the PEG grammar constrained decoder in llama.cpp.

**My solution: don't send `tools`/`tool_choice` to the local endpoint.** Instead:

1. The model outputs tool calls as natural language text (fenced code blocks, JSON blocks, or Qwen's native blank-line format)
2. A 4-pattern regex parser extracts `{"name": "...", "arguments": {...}}` from the free-form output
3. If no tool call is detected, the text is returned as a regular response

This approach works even on models that never saw tool-calling in training — and it's what makes the "old laptop" use case viable.

---

## How It Uses Hermes Agent's Agentic Capabilities

Hermes-Lite is **built on design patterns inspired by Hermes Agent**:

- **Tool registry** — mirrors Hermes' `ToolDefinition` approach with Pydantic validation
- **Sub-agent spawning** — `delegate_task` with isolated context, matching Hermes' delegation model
- **Orchestration loop** — tool-loop pattern directly inspired by Hermes' two-tier loop
- **Observability** — JSONL per-turn logging, same philosophy as Hermes' session tracking

The project demonstrates that these agentic patterns work at the *small end* of the spectrum — you don't need a 70B model or cloud GPU to have a real agent.

---

## Demo

[Screen recording: 3 prompts showing local tool calls + cloud escalation]

```bash
# Start the demo (requires llama-server on port 8080)
python scripts/demo.py
```

---

## Try It Yourself

```bash
brew install llama.cpp
hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf \
    --local-dir ~/.hermes_lite/models/

llama-gguf-split --merge \
    ~/.hermes_lite/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    ~/.hermes_lite/models/qwen2.5-7b-instruct-q4_k_m.gguf

llama-server \
    -m ~/.hermes_lite/models/qwen2.5-7b-instruct-q4_k_m.gguf \
    --port 8080 --temp 0.3 --repeat-penalty 1.1 -ngl 99 -c 4096

git clone https://github.com/ahmedhabibo/hermes-lite.git
cd hermes-lite && pip install -e ".[test]"
python -m hermes_lite
```

**GitHub:** [ahmedhabibo/hermes-lite](https://github.com/ahmedhabibo/hermes-lite)

---

## What's Next

- More tools: `write_file`, `patch`, `web_scrape`
- Streaming responses in the CLI
- Docker image for one-command setup
- Benchmarks comparing 7B local vs cloud on common agent tasks

---

*Built for the Hermes Agent Challenge — proving that agentic AI doesn't require expensive hardware. If an ERP consultant in Cairo can run a full agent on his MacBook, anyone can.*
