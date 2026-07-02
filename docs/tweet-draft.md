🚀 Hermes-Lite: a full AI agent that runs on any laptop — no GPU, no API key, no cost.

Built for the @NousResearch Accelerated Business Hackathon with @NVIDIAAI × @stripe.

What it does on an 8GB MacBook:
⚡ 6 tools (files, search, terminal, memory, web) — all local
🧠 Qwen 2.5 7B Instruct via llama.cpp — 100% offline
🔀 Two-tier routing: simple → local 7B, complex → cloud fallback
💡 Text-based tool-call parser — no PEG grammar, works on small models

The key insight: local quantized models can't reliably parse OpenAI-style tools JSON. So Hermes-Lite skips tools/tool_choice entirely and parses tool calls from free-form text with 4 regex patterns. This makes agentic AI viable on commodity hardware.

312 tests passing. Under 15 min setup.

Demo 👇 #HermesAgentChallenge #BuildWithHermes

🔗 github.com/ahmedhabibo/hermes-lite
