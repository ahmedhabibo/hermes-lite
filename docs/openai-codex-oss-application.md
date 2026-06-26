# OpenAI Codex for Open Source — Application Draft

## Project: Hermes-Lite
**Repository:** https://github.com/ahmedhabibo/hermes-lite
**License:** MIT
**Language:** Python (3.9+)
**Maintainer:** Ahmed Hassan (@ahmedhabibo)

---

## Brief Description

Hermes-Lite is a local-first AI agent framework that runs a full agentic system (tool calls, routing, memory, observability) on any laptop — no GPU, no cloud API key, no cost. The 7B quantized model (Qwen 2.5 7B Instruct Q4_K_M, 4.4 GB) runs via llama.cpp on an 8 GB MacBook, making AI agents accessible to developers on older hardware, in regions with unreliable internet, or with privacy requirements that demand fully offline operation.

## Why This Project Matters

The AI agent ecosystem has a hardware and cost gate. Most frameworks assume either a GPU cluster or a paid cloud API subscription ($50–200+/month). This excludes:

- Developers on 8 GB laptops without discrete GPUs
- Users in emerging markets (Egypt, Nigeria, Pakistan, etc.) where internet is expensive or unreliable
- Privacy-conscious organizations (healthcare, legal, government) that need fully offline agents
- Students and indie hackers who can't justify ongoing API costs

Hermes-Lite proves that a fully agentic system — with tool calling, routing, memory, and observability — can run locally on commodity hardware. The 4-pattern text-based tool-call parser is a novel approach that makes local models viable for agentic workflows even without structured JSON tool schemas.

## How We Would Use API Credits

1. **CI/CD automation** — Use Codex to automate pull request reviews, run test suites, and validate tool-call parser changes across edge cases on every PR
2. **Tool-call parser improvement** — Use Codex to generate synthetic LLM outputs across formats and models, then validate the regex parser against them
3. **Documentation generation** — Auto-generate API docs, changelogs, and migration guides from code changes
4. **Cross-platform testing** — Codex can simulate different Python versions and OS environments to catch platform-specific regressions

## Project Metrics

- **312 tests** — comprehensive coverage across 12 modules
- **5,300+ LOC** — substantial, production-quality Python codebase
- **Zero dependencies on cloud** — the entire agent runs offline with a 4.4 GB model
- **Active development** — 14 commits in the last month, clear progression from v0.1 → v0.3
- **MIT license** — permissive, OSS-friendly

## Current Maintainer Activity

- Primary maintainer with write access
- Active daily development
- Full test suite passing on every commit
- Clear roadmap (more tools, streaming, Docker, benchmarks)

---

*Submitted for the OpenAI Codex for Open Source program. Hermes-Lite makes AI agents accessible to everyone — including those who can't afford cloud API costs.*
