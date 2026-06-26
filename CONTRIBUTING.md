# Contributing to Hermes-Lite

Thanks for your interest! Hermes-Lite is designed to be approachable — even first-time contributors can make meaningful improvements.

## Quick Start

```bash
git clone https://github.com/ahmedhabibo/hermes-lite.git
cd hermes-lite
pip install -e ".[test]"
python -m pytest tests/ -v
```

If all 312 tests pass, you're ready to hack.

## What to Work On

- Check [open issues](https://github.com/ahmedhabibo/hermes-lite/issues) for bug reports and feature requests
- Look for `TODO` / `FIXME` comments in the codebase
- Improve docs, add examples, or fix edge cases in the tool-call parser

## Making Changes

1. **Fork** the repo and create a feature branch from `main`
2. **Write tests first** — we follow TDD. New features need new test cases
3. **Run the full suite** — `python -m pytest tests/ -v` must be green
4. **Keep it small** — one logical change per PR makes review faster
5. **Document** — update README and docstrings for any changed behavior

## Code Style

- Python 3.9+, type hints preferred
- `pydantic` models for all data shapes
- Docstrings on public functions (Google style)
- 88-char line length (Black default)

## Commit Messages

```
feat: add web_fetch tool
fix: handle blank responses from local LLM
docs: update Quick Start for 7B model
```

## Reporting Bugs

Open an issue with:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs (redact any API keys)

## Questions?

Open a [Discussion](https://github.com/ahmedhabibo/hermes-lite/discussions) — no question is too small.
