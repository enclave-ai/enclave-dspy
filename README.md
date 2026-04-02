# enclave-dspy

DSPy-based prompt optimization pipeline for [Enclave](https://github.com/enclave-ai/enclave) security agents.

## What This Does

Takes Enclave's 12 AI agent definitions (Markdown + YAML) and systematically optimizes their prompts using [DSPy](https://dspy.ai/) — Stanford NLP's framework for programming language models. Optimized prompts are exported back to Enclave's format for PR review.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env  # Add your API keys

# Ingest agent definitions from enclave repo
enclave-dspy ingest --enclave-path ../enclave

# Generate training data
enclave-dspy data generate --agent security-analyst --count 20

# Evaluate baseline
enclave-dspy evaluate --agent security-analyst

# Optimize
enclave-dspy optimize --agent security-analyst

# Export improved prompts
enclave-dspy export --agent security-analyst
```

See [docs/usage.md](docs/usage.md) for full documentation.

## Architecture

See [docs/superpowers/specs/2026-04-02-enclave-dspy-design.md](docs/superpowers/specs/2026-04-02-enclave-dspy-design.md) for the full design spec.

## Development

```bash
pip install -e ".[dev]"
pytest                    # Run tests
ruff check src/ tests/    # Lint
```
