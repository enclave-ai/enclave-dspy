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

## Braintrust ETL

Pull real production traces from S3 and extract training data for all agents automatically:

```bash
# Pull and extract all agents
enclave-dspy data pull

# Pull a specific agent
enclave-dspy data pull --agent security-analyst

# Check what you have
enclave-dspy data status
```

Handles Enclave's multi-agent hierarchy — architecture-scanner delegates to security-analyst, finding-verifier, memory-extractor, etc. The ETL correctly attributes LLM calls to the right agent by tracing the span parent chain.

### Current Datasets (from production logs)

| Agent | Train | Dev | Test | Notes |
|---|---|---|---|---|
| finding-verifier | 433 | 92 | 92 | Largest dataset, validates/rejects findings |
| security-analyst | 122 | 25 | 25 | Core vulnerability scanner |
| chat-assistant | 112 | 24 | 24 | User-facing chat, already optimized |
| memory-extractor | 109 | 23 | 23 | Extracts reusable knowledge |
| finding-dedup-checker | 27 | 5 | 5 | Deduplicates vulnerability reports |
| attack-surface-mapper | 14 | 2 | 2 | Maps attack surfaces, delegates to analysts |
| architecture-scanner | 13 | 2 | 2 | Top-level orchestrator |
| diff-analyzer | 8 | 2 | 2 | PR diff security analysis |

## Optimization Results

| Agent | Baseline | Optimized | Improvement | Cost | Optimizer |
|---|---|---|---|---|---|
| chat-assistant | 69.3% | 75.7% | +9.2% | $1.28 | BootstrapFewShot |

## Architecture

See [docs/superpowers/specs/2026-04-02-enclave-dspy-design.md](docs/superpowers/specs/2026-04-02-enclave-dspy-design.md) for the full design spec.

## Development

```bash
pip install -e ".[dev]"
pytest                    # Run tests
ruff check src/ tests/    # Lint
```
