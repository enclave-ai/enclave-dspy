# Enclave-DSPy Usage Guide

## Prerequisites

- Python 3.12+
- Access to Enclave repo (for agent definitions)
- API keys for at least one LLM provider (Anthropic, OpenAI, etc.)

## Installation

```bash
cd enclave-dspy
pip install -e ".[dev]"
```

## Environment Setup

Create a `.env` file:

```bash
# Required: at least one LLM provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional: Braintrust for pulling production traces
BRAINTRUST_API_KEY=...

# Path to enclave repo (for ingesting agent definitions)
ENCLAVE_REPO_PATH=../enclave
```

## Commands

### 1. Ingest Enclave Agent Definitions

Parse Enclave's Markdown agent files into DSPy signatures and modules:

```bash
# Ingest all agents
python -m scripts.ingest --enclave-path ../enclave

# Ingest a specific agent
python -m scripts.ingest --enclave-path ../enclave --agent security-analyst
```

This reads from `packages/ai-client/src/definitions/` in the Enclave repo and generates baseline DSPy signatures in `src/signatures/`.

### 2. Prepare Evaluation Data

#### Pull production traces

```bash
python -m scripts.data pull --source braintrust --agent security-analyst
```

#### Generate synthetic examples

```bash
python -m scripts.data generate --agent security-analyst --count 50 --teacher-model anthropic/claude-opus-4-6
```

#### Curate and review examples

```bash
python -m scripts.data curate --agent security-analyst
```

This opens an interactive review flow for validating synthetic examples.

#### Check dataset status

```bash
python -m scripts.data status
```

Prints a table showing example counts per agent per split (train/dev/test).

### 3. Evaluate Baseline

Measure unoptimized performance before running optimization:

```bash
# Evaluate a single agent
python -m scripts.evaluate --agent security-analyst

# Evaluate all agents
python -m scripts.evaluate --all

# Evaluate a specific optimized version
python -m scripts.evaluate --agent security-analyst --version v3
```

Output includes per-metric scores and an overall weighted score.

### 4. Optimize

Run the progressive optimization pipeline:

```bash
# Optimize a single agent
python -m scripts.optimize --agent security-analyst

# Optimize all agents
python -m scripts.optimize --all

# Run only a specific optimizer
python -m scripts.optimize --agent security-analyst --optimizer bootstrap

# Override config for a run
python -m scripts.optimize --agent security-analyst --max-bootstrapped-demos 8 --max-labeled-demos 16
```

#### Optimizer options

| Flag | Description |
|---|---|
| `--optimizer` | Which optimizer to run: `bootstrap`, `mipro_light`, `mipro_medium`, `gepa` |
| `--max-bootstrapped-demos` | Max self-generated demonstrations (default: 4) |
| `--max-labeled-demos` | Max examples from training data (default: 8) |
| `--teacher-model` | Model used for generating demos (default: from config) |
| `--target-model` | Model being optimized for (default: from config) |
| `--threshold` | Improvement threshold to stop early (default: 0.05) |

#### What happens during optimization

1. Loads the DSPy module and datasets for the target agent
2. Measures baseline score on dev set
3. Runs optimizers progressively (BootstrapFewShot → MIPROv2 → GEPA)
4. Each optimizer only runs if the previous didn't meet the improvement threshold
5. Validates best candidate on held-out test set
6. Saves optimized program to `optimized/{agent-id}/v{N}.json`
7. Writes comparison report to `optimized/{agent-id}/report.md`

### 5. Compare Results

View optimization results:

```bash
# Compare baseline vs all optimized versions
python -m scripts.compare --agent security-analyst

# Compare two specific versions
python -m scripts.compare --agent security-analyst --versions v1 v3
```

Outputs a table with per-metric scores and overall improvement percentages.

### 6. Export to Enclave Format

Convert optimized programs back to Enclave Markdown:

```bash
# Export a specific agent and version
python -m scripts.export --agent security-analyst --version v3

# Export all agents (latest optimized version)
python -m scripts.export --all

# Preview changes without writing files
python -m scripts.export --agent security-analyst --dry-run
```

Exported files land in `exports/{agent-id}.md`. Copy them to Enclave's `packages/ai-client/src/definitions/` and open a PR.

### 7. Run Tests

```bash
# Unit + integration tests
pytest

# Only unit tests
pytest tests/unit/

# Only integration tests
pytest tests/integration/

# With coverage
pytest --cov=src
```

## Typical Workflow

```
1. Ingest    →  python -m scripts.ingest --enclave-path ../enclave
2. Data      →  python -m scripts.data generate --agent security-analyst --count 50
3. Baseline  →  python -m scripts.evaluate --agent security-analyst
4. Optimize  →  python -m scripts.optimize --agent security-analyst
5. Compare   →  python -m scripts.compare --agent security-analyst
6. Export    →  python -m scripts.export --agent security-analyst --version v1
7. PR        →  Copy exports/ to enclave repo, open PR
```

## Configuration Reference

All optimization settings live in `optimization_config.yaml` at the repo root.

### Global Defaults

```yaml
defaults:
  max_bootstrapped_demos: 4
  max_labeled_demos: 8
  improvement_threshold: 0.05
  teacher_model: "anthropic/claude-opus-4-6"
  target_model: "anthropic/claude-sonnet-4-6"
```

### Per-Agent Overrides

```yaml
agents:
  security-analyst:
    optimizers: [bootstrap, mipro_light, gepa]
    metric_weights:
      finding_precision: 0.3
      finding_recall: 0.3
      severity_accuracy: 0.2
      judge: 0.2
    max_bootstrapped_demos: 6  # override default
```

### Dataset Schema

Each example in `datasets/{agent}/train.jsonl`:

```json
{
  "inputs": {
    "code_context": "...",
    "question": "..."
  },
  "outputs": {
    "findings": [...],
    "severity": "high"
  },
  "metadata": {
    "source": "braintrust",
    "confidence": 0.95
  }
}
```

## Troubleshooting

### "No examples found for agent"

Run data generation first: `python -m scripts.data generate --agent {name} --count 20`

### "Optimization score worse than baseline"

This can happen with too few examples or inappropriate metrics. Try:
- Adding more training examples
- Adjusting metric weights in config
- Using a different optimizer

### "Export blocked: score regression"

The export pipeline refuses to export if the optimized score is below baseline. Check the report at `optimized/{agent-id}/report.md` to understand what happened.

### Rate limiting during optimization

Set `--target-model` to a cheaper model during experimentation. Use the full model for final optimization runs.
