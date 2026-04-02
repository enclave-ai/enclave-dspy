# Enclave-DSPy Design Spec

> DSPy-based prompt optimization pipeline for Enclave security agents

## Overview

Enclave-DSPy is an offline optimization tool that uses Stanford's DSPy framework to systematically improve the prompts powering Enclave's 12 AI security agents and 12 audit skills. It mirrors Enclave's agent architecture 1:1 in DSPy, runs evaluation-driven optimization, and exports improved prompts back as Enclave-compatible Markdown files for PR review.

### Goals

- **Infrastructure first**: a well-structured, end-to-end optimization pipeline that works for any Enclave agent
- **Model-agnostic**: optimized prompts should work across providers (Anthropic, OpenAI, etc.)
- **Hybrid data**: combine production traces, synthetic generation, and manual curation
- **Offline tool**: runs separately from Enclave, outputs Markdown files to PR back

### Non-Goals

- Runtime prompt serving (future consideration)
- CI/CD integration (future consideration)
- Replacing Enclave's Markdown agent format

---

## 1. Architecture

```
enclave-dspy/
├── src/
│   ├── ingestion/          # Parse enclave Markdown agent definitions → DSPy signatures
│   ├── signatures/         # Generated + hand-tuned DSPy signatures per agent
│   ├── modules/            # DSPy modules mirroring each enclave agent
│   ├── metrics/            # Evaluation functions (exact match, LLM-as-judge, domain-specific)
│   ├── data/               # Data loaders for training/dev/test sets
│   ├── optimizers/         # Optimization runner configs and scripts
│   ├── export/             # Convert optimized DSPy programs back to enclave Markdown format
│   └── config.py           # LM configuration, DSPy settings
├── datasets/               # Curated evaluation datasets per agent type
│   ├── security-analyst/
│   ├── attack-surface-mapper/
│   ├── chat-assistant/
│   ├── pr-security-reviewer/
│   └── ...                 # One dir per enclave agent
├── optimized/              # Saved optimized programs (versioned JSON)
├── exports/                # Generated Markdown files ready to PR into enclave
├── scripts/                # CLI entry points (ingest, optimize, evaluate, export)
├── tests/
├── pyproject.toml
└── README.md
```

### Data Flow

```
Enclave Markdown agents ──→ Ingestion ──→ DSPy Signatures/Modules
                                              │
Datasets (hybrid sources) ──→ Data Loaders ──→│
                                              ▼
                                     DSPy Optimizer
                                         │
                                         ▼
                                  Optimized Programs (JSON)
                                         │
                                         ▼
                                  Export → Enclave Markdown
```

---

## 2. Ingestion Pipeline

The ingestion layer parses Enclave's Markdown+YAML agent definitions and converts them to DSPy representations.

### Parsed Fields

From each agent file's YAML frontmatter:
- `id`, `description`, `model`, `max_output_tokens`, `max_iterations`
- `tools` list (maps to DSPy tool definitions for ReAct modules)
- System prompt body (Markdown content after frontmatter)

### Output

- An `AgentSpec` dataclass holding all parsed metadata
- An auto-generated DSPy `Signature` class with input/output fields inferred from the agent's purpose
- A corresponding DSPy `Module` (ChainOfThought for simple agents, ReAct for tool-using agents)

### Design Decisions

- **Auto-generation is a starting point.** The ingestion creates a baseline signature, but each agent gets a hand-tuned signature in `src/signatures/` that refines the auto-generated one. The auto-gen runs once; after that, the hand-tuned version is the source of truth.
- **Tool mapping is declarative.** Enclave tools (read_file, ripgrep, bash, etc.) get mapped to Python function stubs that DSPy can reason about during optimization. For evaluation, tool outputs come from recorded traces.
- **Skills are sub-signatures.** Each enclave skill (ssrf-audit, sql-injection-audit, etc.) becomes its own DSPy signature that can be independently optimized and composed into agent modules.

---

## 3. Evaluation Data Pipeline

Three data sources feed into a unified `dspy.Example` format.

### Source 1: Production Traces (Braintrust)

- Pull logged agent interactions: input context, tool calls, final outputs
- Filter by quality signals (user accepted finding, finding verified, etc.)
- Convert to `dspy.Example` with `with_inputs()` marking the input fields

### Source 2: Synthetic Generation

- Use Opus as a teacher model to generate gold-standard examples per agent type
- For security agents: generate (code snippet, expected findings) pairs
- For chat agents: generate (question, context, expected response) pairs
- Each synthetic example gets a confidence score; low-confidence ones get flagged for manual review

### Source 3: Manual Curation

- JSONL files in `datasets/{agent-name}/` with `train.jsonl`, `dev.jsonl`, `test.jsonl`
- Schema: `{"inputs": {...}, "outputs": {...}, "metadata": {...}}`
- A CLI command (`scripts/data.py curate`) opens an interactive review flow for validating synthetic examples

### Pipeline Flow

```
Braintrust traces ──┐
                    ├──→ Normalize ──→ Deduplicate ──→ Split (train/dev/test)
Synthetic gen ──────┤                                      │
                    │                                      ▼
Manual JSONL ───────┘                              datasets/{agent}/
                                                   ├── train.jsonl
                                                   ├── dev.jsonl
                                                   └── test.jsonl
```

### Minimum Viable Dataset

20 examples per agent to start (10 train, 5 dev, 5 test). Enough for BootstrapFewShot. Scale to 200+ for MIPROv2/GEPA.

---

## 4. Metrics System

Three-layer metrics architecture.

### Layer 1: Generic Metrics (all agents)

- `semantic_similarity` — embedding-based similarity between predicted and expected outputs
- `structural_completeness` — does the output contain all required fields/sections
- `instruction_adherence` — LLM-as-judge checking if the output follows the agent's constraints

### Layer 2: Agent-Type Metrics

| Agent Category | Key Metrics |
|---|---|
| Security analysts | Finding precision, finding recall, severity accuracy, CWE classification correctness |
| Attack surface mapper | Coverage (all attack surfaces identified), delegation quality |
| Verification agents | False positive rate, false negative rate, reasoning quality |
| Dedup agents | Dedup accuracy, false merge rate |
| Chat assistant | Response relevance, factual correctness, actionability |
| PR reviewer | Finding relevance to diff, false alarm rate, severity calibration |

### Layer 3: LLM-as-Judge

- A DSPy `ChainOfThought` module scoring outputs on multiple dimensions
- Returns both a scalar score (for optimization) and textual feedback (for GEPA/SIMBA)
- Uses a separate judge model to avoid self-evaluation bias

### Metric Composition

```python
def agent_metric(example, pred, trace=None):
    scores = {
        "semantic": semantic_similarity(example, pred),
        "structural": structural_completeness(example, pred),
        "domain": domain_specific_metric(example, pred),
        "judge": llm_judge_score(example, pred),
    }
    weights = get_weights_for_agent(example.agent_type)
    return sum(scores[k] * weights[k] for k in scores)
```

Weights are configurable per agent in `optimization_config.yaml`.

---

## 5. Optimization Pipeline

### Three-Stage Strategy (per agent)

**Stage 1: Baseline measurement**
- Run the unoptimized DSPy module against dev set
- Record scores per metric dimension
- This is the benchmark all optimization compares against

**Stage 2: Progressive optimization**

| Step | Optimizer | Condition | Min Data |
|---|---|---|---|
| 1 | `BootstrapFewShot` | Always first — fast, cheap | 10 examples |
| 2 | `MIPROv2(auto="light")` | Bootstrap improves <10% | 50 examples |
| 3 | `MIPROv2(auto="medium")` | Light improves <5% more | 100 examples |
| 4 | `GEPA` | For agents with rich textual feedback metrics | 50 examples |

Each step only runs if the previous didn't hit the threshold. Thresholds are configurable per agent in `optimization_config.yaml`.

**Stage 3: Validation and selection**
- Run all optimized candidates against held-out test set
- Compare against baseline
- Best performer saved to `optimized/{agent-id}/v{N}.json`
- Results logged to `optimized/{agent-id}/report.md`

### Configuration

```yaml
# optimization_config.yaml
defaults:
  max_bootstrapped_demos: 4
  max_labeled_demos: 8
  improvement_threshold: 0.05
  teacher_model: "anthropic/claude-opus-4-6"
  target_model: "anthropic/claude-sonnet-4-6"

agents:
  security-analyst:
    optimizers: [bootstrap, mipro_light, gepa]
    metric_weights:
      finding_precision: 0.3
      finding_recall: 0.3
      severity_accuracy: 0.2
      judge: 0.2
  chat-assistant:
    optimizers: [bootstrap, mipro_medium]
    metric_weights:
      relevance: 0.4
      correctness: 0.3
      actionability: 0.3
```

---

## 6. Export Pipeline

Converts optimized DSPy programs back into Enclave's Markdown+YAML format.

### Exported Content

- Optimized system prompt instructions (from DSPy's learned instructions)
- Curated few-shot examples (from bootstrapped demonstrations) as example sections
- Original frontmatter metadata preserved (id, model, tools unchanged)

### Process

1. Load optimized program JSON from `optimized/{agent-id}/v{N}.json`
2. Extract learned instructions and demonstrations
3. Merge with original agent frontmatter
4. Write to `exports/{agent-id}.md` in Enclave's exact format
5. Generate diff summary showing what changed

### Guardrails

- Exports include a `# DSPy Optimized` comment header with version, date, and score comparison
- If optimization score is worse than baseline, export is blocked with a warning
- Original agent definitions are never modified

---

## 7. Testing Strategy

### Unit Tests

- **Ingestion**: parse every Enclave agent definition without error, verify field extraction
- **Signatures**: validate all DSPy signatures compile with correct input/output fields
- **Metrics**: test each metric with known good/bad pairs, assert expected score ranges
- **Export**: round-trip test — ingest agent, export unoptimized, verify format matches original

### Integration Tests

- End-to-end: ingest → load data → optimize (BootstrapFewShot on tiny dataset) → evaluate → export
- Uses mocked/cached LLM responses for speed and determinism
- Verify exported Markdown is valid for Enclave's agent loader

### Evaluation Tests (on demand)

- Actually call LLMs against dev sets
- Compare optimized vs baseline scores
- Run via `python -m scripts.evaluate --agent {name}`

### Test Structure

```
tests/
├── unit/
│   ├── test_ingestion.py
│   ├── test_signatures.py
│   ├── test_metrics.py
│   └── test_export.py
├── integration/
│   └── test_pipeline.py
└── conftest.py
```

CI runs unit + integration on every push. Evaluation tests are manual/nightly.

---

## Tech Stack

- **Python 3.12+**
- **DSPy 2.x** — core optimization framework
- **LiteLLM** (via DSPy) — multi-provider LLM access
- **Pydantic** — data validation for agent specs and datasets
- **PyYAML + python-frontmatter** — parsing Enclave Markdown definitions
- **pytest** — testing
- **Ruff** — linting and formatting
- **Click** — CLI framework for scripts

---

## Enclave Agents Covered

| Agent ID | Type | DSPy Module |
|---|---|---|
| security-analyst | Tool-using analyzer | ReAct |
| attack-surface-mapper | Delegating coordinator | ReAct |
| chat-assistant | Interactive responder | ChainOfThought |
| pr-security-reviewer | Diff analyzer | ChainOfThought |
| finding-verifier | Binary classifier | Predict |
| finding-dedup-checker | Similarity matcher | ChainOfThought |
| diff-analyzer | Code diff interpreter | ChainOfThought |
| conversation-compactor | Summarizer | Predict |
| memory-extractor | Knowledge extractor | ChainOfThought |
| suggestion-summarizer | Summarizer | Predict |
| architecture-scanner | Code mapper | ReAct |
| dedup-checker | Generic dedup | ChainOfThought |

Plus 12 security audit skills as independent sub-signatures.
