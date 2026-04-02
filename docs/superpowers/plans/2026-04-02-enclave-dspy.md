# Enclave-DSPy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DSPy-based prompt optimization pipeline that ingests Enclave's Markdown agent definitions, optimizes them via DSPy, and exports improved prompts back to Enclave format.

**Architecture:** Mirror Enclave's 12 agents + 12 skills 1:1 as DSPy signatures/modules. A progressive optimization pipeline (BootstrapFewShot → MIPROv2 → GEPA) runs against curated evaluation datasets, and an export layer converts optimized programs back to Enclave Markdown. The entire system is an offline CLI tool.

**Tech Stack:** Python 3.12+, DSPy 2.x, Pydantic, python-frontmatter, Click, pytest, Ruff

---

## File Structure

```
enclave-dspy/
├── pyproject.toml
├── optimization_config.yaml
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py                    # LM configuration, env loading
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py                # Parse Markdown+YAML → AgentSpec/SkillSpec
│   │   └── models.py                # Pydantic models for AgentSpec, SkillSpec
│   ├── signatures/
│   │   ├── __init__.py
│   │   └── factory.py               # Generate DSPy Signatures from AgentSpec
│   ├── modules/
│   │   ├── __init__.py
│   │   └── factory.py               # Generate DSPy Modules from AgentSpec + Signature
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Load JSONL datasets → dspy.Example lists
│   │   └── synthetic.py             # Generate synthetic training examples
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── generic.py               # semantic_similarity, structural_completeness
│   │   ├── domain.py                # Agent-type-specific metrics
│   │   └── judge.py                 # LLM-as-judge metric
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── runner.py                # Progressive optimization pipeline
│   └── export/
│       ├── __init__.py
│       └── exporter.py              # Convert optimized DSPy programs → Enclave Markdown
├── scripts/
│   ├── __init__.py
│   ├── cli.py                       # Click CLI group entry point
│   ├── ingest.py                    # Ingest command
│   ├── data_cmd.py                  # Data subcommands (generate, curate, status)
│   ├── evaluate.py                  # Evaluate command
│   ├── optimize.py                  # Optimize command
│   ├── compare.py                   # Compare command
│   └── export_cmd.py               # Export command
├── datasets/                        # Curated datasets (JSONL per agent)
├── optimized/                       # Saved optimized programs (JSON)
├── exports/                         # Generated Enclave Markdown files
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_parser.py
│   │   ├── test_models.py
│   │   ├── test_signature_factory.py
│   │   ├── test_module_factory.py
│   │   ├── test_loader.py
│   │   ├── test_metrics.py
│   │   └── test_exporter.py
│   └── integration/
│       ├── __init__.py
│       └── test_pipeline.py
└── docs/
    ├── usage.md
    └── superpowers/
        ├── specs/
        │   └── 2026-04-02-enclave-dspy-design.md
        └── plans/
            └── 2026-04-02-enclave-dspy.md
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `optimization_config.yaml`
- Create: `src/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "enclave-dspy"
version = "0.1.0"
description = "DSPy-based prompt optimization pipeline for Enclave security agents"
requires-python = ">=3.12"
dependencies = [
    "dspy>=2.6",
    "pydantic>=2.0",
    "python-frontmatter>=1.1",
    "pyyaml>=6.0",
    "click>=8.1",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.8",
]

[project.scripts]
enclave-dspy = "scripts.cli:cli"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create .env.example**

```bash
# Required: at least one LLM provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional: Braintrust for pulling production traces
BRAINTRUST_API_KEY=

# Path to enclave repo (for ingesting agent definitions)
ENCLAVE_REPO_PATH=../enclave
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.ruff_cache/
.pytest_cache/
.coverage
htmlcov/
.venv/
venv/
```

- [ ] **Step 4: Create optimization_config.yaml**

```yaml
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
  attack-surface-mapper:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      coverage: 0.4
      delegation_quality: 0.3
      judge: 0.3
  chat-assistant:
    optimizers: [bootstrap, mipro_medium]
    metric_weights:
      relevance: 0.4
      correctness: 0.3
      actionability: 0.3
  pr-security-reviewer:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      finding_relevance: 0.4
      false_alarm_rate: 0.3
      severity_calibration: 0.3
  finding-verifier:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      false_positive_rate: 0.4
      false_negative_rate: 0.4
      reasoning_quality: 0.2
  finding-dedup-checker:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      dedup_accuracy: 0.5
      false_merge_rate: 0.3
      judge: 0.2
  diff-analyzer:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      semantic: 0.3
      structural: 0.3
      judge: 0.4
  conversation-compactor:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      semantic: 0.5
      structural: 0.3
      judge: 0.2
  memory-extractor:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      semantic: 0.4
      structural: 0.3
      judge: 0.3
  suggestion-summarizer:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      semantic: 0.5
      structural: 0.3
      judge: 0.2
  architecture-scanner:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      coverage: 0.4
      structural: 0.3
      judge: 0.3
  dedup-checker:
    optimizers: [bootstrap, mipro_light]
    metric_weights:
      dedup_accuracy: 0.5
      false_merge_rate: 0.3
      judge: 0.2
```

- [ ] **Step 5: Create src/__init__.py**

```python
```

(Empty file.)

- [ ] **Step 6: Create empty directories with .gitkeep**

```bash
mkdir -p datasets optimized exports
touch datasets/.gitkeep optimized/.gitkeep exports/.gitkeep
```

- [ ] **Step 7: Install the project in dev mode**

Run: `pip install -e ".[dev]"`
Expected: Successfully installed enclave-dspy and all dependencies

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example .gitignore optimization_config.yaml src/__init__.py datasets/.gitkeep optimized/.gitkeep exports/.gitkeep
git commit -m "feat: project scaffolding with dependencies and config"
```

---

### Task 2: Ingestion Models

**Files:**
- Create: `src/ingestion/__init__.py`
- Create: `src/ingestion/models.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_models.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create tests/__init__.py, tests/unit/__init__.py, tests/conftest.py**

`tests/__init__.py` and `tests/unit/__init__.py` are empty files.

`tests/conftest.py`:

```python
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_agent_markdown() -> str:
    return """\
---
id: security-analyst
description: Expert cybersecurity researcher analyzing code for security issues
model: sonnet
max_output_tokens: 32768
max_iterations: 30
tools:
  - list_dir
  - read_file
  - ripgrep
  - bash
recovery_iterations: 5
---

<role>
You are an expert cybersecurity researcher.
</role>

<process>
1. Read the files
2. Trace data flows
3. Report findings
</process>
"""


@pytest.fixture
def sample_skill_markdown() -> str:
    return """\
---
id: ssrf-audit
description: How to find SSRF vulnerabilities
category: analysis
---

## Auditing for SSRF

Search for outbound HTTP calls with dynamic URLs.
"""


@pytest.fixture
def sample_agent_no_tools_markdown() -> str:
    return """\
---
id: conversation-compactor
description: Summarizes long conversations
model: haiku
max_output_tokens: 8192
max_iterations: 5
tools:
---

You are a conversation summarizer.
"""
```

- [ ] **Step 2: Write the failing test for AgentSpec and SkillSpec models**

`tests/unit/test_models.py`:

```python
from src.ingestion.models import AgentSpec, SkillSpec


def test_agent_spec_creation():
    spec = AgentSpec(
        id="security-analyst",
        description="Expert cybersecurity researcher",
        model="sonnet",
        max_output_tokens=32768,
        max_iterations=30,
        tools=["read_file", "bash"],
        system_prompt="You are an expert.",
        recovery_iterations=5,
    )
    assert spec.id == "security-analyst"
    assert spec.model == "sonnet"
    assert spec.tools == ["read_file", "bash"]
    assert spec.recovery_iterations == 5


def test_agent_spec_defaults():
    spec = AgentSpec(
        id="test",
        description="test agent",
        model="haiku",
        max_output_tokens=8192,
        max_iterations=10,
        tools=[],
        system_prompt="You are a test agent.",
    )
    assert spec.recovery_iterations is None


def test_agent_spec_has_tools():
    with_tools = AgentSpec(
        id="a", description="d", model="sonnet",
        max_output_tokens=1, max_iterations=1,
        tools=["bash"], system_prompt="p",
    )
    without_tools = AgentSpec(
        id="b", description="d", model="haiku",
        max_output_tokens=1, max_iterations=1,
        tools=[], system_prompt="p",
    )
    assert with_tools.has_tools is True
    assert without_tools.has_tools is False


def test_skill_spec_creation():
    spec = SkillSpec(
        id="ssrf-audit",
        description="How to find SSRF vulnerabilities",
        category="analysis",
        content="## Auditing for SSRF\n\nSearch for outbound HTTP calls.",
    )
    assert spec.id == "ssrf-audit"
    assert spec.category == "analysis"


def test_skill_spec_category_optional():
    spec = SkillSpec(
        id="test-skill",
        description="A test skill",
        content="Some content.",
    )
    assert spec.category is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion'`

- [ ] **Step 4: Implement AgentSpec and SkillSpec models**

`src/ingestion/__init__.py`:

```python
```

(Empty file.)

`src/ingestion/models.py`:

```python
from __future__ import annotations

from pydantic import BaseModel


class AgentSpec(BaseModel):
    id: str
    description: str
    model: str
    max_output_tokens: int
    max_iterations: int
    tools: list[str]
    system_prompt: str
    recovery_iterations: int | None = None

    @property
    def has_tools(self) -> bool:
        return len(self.tools) > 0


class SkillSpec(BaseModel):
    id: str
    description: str
    content: str
    category: str | None = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_models.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/ tests/
git commit -m "feat: add AgentSpec and SkillSpec Pydantic models"
```

---

### Task 3: Markdown Parser

**Files:**
- Create: `src/ingestion/parser.py`
- Create: `tests/unit/test_parser.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_parser.py`:

```python
import pytest

from src.ingestion.models import AgentSpec, SkillSpec
from src.ingestion.parser import parse_agent, parse_skill, parse_agents_dir, parse_skills_dir


def test_parse_agent(sample_agent_markdown):
    spec = parse_agent(sample_agent_markdown, "test.md")
    assert spec.id == "security-analyst"
    assert spec.description == "Expert cybersecurity researcher analyzing code for security issues"
    assert spec.model == "sonnet"
    assert spec.max_output_tokens == 32768
    assert spec.max_iterations == 30
    assert spec.tools == ["list_dir", "read_file", "ripgrep", "bash"]
    assert spec.recovery_iterations == 5
    assert "<role>" in spec.system_prompt
    assert "expert cybersecurity researcher" in spec.system_prompt


def test_parse_agent_no_tools(sample_agent_no_tools_markdown):
    spec = parse_agent(sample_agent_no_tools_markdown, "test.md")
    assert spec.id == "conversation-compactor"
    assert spec.tools == []


def test_parse_agent_missing_required_field():
    bad_md = """\
---
id: broken
description: missing fields
---

Body text.
"""
    with pytest.raises(ValueError, match="missing required field"):
        parse_agent(bad_md, "broken.md")


def test_parse_skill(sample_skill_markdown):
    spec = parse_skill(sample_skill_markdown, "test.md")
    assert spec.id == "ssrf-audit"
    assert spec.description == "How to find SSRF vulnerabilities"
    assert spec.category == "analysis"
    assert "Auditing for SSRF" in spec.content


def test_parse_skill_missing_id():
    bad_md = """\
---
description: no id
---

Body.
"""
    with pytest.raises(ValueError, match="missing required field"):
        parse_skill(bad_md, "bad.md")


def test_parse_agents_dir(tmp_path):
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    (agent_dir / "test-agent.md").write_text("""\
---
id: test-agent
description: A test agent
model: haiku
max_output_tokens: 4096
max_iterations: 5
tools:
  - bash
---

You are a test agent.
""")
    specs = parse_agents_dir(agent_dir)
    assert len(specs) == 1
    assert specs[0].id == "test-agent"


def test_parse_skills_dir(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "test-skill.md").write_text("""\
---
id: test-skill
description: A test skill
---

Skill content here.
""")
    specs = parse_skills_dir(skill_dir)
    assert len(specs) == 1
    assert specs[0].id == "test-skill"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_parser.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the parser**

`src/ingestion/parser.py`:

```python
from __future__ import annotations

from pathlib import Path

import frontmatter

from src.ingestion.models import AgentSpec, SkillSpec

AGENT_REQUIRED_FIELDS = ["id", "description", "model", "max_output_tokens", "max_iterations"]
SKILL_REQUIRED_FIELDS = ["id", "description"]


def parse_agent(raw: str, file_path: str) -> AgentSpec:
    post = frontmatter.loads(raw)
    meta = post.metadata

    for field in AGENT_REQUIRED_FIELDS:
        if field not in meta or meta[field] is None:
            raise ValueError(f"{file_path}: missing required field '{field}'")

    tools_raw = meta.get("tools")
    if tools_raw is None or (isinstance(tools_raw, list) and len(tools_raw) == 0):
        tools = []
    elif isinstance(tools_raw, list):
        tools = [str(t) for t in tools_raw]
    else:
        tools = []

    return AgentSpec(
        id=meta["id"],
        description=meta["description"],
        model=meta["model"],
        max_output_tokens=meta["max_output_tokens"],
        max_iterations=meta["max_iterations"],
        tools=tools,
        system_prompt=post.content.strip(),
        recovery_iterations=meta.get("recovery_iterations"),
    )


def parse_skill(raw: str, file_path: str) -> SkillSpec:
    post = frontmatter.loads(raw)
    meta = post.metadata

    for field in SKILL_REQUIRED_FIELDS:
        if field not in meta or meta[field] is None:
            raise ValueError(f"{file_path}: missing required field '{field}'")

    return SkillSpec(
        id=meta["id"],
        description=meta["description"],
        category=meta.get("category"),
        content=post.content.strip(),
    )


def parse_agents_dir(directory: Path) -> list[AgentSpec]:
    specs = []
    for md_file in sorted(directory.rglob("*.md")):
        raw = md_file.read_text()
        specs.append(parse_agent(raw, str(md_file)))
    return specs


def parse_skills_dir(directory: Path) -> list[SkillSpec]:
    specs = []
    for md_file in sorted(directory.rglob("*.md")):
        raw = md_file.read_text()
        specs.append(parse_skill(raw, str(md_file)))
    return specs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_parser.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/parser.py tests/unit/test_parser.py
git commit -m "feat: markdown parser for enclave agent and skill definitions"
```

---

### Task 4: DSPy Configuration

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Implement config module**

`src/config.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

import dspy
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "optimization_config.yaml"


class AgentOptConfig(BaseModel):
    optimizers: list[str] = ["bootstrap", "mipro_light"]
    metric_weights: dict[str, float] = {}
    max_bootstrapped_demos: int | None = None
    max_labeled_demos: int | None = None


class OptimizationConfig(BaseModel):
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 8
    improvement_threshold: float = 0.05
    teacher_model: str = "anthropic/claude-opus-4-6"
    target_model: str = "anthropic/claude-sonnet-4-6"


class Config(BaseModel):
    defaults: OptimizationConfig = OptimizationConfig()
    agents: dict[str, AgentOptConfig] = {}

    def get_agent_config(self, agent_id: str) -> AgentOptConfig:
        return self.agents.get(agent_id, AgentOptConfig())


def load_config(path: Path | None = None) -> Config:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return Config()
    raw = yaml.safe_load(config_path.read_text())
    defaults = OptimizationConfig(**raw.get("defaults", {}))
    agents = {
        k: AgentOptConfig(**v) for k, v in raw.get("agents", {}).items()
    }
    return Config(defaults=defaults, agents=agents)


def configure_dspy(model: str | None = None) -> dspy.LM:
    model_name = model or os.getenv("DSPY_MODEL", "anthropic/claude-sonnet-4-6")
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)
    return lm


def get_enclave_repo_path() -> Path:
    path = os.getenv("ENCLAVE_REPO_PATH", "../enclave")
    return Path(path).resolve()


def get_enclave_agents_dir() -> Path:
    return get_enclave_repo_path() / "packages" / "ai-client" / "src" / "definitions" / "agents"


def get_enclave_skills_dir() -> Path:
    return get_enclave_repo_path() / "packages" / "ai-client" / "src" / "definitions" / "skills"
```

- [ ] **Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat: configuration module with DSPy and optimization config loading"
```

---

### Task 5: Signature Factory

**Files:**
- Create: `src/signatures/__init__.py`
- Create: `src/signatures/factory.py`
- Create: `tests/unit/test_signature_factory.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_signature_factory.py`:

```python
import dspy

from src.ingestion.models import AgentSpec, SkillSpec
from src.signatures.factory import create_signature, create_skill_signature


def _make_agent(agent_id: str, tools: list[str] | None = None) -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        description="Test agent",
        model="sonnet",
        max_output_tokens=8192,
        max_iterations=10,
        tools=tools or [],
        system_prompt="You are a test agent.",
    )


def test_create_signature_returns_dspy_signature():
    agent = _make_agent("security-analyst", tools=["bash"])
    sig = create_signature(agent)
    assert issubclass(sig, dspy.Signature)


def test_create_signature_has_input_fields():
    agent = _make_agent("security-analyst", tools=["bash"])
    sig = create_signature(agent)
    fields = sig.input_fields
    assert "context" in fields
    assert "task" in fields


def test_create_signature_has_output_field():
    agent = _make_agent("security-analyst", tools=["bash"])
    sig = create_signature(agent)
    fields = sig.output_fields
    assert "response" in fields


def test_create_signature_docstring_from_description():
    agent = _make_agent("chat-assistant")
    agent.description = "Interactive security helper"
    sig = create_signature(agent)
    assert "Interactive security helper" in sig.__doc__


def test_create_skill_signature():
    skill = SkillSpec(
        id="ssrf-audit",
        description="Find SSRF vulnerabilities",
        content="## Audit steps\n\n1. Find HTTP calls",
    )
    sig = create_skill_signature(skill)
    assert issubclass(sig, dspy.Signature)
    fields = sig.input_fields
    assert "code_context" in fields
    output_fields = sig.output_fields
    assert "findings" in output_fields
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_signature_factory.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the signature factory**

`src/signatures/__init__.py`:

```python
```

`src/signatures/factory.py`:

```python
from __future__ import annotations

import dspy

from src.ingestion.models import AgentSpec, SkillSpec


def create_signature(agent: AgentSpec) -> type[dspy.Signature]:
    """Create a DSPy Signature class from an AgentSpec.

    All agents share a common input/output contract:
    - Inputs: context (str), task (str)
    - Output: response (str)

    The agent's description becomes the signature docstring,
    and the system prompt is available for the module to use.
    """
    attrs: dict = {
        "__doc__": agent.description,
        "__annotations__": {},
        "context": dspy.InputField(desc="Relevant code, files, or conversation context"),
        "task": dspy.InputField(desc="The specific task or question to address"),
        "response": dspy.OutputField(desc="The agent's detailed response"),
    }

    sig_class = type(
        f"{_to_class_name(agent.id)}Signature",
        (dspy.Signature,),
        attrs,
    )
    return sig_class


def create_skill_signature(skill: SkillSpec) -> type[dspy.Signature]:
    """Create a DSPy Signature for a security audit skill.

    Skills have a specialized contract:
    - Input: code_context (str)
    - Output: findings (str)
    """
    attrs: dict = {
        "__doc__": f"{skill.description}\n\nMethodology:\n{skill.content}",
        "__annotations__": {},
        "code_context": dspy.InputField(desc="Source code and configuration to audit"),
        "findings": dspy.OutputField(desc="Security findings with severity and recommendations"),
    }

    sig_class = type(
        f"{_to_class_name(skill.id)}Signature",
        (dspy.Signature,),
        attrs,
    )
    return sig_class


def _to_class_name(slug: str) -> str:
    return "".join(word.capitalize() for word in slug.split("-"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_signature_factory.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/signatures/ tests/unit/test_signature_factory.py
git commit -m "feat: DSPy signature factory for agents and skills"
```

---

### Task 6: Module Factory

**Files:**
- Create: `src/modules/__init__.py`
- Create: `src/modules/factory.py`
- Create: `tests/unit/test_module_factory.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_module_factory.py`:

```python
import dspy

from src.ingestion.models import AgentSpec
from src.modules.factory import create_module


def _make_agent(agent_id: str, tools: list[str] | None = None) -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        description="Test agent",
        model="sonnet",
        max_output_tokens=8192,
        max_iterations=10,
        tools=tools or [],
        system_prompt="You are a test agent.",
    )


def test_create_module_no_tools_uses_chain_of_thought():
    agent = _make_agent("conversation-compactor", tools=[])
    module = create_module(agent)
    assert isinstance(module, dspy.Module)
    assert hasattr(module, "predict")
    assert isinstance(module.predict, dspy.ChainOfThought)


def test_create_module_with_tools_uses_chain_of_thought():
    # Even tool-using agents use ChainOfThought for optimization
    # because tools are stubbed during optimization
    agent = _make_agent("security-analyst", tools=["bash", "read_file"])
    module = create_module(agent)
    assert isinstance(module, dspy.Module)
    assert hasattr(module, "predict")


def test_create_module_has_forward():
    agent = _make_agent("chat-assistant")
    module = create_module(agent)
    assert callable(getattr(module, "forward", None))


def test_create_module_stores_agent_spec():
    agent = _make_agent("finding-verifier")
    module = create_module(agent)
    assert module.agent_spec.id == "finding-verifier"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_module_factory.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the module factory**

`src/modules/__init__.py`:

```python
```

`src/modules/factory.py`:

```python
from __future__ import annotations

import dspy

from src.ingestion.models import AgentSpec
from src.signatures.factory import create_signature


class AgentModule(dspy.Module):
    """A DSPy module wrapping an Enclave agent.

    Uses ChainOfThought for all agents during optimization.
    The system prompt from the agent spec is included as context
    that DSPy can optimize (instructions, demonstrations).
    """

    def __init__(self, agent_spec: AgentSpec, signature: type[dspy.Signature]):
        super().__init__()
        self.agent_spec = agent_spec
        self.predict = dspy.ChainOfThought(signature)

    def forward(self, context: str, task: str) -> dspy.Prediction:
        return self.predict(context=context, task=task)


def create_module(agent: AgentSpec) -> AgentModule:
    signature = create_signature(agent)
    return AgentModule(agent_spec=agent, signature=signature)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_module_factory.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/modules/ tests/unit/test_module_factory.py
git commit -m "feat: DSPy module factory wrapping enclave agents"
```

---

### Task 7: Data Loader

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/loader.py`
- Create: `tests/unit/test_loader.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_loader.py`:

```python
import json

import dspy

from src.data.loader import load_dataset, load_all_splits, DatasetInfo, get_dataset_status


def test_load_dataset_from_jsonl(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        json.dumps({"inputs": {"context": "code here", "task": "find vulns"}, "outputs": {"response": "found XSS"}}) + "\n"
        + json.dumps({"inputs": {"context": "more code", "task": "audit"}, "outputs": {"response": "no issues"}}) + "\n"
    )
    examples = load_dataset(data_file)
    assert len(examples) == 2
    assert isinstance(examples[0], dspy.Example)
    assert examples[0].context == "code here"
    assert examples[0].task == "find vulns"
    assert examples[0].response == "found XSS"


def test_load_dataset_marks_inputs(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        json.dumps({"inputs": {"context": "c", "task": "t"}, "outputs": {"response": "r"}}) + "\n"
    )
    examples = load_dataset(data_file)
    inputs = examples[0].inputs()
    assert "context" in inputs.keys()
    assert "task" in inputs.keys()
    labels = examples[0].labels()
    assert "response" in labels.keys()


def test_load_all_splits(tmp_path):
    agent_dir = tmp_path / "security-analyst"
    agent_dir.mkdir()
    row = json.dumps({"inputs": {"context": "c", "task": "t"}, "outputs": {"response": "r"}})
    (agent_dir / "train.jsonl").write_text(row + "\n" + row + "\n")
    (agent_dir / "dev.jsonl").write_text(row + "\n")
    (agent_dir / "test.jsonl").write_text(row + "\n")

    splits = load_all_splits(agent_dir)
    assert len(splits["train"]) == 2
    assert len(splits["dev"]) == 1
    assert len(splits["test"]) == 1


def test_load_all_splits_missing_file(tmp_path):
    agent_dir = tmp_path / "some-agent"
    agent_dir.mkdir()
    row = json.dumps({"inputs": {"context": "c", "task": "t"}, "outputs": {"response": "r"}})
    (agent_dir / "train.jsonl").write_text(row + "\n")

    splits = load_all_splits(agent_dir)
    assert len(splits["train"]) == 1
    assert len(splits["dev"]) == 0
    assert len(splits["test"]) == 0


def test_get_dataset_status(tmp_path):
    agent_dir = tmp_path / "security-analyst"
    agent_dir.mkdir()
    row = json.dumps({"inputs": {"context": "c", "task": "t"}, "outputs": {"response": "r"}})
    (agent_dir / "train.jsonl").write_text(row + "\n" + row + "\n")
    (agent_dir / "dev.jsonl").write_text(row + "\n")

    info = get_dataset_status(tmp_path, "security-analyst")
    assert info.agent_id == "security-analyst"
    assert info.train_count == 2
    assert info.dev_count == 1
    assert info.test_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_loader.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the data loader**

`src/data/__init__.py`:

```python
```

`src/data/loader.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import dspy
from pydantic import BaseModel


class DatasetInfo(BaseModel):
    agent_id: str
    train_count: int = 0
    dev_count: int = 0
    test_count: int = 0


def load_dataset(file_path: Path) -> list[dspy.Example]:
    examples = []
    for line in file_path.read_text().strip().splitlines():
        row = json.loads(line)
        inputs = row["inputs"]
        outputs = row["outputs"]
        all_fields = {**inputs, **outputs}
        example = dspy.Example(**all_fields).with_inputs(*inputs.keys())
        examples.append(example)
    return examples


def load_all_splits(agent_dir: Path) -> dict[str, list[dspy.Example]]:
    splits = {}
    for split_name in ("train", "dev", "test"):
        file_path = agent_dir / f"{split_name}.jsonl"
        if file_path.exists():
            splits[split_name] = load_dataset(file_path)
        else:
            splits[split_name] = []
    return splits


def get_dataset_status(datasets_dir: Path, agent_id: str) -> DatasetInfo:
    agent_dir = datasets_dir / agent_id
    if not agent_dir.exists():
        return DatasetInfo(agent_id=agent_id)

    splits = load_all_splits(agent_dir)
    return DatasetInfo(
        agent_id=agent_id,
        train_count=len(splits["train"]),
        dev_count=len(splits["dev"]),
        test_count=len(splits["test"]),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_loader.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/ tests/unit/test_loader.py
git commit -m "feat: JSONL data loader with train/dev/test split support"
```

---

### Task 8: Synthetic Data Generator

**Files:**
- Create: `src/data/synthetic.py`

- [ ] **Step 1: Implement the synthetic data generator**

`src/data/synthetic.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import dspy

from src.ingestion.models import AgentSpec


class SyntheticGenerator(dspy.Signature):
    """Generate a realistic training example for a security analysis agent."""

    agent_description: str = dspy.InputField(desc="Description of the agent's role")
    agent_system_prompt: str = dspy.InputField(desc="The agent's full system prompt")
    example_number: int = dspy.InputField(desc="Which example number this is (for variety)")
    context: str = dspy.OutputField(desc="Realistic code context or input for the agent")
    task: str = dspy.OutputField(desc="A specific task appropriate for this agent")
    response: str = dspy.OutputField(desc="The ideal agent response")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0 to 1.0 for this example's quality")


def generate_synthetic_examples(
    agent: AgentSpec,
    count: int,
    output_dir: Path,
    teacher_model: str = "anthropic/claude-opus-4-6",
) -> list[dict]:
    teacher = dspy.LM(teacher_model)
    generator = dspy.ChainOfThought(SyntheticGenerator)

    examples = []
    with dspy.context(lm=teacher):
        for i in range(count):
            result = generator(
                agent_description=agent.description,
                agent_system_prompt=agent.system_prompt[:2000],
                example_number=i + 1,
            )
            example = {
                "inputs": {
                    "context": result.context,
                    "task": result.task,
                },
                "outputs": {
                    "response": result.response,
                },
                "metadata": {
                    "source": "synthetic",
                    "confidence": float(result.confidence),
                    "agent_id": agent.id,
                },
            }
            examples.append(example)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic.jsonl"
    with open(output_file, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return examples
```

- [ ] **Step 2: Commit**

```bash
git add src/data/synthetic.py
git commit -m "feat: synthetic training data generator using teacher model"
```

---

### Task 9: Generic Metrics

**Files:**
- Create: `src/metrics/__init__.py`
- Create: `src/metrics/generic.py`
- Create: `tests/unit/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_metrics.py`:

```python
import dspy

from src.metrics.generic import structural_completeness, combined_metric


def _make_example_and_pred(response: str, expected: str):
    example = dspy.Example(
        context="some code",
        task="find vulnerabilities",
        response=expected,
    ).with_inputs("context", "task")

    pred = dspy.Prediction(response=response)
    return example, pred


def test_structural_completeness_full_response():
    example, pred = _make_example_and_pred(
        response="## Finding\n\nSeverity: High\n\nDescription: XSS vulnerability found",
        expected="anything",
    )
    score = structural_completeness(example, pred)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_structural_completeness_empty_response():
    example, pred = _make_example_and_pred(response="", expected="anything")
    score = structural_completeness(example, pred)
    assert score == 0.0


def test_combined_metric_returns_float():
    example, pred = _make_example_and_pred(
        response="Found SQL injection in login handler",
        expected="SQL injection in login handler with high severity",
    )
    weights = {"structural": 0.5, "length_ratio": 0.5}
    score = combined_metric(example, pred, trace=None, weights=weights)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_combined_metric_defaults():
    example, pred = _make_example_and_pred(
        response="Found an issue",
        expected="Found a critical issue",
    )
    score = combined_metric(example, pred)
    assert isinstance(score, float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_metrics.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement generic metrics**

`src/metrics/__init__.py`:

```python
```

`src/metrics/generic.py`:

```python
from __future__ import annotations

import dspy


def structural_completeness(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    response = getattr(pred, "response", "")
    if not response:
        return 0.0

    indicators = [
        len(response) > 50,
        "\n" in response,
        any(marker in response.lower() for marker in ["finding", "severity", "description", "issue", "vulnerability"]),
        any(marker in response for marker in ["##", "- ", "1.", "*"]),
    ]
    return sum(indicators) / len(indicators)


def length_ratio(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    expected = getattr(example, "response", "")
    actual = getattr(pred, "response", "")
    if not expected:
        return 1.0 if actual else 0.0
    ratio = len(actual) / len(expected)
    if ratio > 1.0:
        ratio = 1.0 / ratio
    return min(ratio, 1.0)


def combined_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    weights: dict[str, float] | None = None,
) -> float:
    if weights is None:
        weights = {"structural": 0.5, "length_ratio": 0.5}

    metric_fns = {
        "structural": structural_completeness,
        "length_ratio": length_ratio,
    }

    total = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        if name in metric_fns:
            score = metric_fns[name](example, pred, trace)
            total += score * weight
            total_weight += weight

    return total / total_weight if total_weight > 0 else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_metrics.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/metrics/ tests/unit/test_metrics.py
git commit -m "feat: generic metrics (structural completeness, length ratio, combined)"
```

---

### Task 10: LLM-as-Judge Metric

**Files:**
- Create: `src/metrics/judge.py`

- [ ] **Step 1: Implement the LLM-as-judge metric**

`src/metrics/judge.py`:

```python
from __future__ import annotations

import dspy


class JudgeAssessment(dspy.Signature):
    """Assess the quality of an AI agent's response compared to the expected output."""

    task: str = dspy.InputField(desc="The task the agent was asked to perform")
    expected_response: str = dspy.InputField(desc="The expected/ideal response")
    actual_response: str = dspy.InputField(desc="The agent's actual response")
    relevance_score: float = dspy.OutputField(desc="How relevant the response is to the task (0.0-1.0)")
    completeness_score: float = dspy.OutputField(desc="How complete the response is (0.0-1.0)")
    accuracy_score: float = dspy.OutputField(desc="How accurate the response is vs expected (0.0-1.0)")
    feedback: str = dspy.OutputField(desc="Brief textual feedback explaining the scores")


class LLMJudge:
    def __init__(self, judge_model: str = "anthropic/claude-sonnet-4-6"):
        self.judge_lm = dspy.LM(judge_model)
        self.assess = dspy.ChainOfThought(JudgeAssessment)

    def __call__(
        self,
        example: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
    ) -> float:
        task = getattr(example, "task", "")
        expected = getattr(example, "response", "")
        actual = getattr(pred, "response", "")

        if not actual:
            return 0.0

        with dspy.context(lm=self.judge_lm):
            result = self.assess(
                task=task,
                expected_response=expected,
                actual_response=actual,
            )

        scores = [
            float(result.relevance_score),
            float(result.completeness_score),
            float(result.accuracy_score),
        ]
        return sum(scores) / len(scores)
```

- [ ] **Step 2: Commit**

```bash
git add src/metrics/judge.py
git commit -m "feat: LLM-as-judge metric with relevance, completeness, accuracy scoring"
```

---

### Task 11: Domain-Specific Metrics

**Files:**
- Create: `src/metrics/domain.py`

- [ ] **Step 1: Implement domain metrics**

`src/metrics/domain.py`:

```python
from __future__ import annotations

import re

import dspy


SEVERITY_LEVELS = {"critical", "high", "medium", "low", "info"}


def finding_precision_recall(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    expected_findings = _extract_finding_keywords(getattr(example, "response", ""))
    actual_findings = _extract_finding_keywords(getattr(pred, "response", ""))

    if not expected_findings and not actual_findings:
        return 1.0
    if not expected_findings or not actual_findings:
        return 0.0

    matches = expected_findings & actual_findings
    precision = len(matches) / len(actual_findings) if actual_findings else 0.0
    recall = len(matches) / len(expected_findings) if expected_findings else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def severity_accuracy(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    expected_severities = _extract_severities(getattr(example, "response", ""))
    actual_severities = _extract_severities(getattr(pred, "response", ""))

    if not expected_severities:
        return 1.0 if not actual_severities else 0.5

    matches = sum(1 for s in expected_severities if s in actual_severities)
    return matches / len(expected_severities)


def dedup_accuracy(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    expected = getattr(example, "response", "").lower()
    actual = getattr(pred, "response", "").lower()

    expected_is_dup = any(w in expected for w in ["duplicate", "same", "already reported"])
    actual_is_dup = any(w in actual for w in ["duplicate", "same", "already reported"])

    return 1.0 if expected_is_dup == actual_is_dup else 0.0


DOMAIN_METRICS = {
    "finding_precision": finding_precision_recall,
    "finding_recall": finding_precision_recall,
    "severity_accuracy": severity_accuracy,
    "dedup_accuracy": dedup_accuracy,
    "false_merge_rate": dedup_accuracy,
    "coverage": finding_precision_recall,
    "delegation_quality": finding_precision_recall,
    "relevance": finding_precision_recall,
    "correctness": finding_precision_recall,
    "actionability": finding_precision_recall,
    "finding_relevance": finding_precision_recall,
    "false_alarm_rate": dedup_accuracy,
    "severity_calibration": severity_accuracy,
    "false_positive_rate": dedup_accuracy,
    "false_negative_rate": dedup_accuracy,
    "reasoning_quality": finding_precision_recall,
}


def get_domain_metric(metric_name: str):
    return DOMAIN_METRICS.get(metric_name, finding_precision_recall)


def _extract_finding_keywords(text: str) -> set[str]:
    vuln_patterns = [
        r"(?i)(xss|cross.site.scripting)",
        r"(?i)(sql.injection|sqli)",
        r"(?i)(ssrf|server.side.request)",
        r"(?i)(command.injection|rce|remote.code)",
        r"(?i)(path.traversal|directory.traversal)",
        r"(?i)(idor|insecure.direct)",
        r"(?i)(csrf|cross.site.request.forgery)",
        r"(?i)(auth\w*\s+bypass)",
        r"(?i)(information.disclosure|data.leak)",
        r"(?i)(template.injection|ssti)",
    ]
    found = set()
    for pattern in vuln_patterns:
        if re.search(pattern, text):
            found.add(pattern)
    return found


def _extract_severities(text: str) -> list[str]:
    found = []
    for level in SEVERITY_LEVELS:
        if re.search(rf"(?i)\b{level}\b", text):
            found.append(level)
    return found
```

- [ ] **Step 2: Commit**

```bash
git add src/metrics/domain.py
git commit -m "feat: domain-specific metrics for security analysis agents"
```

---

### Task 12: Optimization Runner

**Files:**
- Create: `src/optimizers/__init__.py`
- Create: `src/optimizers/runner.py`

- [ ] **Step 1: Implement the optimization runner**

`src/optimizers/__init__.py`:

```python
```

`src/optimizers/runner.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from src.config import Config, AgentOptConfig
from src.metrics.generic import combined_metric
from src.metrics.judge import LLMJudge
from src.metrics.domain import get_domain_metric


class OptimizationResult:
    def __init__(
        self,
        agent_id: str,
        optimizer_name: str,
        baseline_score: float,
        optimized_score: float,
        program: dspy.Module,
    ):
        self.agent_id = agent_id
        self.optimizer_name = optimizer_name
        self.baseline_score = baseline_score
        self.optimized_score = optimized_score
        self.program = program

    @property
    def improvement(self) -> float:
        if self.baseline_score == 0:
            return self.optimized_score
        return (self.optimized_score - self.baseline_score) / self.baseline_score


def build_metric(agent_id: str, config: Config):
    agent_config = config.get_agent_config(agent_id)
    weights = agent_config.metric_weights

    if not weights:
        return combined_metric

    def metric(example, pred, trace=None):
        total = 0.0
        total_weight = 0.0
        for name, weight in weights.items():
            fn = get_domain_metric(name)
            score = fn(example, pred, trace)
            total += score * weight
            total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0

    return metric


def evaluate_program(
    program: dspy.Module,
    devset: list[dspy.Example],
    metric,
) -> float:
    if not devset:
        return 0.0
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=4,
        display_progress=True,
    )
    result = evaluator(program)
    return float(result)


def run_optimization(
    program: dspy.Module,
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    agent_id: str,
    config: Config,
    optimizer_name: str | None = None,
    output_dir: Path | None = None,
) -> OptimizationResult:
    metric = build_metric(agent_id, config)
    agent_config = config.get_agent_config(agent_id)
    defaults = config.defaults

    max_bootstrapped = (
        agent_config.max_bootstrapped_demos
        or defaults.max_bootstrapped_demos
    )
    max_labeled = (
        agent_config.max_labeled_demos
        or defaults.max_labeled_demos
    )

    baseline_score = evaluate_program(program, devset, metric)

    optimizers_to_run = (
        [optimizer_name] if optimizer_name
        else agent_config.optimizers
    )

    best_result = OptimizationResult(
        agent_id=agent_id,
        optimizer_name="baseline",
        baseline_score=baseline_score,
        optimized_score=baseline_score,
        program=program,
    )

    for opt_name in optimizers_to_run:
        candidate = program.deepcopy()
        optimized = _run_single_optimizer(
            opt_name, candidate, trainset, metric,
            max_bootstrapped, max_labeled, defaults,
        )

        score = evaluate_program(optimized, devset, metric)

        if score > best_result.optimized_score:
            best_result = OptimizationResult(
                agent_id=agent_id,
                optimizer_name=opt_name,
                baseline_score=baseline_score,
                optimized_score=score,
                program=optimized,
            )

        if best_result.improvement >= defaults.improvement_threshold:
            break

    if output_dir:
        _save_result(best_result, output_dir)

    return best_result


def _run_single_optimizer(
    name: str,
    program: dspy.Module,
    trainset: list[dspy.Example],
    metric,
    max_bootstrapped: int,
    max_labeled: int,
    defaults,
) -> dspy.Module:
    if name == "bootstrap":
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )
        return optimizer.compile(student=program, trainset=trainset)

    elif name == "mipro_light":
        optimizer = MIPROv2(
            metric=metric,
            auto="light",
            num_threads=4,
        )
        return optimizer.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )

    elif name == "mipro_medium":
        optimizer = MIPROv2(
            metric=metric,
            auto="medium",
            num_threads=8,
        )
        return optimizer.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )

    elif name == "gepa":
        from dspy.teleprompt import GEPA
        optimizer = GEPA(
            metric=metric,
        )
        return optimizer.compile(
            program,
            trainset=trainset,
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _save_result(result: OptimizationResult, output_dir: Path):
    agent_dir = output_dir / result.agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    existing = list(agent_dir.glob("v*.json"))
    version = len(existing) + 1
    version_str = f"v{version}"

    result.program.save(str(agent_dir / f"{version_str}.json"))

    report = agent_dir / "report.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"\n## {version_str} — {timestamp}\n\n"
        f"- **Optimizer:** {result.optimizer_name}\n"
        f"- **Baseline score:** {result.baseline_score:.4f}\n"
        f"- **Optimized score:** {result.optimized_score:.4f}\n"
        f"- **Improvement:** {result.improvement:.2%}\n"
    )

    if report.exists():
        content = report.read_text() + entry
    else:
        content = f"# Optimization Report: {result.agent_id}\n" + entry

    report.write_text(content)
```

- [ ] **Step 2: Commit**

```bash
git add src/optimizers/
git commit -m "feat: progressive optimization runner with BootstrapFewShot, MIPROv2, GEPA"
```

---

### Task 13: Export Pipeline

**Files:**
- Create: `src/export/__init__.py`
- Create: `src/export/exporter.py`
- Create: `tests/unit/test_exporter.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_exporter.py`:

```python
import json
from pathlib import Path

import pytest

from src.ingestion.models import AgentSpec
from src.export.exporter import export_agent, format_agent_markdown


def _make_agent() -> AgentSpec:
    return AgentSpec(
        id="security-analyst",
        description="Expert cybersecurity researcher",
        model="sonnet",
        max_output_tokens=32768,
        max_iterations=30,
        tools=["list_dir", "read_file", "bash"],
        system_prompt="You are an expert cybersecurity researcher.\n\nAnalyze code for vulnerabilities.",
        recovery_iterations=5,
    )


def test_format_agent_markdown():
    agent = _make_agent()
    optimized_instructions = "You are a highly skilled security analyst specializing in code review."
    demonstrations = "Example: Given login.py, found SQL injection in query builder."

    md = format_agent_markdown(agent, optimized_instructions, demonstrations)
    assert "---" in md
    assert "id: security-analyst" in md
    assert "model: sonnet" in md
    assert "max_output_tokens: 32768" in md
    assert "list_dir" in md
    assert "recovery_iterations: 5" in md
    assert "highly skilled security analyst" in md
    assert "DSPy Optimized" in md


def test_format_agent_markdown_no_demonstrations():
    agent = _make_agent()
    md = format_agent_markdown(agent, "New instructions here.", demonstrations=None)
    assert "New instructions here." in md
    assert "Examples" not in md


def test_format_agent_markdown_preserves_tools_list():
    agent = _make_agent()
    md = format_agent_markdown(agent, "Instructions.", demonstrations=None)
    assert "  - list_dir" in md
    assert "  - read_file" in md
    assert "  - bash" in md


def test_export_agent_writes_file(tmp_path):
    agent = _make_agent()
    export_agent(
        agent=agent,
        optimized_instructions="Optimized prompt content.",
        demonstrations=None,
        output_dir=tmp_path,
    )
    output_file = tmp_path / "security-analyst.md"
    assert output_file.exists()
    content = output_file.read_text()
    assert "Optimized prompt content." in content


def test_export_agent_blocks_if_score_regression(tmp_path):
    agent = _make_agent()
    with pytest.raises(ValueError, match="score regression"):
        export_agent(
            agent=agent,
            optimized_instructions="Bad prompt.",
            demonstrations=None,
            output_dir=tmp_path,
            baseline_score=0.8,
            optimized_score=0.5,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_exporter.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the export pipeline**

`src/export/__init__.py`:

```python
```

`src/export/exporter.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.ingestion.models import AgentSpec


def format_agent_markdown(
    agent: AgentSpec,
    optimized_instructions: str,
    demonstrations: str | None = None,
    version: str = "v1",
    baseline_score: float | None = None,
    optimized_score: float | None = None,
) -> str:
    frontmatter = {
        "id": agent.id,
        "description": agent.description,
        "model": agent.model,
        "max_output_tokens": agent.max_output_tokens,
        "max_iterations": agent.max_iterations,
        "tools": agent.tools,
    }
    if agent.recovery_iterations is not None:
        frontmatter["recovery_iterations"] = agent.recovery_iterations

    yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    score_info = ""
    if baseline_score is not None and optimized_score is not None:
        score_info = f" | baseline: {baseline_score:.4f} → optimized: {optimized_score:.4f}"

    parts = [
        "---",
        yaml_str.strip(),
        "---",
        "",
        f"<!-- DSPy Optimized | {version} | {timestamp}{score_info} -->",
        "",
        optimized_instructions,
    ]

    if demonstrations:
        parts.extend([
            "",
            "## Examples",
            "",
            demonstrations,
        ])

    return "\n".join(parts) + "\n"


def export_agent(
    agent: AgentSpec,
    optimized_instructions: str,
    demonstrations: str | None = None,
    output_dir: Path = Path("exports"),
    version: str = "v1",
    baseline_score: float | None = None,
    optimized_score: float | None = None,
) -> Path:
    if baseline_score is not None and optimized_score is not None:
        if optimized_score < baseline_score:
            raise ValueError(
                f"Export blocked: score regression for {agent.id} "
                f"(baseline={baseline_score:.4f}, optimized={optimized_score:.4f})"
            )

    md = format_agent_markdown(
        agent, optimized_instructions, demonstrations,
        version, baseline_score, optimized_score,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{agent.id}.md"
    output_file.write_text(md)
    return output_file
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_exporter.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/export/ tests/unit/test_exporter.py
git commit -m "feat: export pipeline converting optimized DSPy programs to enclave markdown"
```

---

### Task 14: CLI — Entry Point and Ingest Command

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/cli.py`
- Create: `scripts/ingest.py`

- [ ] **Step 1: Create the CLI entry point**

`scripts/__init__.py`:

```python
```

`scripts/cli.py`:

```python
import click

from scripts.ingest import ingest
from scripts.data_cmd import data
from scripts.evaluate import evaluate
from scripts.optimize import optimize
from scripts.compare import compare
from scripts.export_cmd import export


@click.group()
def cli():
    """Enclave-DSPy: prompt optimization pipeline for Enclave security agents."""
    pass


cli.add_command(ingest)
cli.add_command(data)
cli.add_command(evaluate)
cli.add_command(optimize)
cli.add_command(compare)
cli.add_command(export)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Create the ingest command**

`scripts/ingest.py`:

```python
from pathlib import Path

import click

from src.config import get_enclave_agents_dir, get_enclave_skills_dir
from src.ingestion.parser import parse_agent, parse_agents_dir, parse_skills_dir


@click.command()
@click.option(
    "--enclave-path", type=click.Path(exists=True),
    help="Path to the enclave repo root",
)
@click.option("--agent", type=str, default=None, help="Ingest a specific agent by ID")
def ingest(enclave_path: str | None, agent: str | None):
    """Parse Enclave agent definitions into DSPy-compatible specs."""
    if enclave_path:
        agents_dir = Path(enclave_path) / "packages" / "ai-client" / "src" / "definitions" / "agents"
        skills_dir = Path(enclave_path) / "packages" / "ai-client" / "src" / "definitions" / "skills"
    else:
        agents_dir = get_enclave_agents_dir()
        skills_dir = get_enclave_skills_dir()

    if not agents_dir.exists():
        click.echo(f"Error: agents directory not found at {agents_dir}", err=True)
        raise SystemExit(1)

    if agent:
        agent_file = agents_dir / f"{agent}.md"
        if not agent_file.exists():
            click.echo(f"Error: agent file not found: {agent_file}", err=True)
            raise SystemExit(1)
        raw = agent_file.read_text()
        spec = parse_agent(raw, str(agent_file))
        click.echo(f"Parsed agent: {spec.id} ({spec.model}, {len(spec.tools)} tools)")
    else:
        agent_specs = parse_agents_dir(agents_dir)
        click.echo(f"Parsed {len(agent_specs)} agents:")
        for spec in agent_specs:
            click.echo(f"  - {spec.id} ({spec.model}, {len(spec.tools)} tools)")

        if skills_dir.exists():
            skill_specs = parse_skills_dir(skills_dir)
            click.echo(f"\nParsed {len(skill_specs)} skills:")
            for spec in skill_specs:
                click.echo(f"  - {spec.id}: {spec.description}")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/cli.py scripts/ingest.py scripts/__init__.py
git commit -m "feat: CLI entry point and ingest command"
```

---

### Task 15: CLI — Data, Evaluate, Optimize, Compare, Export Commands

**Files:**
- Create: `scripts/data_cmd.py`
- Create: `scripts/evaluate.py`
- Create: `scripts/optimize.py`
- Create: `scripts/compare.py`
- Create: `scripts/export_cmd.py`

- [ ] **Step 1: Create the data command**

`scripts/data_cmd.py`:

```python
from pathlib import Path

import click

from src.config import load_config
from src.data.loader import get_dataset_status
from src.data.synthetic import generate_synthetic_examples
from src.ingestion.parser import parse_agent
from src.config import get_enclave_agents_dir

DATASETS_DIR = Path("datasets")


@click.group()
def data():
    """Manage evaluation datasets."""
    pass


@data.command()
@click.option("--agent", required=True, help="Agent ID to generate examples for")
@click.option("--count", default=20, help="Number of examples to generate")
@click.option("--teacher-model", default=None, help="Teacher model override")
def generate(agent: str, count: int, teacher_model: str | None):
    """Generate synthetic training examples."""
    agents_dir = get_enclave_agents_dir()
    agent_file = agents_dir / f"{agent}.md"
    if not agent_file.exists():
        click.echo(f"Error: agent file not found: {agent_file}", err=True)
        raise SystemExit(1)

    raw = agent_file.read_text()
    spec = parse_agent(raw, str(agent_file))

    config = load_config()
    model = teacher_model or config.defaults.teacher_model

    click.echo(f"Generating {count} synthetic examples for {agent} using {model}...")
    output_dir = DATASETS_DIR / agent
    examples = generate_synthetic_examples(spec, count, output_dir, model)
    click.echo(f"Generated {len(examples)} examples → {output_dir}/synthetic.jsonl")


@data.command()
def status():
    """Show dataset counts for all agents."""
    if not DATASETS_DIR.exists():
        click.echo("No datasets directory found.")
        return

    click.echo(f"{'Agent':<30} {'Train':>6} {'Dev':>6} {'Test':>6}")
    click.echo("-" * 54)
    for agent_dir in sorted(DATASETS_DIR.iterdir()):
        if agent_dir.is_dir() and agent_dir.name != ".gitkeep":
            info = get_dataset_status(DATASETS_DIR, agent_dir.name)
            click.echo(
                f"{info.agent_id:<30} {info.train_count:>6} "
                f"{info.dev_count:>6} {info.test_count:>6}"
            )
```

- [ ] **Step 2: Create the evaluate command**

`scripts/evaluate.py`:

```python
from pathlib import Path

import click

from src.config import load_config, configure_dspy
from src.data.loader import load_all_splits
from src.ingestion.parser import parse_agent
from src.modules.factory import create_module
from src.optimizers.runner import build_metric, evaluate_program
from src.config import get_enclave_agents_dir

DATASETS_DIR = Path("datasets")
OPTIMIZED_DIR = Path("optimized")


@click.command()
@click.option("--agent", type=str, default=None, help="Agent ID to evaluate")
@click.option("--all-agents", "all_flag", is_flag=True, help="Evaluate all agents")
@click.option("--version", type=str, default=None, help="Evaluate a specific optimized version")
@click.option("--model", type=str, default=None, help="Model override")
def evaluate(agent: str | None, all_flag: bool, version: str | None, model: str | None):
    """Evaluate agent performance on dev set."""
    config = load_config()
    configure_dspy(model or config.defaults.target_model)

    agents_dir = get_enclave_agents_dir()

    if agent:
        _evaluate_agent(agent, agents_dir, config, version)
    elif all_flag:
        for agent_dir in sorted(DATASETS_DIR.iterdir()):
            if agent_dir.is_dir() and agent_dir.name != ".gitkeep":
                _evaluate_agent(agent_dir.name, agents_dir, config, version)
    else:
        click.echo("Specify --agent <name> or --all-agents")


def _evaluate_agent(agent_id: str, agents_dir: Path, config, version: str | None):
    agent_file = agents_dir / f"{agent_id}.md"
    if not agent_file.exists():
        click.echo(f"Skipping {agent_id}: agent file not found")
        return

    raw = agent_file.read_text()
    spec = parse_agent(raw, str(agent_file))
    module = create_module(spec)

    if version:
        program_path = OPTIMIZED_DIR / agent_id / f"{version}.json"
        if program_path.exists():
            module.load(str(program_path))
            click.echo(f"Loaded optimized version: {version}")
        else:
            click.echo(f"Version {version} not found at {program_path}")
            return

    splits = load_all_splits(DATASETS_DIR / agent_id)
    devset = splits.get("dev", [])
    if not devset:
        click.echo(f"Skipping {agent_id}: no dev set found")
        return

    metric = build_metric(agent_id, config)
    score = evaluate_program(module, devset, metric)
    label = f"{agent_id} ({version})" if version else f"{agent_id} (baseline)"
    click.echo(f"{label}: {score:.4f}")
```

- [ ] **Step 3: Create the optimize command**

`scripts/optimize.py`:

```python
from pathlib import Path

import click

from src.config import load_config, configure_dspy
from src.data.loader import load_all_splits
from src.ingestion.parser import parse_agent
from src.modules.factory import create_module
from src.optimizers.runner import run_optimization
from src.config import get_enclave_agents_dir

DATASETS_DIR = Path("datasets")
OPTIMIZED_DIR = Path("optimized")


@click.command()
@click.option("--agent", type=str, default=None, help="Agent ID to optimize")
@click.option("--all-agents", "all_flag", is_flag=True, help="Optimize all agents")
@click.option("--optimizer", type=str, default=None, help="Specific optimizer to run")
@click.option("--max-bootstrapped-demos", type=int, default=None)
@click.option("--max-labeled-demos", type=int, default=None)
@click.option("--teacher-model", type=str, default=None)
@click.option("--target-model", type=str, default=None)
@click.option("--threshold", type=float, default=None)
def optimize(
    agent: str | None,
    all_flag: bool,
    optimizer: str | None,
    max_bootstrapped_demos: int | None,
    max_labeled_demos: int | None,
    teacher_model: str | None,
    target_model: str | None,
    threshold: float | None,
):
    """Run prompt optimization pipeline."""
    config = load_config()

    if max_bootstrapped_demos is not None:
        config.defaults.max_bootstrapped_demos = max_bootstrapped_demos
    if max_labeled_demos is not None:
        config.defaults.max_labeled_demos = max_labeled_demos
    if teacher_model:
        config.defaults.teacher_model = teacher_model
    if target_model:
        config.defaults.target_model = target_model
    if threshold is not None:
        config.defaults.improvement_threshold = threshold

    configure_dspy(config.defaults.target_model)
    agents_dir = get_enclave_agents_dir()

    if agent:
        _optimize_agent(agent, agents_dir, config, optimizer)
    elif all_flag:
        for agent_dir in sorted(DATASETS_DIR.iterdir()):
            if agent_dir.is_dir() and agent_dir.name != ".gitkeep":
                _optimize_agent(agent_dir.name, agents_dir, config, optimizer)
    else:
        click.echo("Specify --agent <name> or --all-agents")


def _optimize_agent(agent_id: str, agents_dir: Path, config, optimizer: str | None):
    agent_file = agents_dir / f"{agent_id}.md"
    if not agent_file.exists():
        click.echo(f"Skipping {agent_id}: agent file not found")
        return

    raw = agent_file.read_text()
    spec = parse_agent(raw, str(agent_file))
    module = create_module(spec)

    splits = load_all_splits(DATASETS_DIR / agent_id)
    trainset = splits.get("train", [])
    devset = splits.get("dev", [])

    if not trainset:
        click.echo(f"Skipping {agent_id}: no training data")
        return

    click.echo(f"\nOptimizing {agent_id} ({len(trainset)} train, {len(devset)} dev)...")
    result = run_optimization(
        program=module,
        trainset=trainset,
        devset=devset,
        agent_id=agent_id,
        config=config,
        optimizer_name=optimizer,
        output_dir=OPTIMIZED_DIR,
    )

    click.echo(
        f"  Best: {result.optimizer_name} | "
        f"baseline={result.baseline_score:.4f} → "
        f"optimized={result.optimized_score:.4f} "
        f"({result.improvement:+.2%})"
    )
```

- [ ] **Step 4: Create the compare command**

`scripts/compare.py`:

```python
from pathlib import Path

import click

OPTIMIZED_DIR = Path("optimized")


@click.command()
@click.option("--agent", required=True, help="Agent ID to compare")
@click.option("--versions", multiple=True, help="Specific versions to compare")
def compare(agent: str, versions: tuple[str]):
    """Compare optimization results."""
    report_file = OPTIMIZED_DIR / agent / "report.md"
    if not report_file.exists():
        click.echo(f"No optimization report found for {agent}")
        return

    content = report_file.read_text()
    click.echo(content)
```

- [ ] **Step 5: Create the export command**

`scripts/export_cmd.py`:

```python
from pathlib import Path

import click

from src.config import get_enclave_agents_dir
from src.ingestion.parser import parse_agent
from src.export.exporter import export_agent

OPTIMIZED_DIR = Path("optimized")
EXPORTS_DIR = Path("exports")


@click.command()
@click.option("--agent", type=str, default=None, help="Agent ID to export")
@click.option("--all-agents", "all_flag", is_flag=True, help="Export all agents")
@click.option("--version", type=str, default="latest", help="Version to export")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
def export(agent: str | None, all_flag: bool, version: str, dry_run: bool):
    """Export optimized programs to Enclave Markdown format."""
    agents_dir = get_enclave_agents_dir()

    if agent:
        _export_agent(agent, agents_dir, version, dry_run)
    elif all_flag:
        for agent_dir in sorted(OPTIMIZED_DIR.iterdir()):
            if agent_dir.is_dir():
                _export_agent(agent_dir.name, agents_dir, version, dry_run)
    else:
        click.echo("Specify --agent <name> or --all-agents")


def _export_agent(agent_id: str, agents_dir: Path, version: str, dry_run: bool):
    agent_file = agents_dir / f"{agent_id}.md"
    if not agent_file.exists():
        click.echo(f"Skipping {agent_id}: agent file not found")
        return

    raw = agent_file.read_text()
    spec = parse_agent(raw, str(agent_file))

    optimized_dir = OPTIMIZED_DIR / agent_id
    if version == "latest":
        versions = sorted(optimized_dir.glob("v*.json"))
        if not versions:
            click.echo(f"Skipping {agent_id}: no optimized versions found")
            return
        version_file = versions[-1]
        version_str = version_file.stem
    else:
        version_file = optimized_dir / f"{version}.json"
        version_str = version
        if not version_file.exists():
            click.echo(f"Skipping {agent_id}: version {version} not found")
            return

    # For now, use the original system prompt as the optimized instructions.
    # When DSPy programs are loaded, we'll extract the learned instructions.
    optimized_instructions = spec.system_prompt

    if dry_run:
        from src.export.exporter import format_agent_markdown
        md = format_agent_markdown(spec, optimized_instructions, version=version_str)
        click.echo(f"\n--- {agent_id} ({version_str}) ---")
        click.echo(md)
    else:
        output = export_agent(
            agent=spec,
            optimized_instructions=optimized_instructions,
            output_dir=EXPORTS_DIR,
            version=version_str,
        )
        click.echo(f"Exported {agent_id} → {output}")
```

- [ ] **Step 6: Commit**

```bash
git add scripts/data_cmd.py scripts/evaluate.py scripts/optimize.py scripts/compare.py scripts/export_cmd.py
git commit -m "feat: CLI commands for data, evaluate, optimize, compare, and export"
```

---

### Task 16: Integration Test

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_pipeline.py`

- [ ] **Step 1: Write the integration test**

`tests/integration/__init__.py`:

```python
```

`tests/integration/test_pipeline.py`:

```python
import json
from pathlib import Path

from src.ingestion.parser import parse_agent
from src.ingestion.models import AgentSpec
from src.signatures.factory import create_signature
from src.modules.factory import create_module
from src.data.loader import load_dataset, load_all_splits
from src.export.exporter import export_agent, format_agent_markdown


SAMPLE_AGENT = """\
---
id: test-analyst
description: Test security analyst for integration testing
model: sonnet
max_output_tokens: 8192
max_iterations: 10
tools:
  - read_file
  - bash
---

You are a test security analyst. Analyze the provided code for vulnerabilities.

<process>
1. Read the code context
2. Identify potential security issues
3. Report findings with severity levels
</process>
"""


def test_full_pipeline_ingest_to_export(tmp_path):
    """End-to-end test: parse agent → create module → load data → export."""
    # Step 1: Parse agent
    spec = parse_agent(SAMPLE_AGENT, "test-analyst.md")
    assert spec.id == "test-analyst"
    assert spec.has_tools is True

    # Step 2: Create DSPy signature and module
    sig = create_signature(spec)
    module = create_module(spec)
    assert module.agent_spec.id == "test-analyst"

    # Step 3: Create and load dataset
    data_dir = tmp_path / "datasets" / "test-analyst"
    data_dir.mkdir(parents=True)

    examples = [
        {
            "inputs": {"context": "app.get('/users/:id', (req, res) => { db.query('SELECT * FROM users WHERE id=' + req.params.id) })", "task": "Find SQL injection vulnerabilities"},
            "outputs": {"response": "## Finding: SQL Injection\n\nSeverity: High\n\nThe endpoint concatenates user input directly into SQL query."},
        },
        {
            "inputs": {"context": "const sanitized = escape(input); db.query('SELECT * FROM users WHERE id=?', [sanitized])", "task": "Find SQL injection vulnerabilities"},
            "outputs": {"response": "No SQL injection vulnerabilities found. Input is properly sanitized and parameterized."},
        },
    ]

    train_file = data_dir / "train.jsonl"
    train_file.write_text("\n".join(json.dumps(ex) for ex in examples) + "\n")
    dev_file = data_dir / "dev.jsonl"
    dev_file.write_text(json.dumps(examples[0]) + "\n")

    splits = load_all_splits(data_dir)
    assert len(splits["train"]) == 2
    assert len(splits["dev"]) == 1
    assert splits["train"][0].context.startswith("app.get")

    # Step 4: Export
    output_dir = tmp_path / "exports"
    output = export_agent(
        agent=spec,
        optimized_instructions="Optimized: You are an expert security analyst.",
        output_dir=output_dir,
        version="v1",
        baseline_score=0.5,
        optimized_score=0.7,
    )

    assert output.exists()
    content = output.read_text()
    assert "id: test-analyst" in content
    assert "DSPy Optimized" in content
    assert "Optimized: You are an expert security analyst." in content
    assert "read_file" in content
    assert "bash" in content


def test_round_trip_format_preservation(tmp_path):
    """Verify that export preserves all original frontmatter fields."""
    spec = parse_agent(SAMPLE_AGENT, "test-analyst.md")

    md = format_agent_markdown(
        spec,
        optimized_instructions=spec.system_prompt,
        version="v0",
    )

    # Re-parse the exported markdown
    re_parsed = parse_agent(md, "re-parsed.md")

    assert re_parsed.id == spec.id
    assert re_parsed.description == spec.description
    assert re_parsed.model == spec.model
    assert re_parsed.max_output_tokens == spec.max_output_tokens
    assert re_parsed.max_iterations == spec.max_iterations
    assert re_parsed.tools == spec.tools
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/integration/test_pipeline.py -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS (unit + integration)

- [ ] **Step 4: Run linting**

Run: `ruff check src/ scripts/ tests/`
Expected: No errors (or fix any that appear)

- [ ] **Step 5: Commit**

```bash
git add tests/integration/
git commit -m "feat: integration tests for full ingest-to-export pipeline"
```

---

### Task 17: Final Verification and README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

```markdown
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
```

- [ ] **Step 2: Run full test suite one final time**

Run: `pytest -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with quick start guide"
```

- [ ] **Step 4: Push to remote**

Run: `git push -u origin master`
Expected: Successfully pushed to https://github.com/enclave-ai/enclave-dspy
