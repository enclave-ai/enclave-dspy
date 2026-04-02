import json

from src.data.loader import load_all_splits
from src.export.exporter import export_agent, format_agent_markdown
from src.ingestion.parser import parse_agent
from src.modules.factory import create_module
from src.signatures.factory import create_signature

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
    """End-to-end test: parse agent -> create module -> load data -> export."""
    # Step 1: Parse agent
    spec = parse_agent(SAMPLE_AGENT, "test-analyst.md")
    assert spec.id == "test-analyst"
    assert spec.has_tools is True

    # Step 2: Create DSPy signature and module
    create_signature(spec)
    module = create_module(spec)
    assert module.agent_spec.id == "test-analyst"

    # Step 3: Create and load dataset
    data_dir = tmp_path / "datasets" / "test-analyst"
    data_dir.mkdir(parents=True)

    vuln_context = (
        "app.get('/users/:id', (req, res) => { "
        "db.query('SELECT * FROM users WHERE id=' + req.params.id) })"
    )
    safe_context = (
        "const sanitized = escape(input); "
        "db.query('SELECT * FROM users WHERE id=?', [sanitized])"
    )
    examples = [
        {
            "inputs": {"context": vuln_context, "task": "Find SQL injection vulnerabilities"},
            "outputs": {
                "response": (
                    "## Finding: SQL Injection\n\nSeverity: High\n\n"
                    "The endpoint concatenates user input directly into SQL query."
                )
            },
        },
        {
            "inputs": {"context": safe_context, "task": "Find SQL injection vulnerabilities"},
            "outputs": {
                "response": (
                    "No SQL injection vulnerabilities found. "
                    "Input is properly sanitized and parameterized."
                )
            },
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
