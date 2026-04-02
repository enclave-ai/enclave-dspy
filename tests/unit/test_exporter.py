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
    assert "  - list_dir" in md or "- list_dir" in md
    assert "  - read_file" in md or "- read_file" in md
    assert "  - bash" in md or "- bash" in md


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
