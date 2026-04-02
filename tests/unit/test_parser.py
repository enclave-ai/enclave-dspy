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
