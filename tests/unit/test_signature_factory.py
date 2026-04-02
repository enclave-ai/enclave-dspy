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
