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
