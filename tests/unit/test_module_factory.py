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
