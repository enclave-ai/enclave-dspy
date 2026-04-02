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
