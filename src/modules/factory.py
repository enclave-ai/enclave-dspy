from __future__ import annotations

import dspy

from src.ingestion.models import AgentSpec
from src.signatures.factory import create_signature


class AgentModule(dspy.Module):
    def __init__(self, agent_spec: AgentSpec, signature: type[dspy.Signature]):
        super().__init__()
        self.agent_spec = agent_spec
        self.predict = dspy.ChainOfThought(signature)

    def forward(self, context: str, task: str) -> dspy.Prediction:
        return self.predict(context=context, task=task)


def create_module(agent: AgentSpec) -> AgentModule:
    signature = create_signature(agent)
    return AgentModule(agent_spec=agent, signature=signature)
