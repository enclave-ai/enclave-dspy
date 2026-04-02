from __future__ import annotations

import dspy

from src.ingestion.models import AgentSpec, SkillSpec


def create_signature(agent: AgentSpec) -> type[dspy.Signature]:
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
