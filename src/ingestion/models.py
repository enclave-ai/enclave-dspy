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
