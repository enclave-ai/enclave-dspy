from __future__ import annotations

from pathlib import Path

import frontmatter

from src.ingestion.models import AgentSpec, SkillSpec

AGENT_REQUIRED_FIELDS = ["id", "description", "model", "max_output_tokens", "max_iterations"]
SKILL_REQUIRED_FIELDS = ["id", "description"]


def parse_agent(raw: str, file_path: str) -> AgentSpec:
    post = frontmatter.loads(raw)
    meta = post.metadata

    for field in AGENT_REQUIRED_FIELDS:
        if field not in meta or meta[field] is None:
            raise ValueError(f"{file_path}: missing required field '{field}'")

    tools_raw = meta.get("tools")
    if tools_raw is None or (isinstance(tools_raw, list) and len(tools_raw) == 0):
        tools = []
    elif isinstance(tools_raw, list):
        tools = [str(t) for t in tools_raw]
    else:
        tools = []

    return AgentSpec(
        id=meta["id"],
        description=meta["description"],
        model=meta["model"],
        max_output_tokens=meta["max_output_tokens"],
        max_iterations=meta["max_iterations"],
        tools=tools,
        system_prompt=post.content.strip(),
        recovery_iterations=meta.get("recovery_iterations"),
    )


def parse_skill(raw: str, file_path: str) -> SkillSpec:
    post = frontmatter.loads(raw)
    meta = post.metadata

    for field in SKILL_REQUIRED_FIELDS:
        if field not in meta or meta[field] is None:
            raise ValueError(f"{file_path}: missing required field '{field}'")

    return SkillSpec(
        id=meta["id"],
        description=meta["description"],
        category=meta.get("category"),
        content=post.content.strip(),
    )


def parse_agents_dir(directory: Path) -> list[AgentSpec]:
    specs = []
    for md_file in sorted(directory.rglob("*.md")):
        raw = md_file.read_text()
        specs.append(parse_agent(raw, str(md_file)))
    return specs


def parse_skills_dir(directory: Path) -> list[SkillSpec]:
    specs = []
    for md_file in sorted(directory.rglob("*.md")):
        raw = md_file.read_text()
        specs.append(parse_skill(raw, str(md_file)))
    return specs
