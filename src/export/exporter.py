from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.ingestion.models import AgentSpec


def format_agent_markdown(
    agent: AgentSpec,
    optimized_instructions: str,
    demonstrations: str | None = None,
    version: str = "v1",
    baseline_score: float | None = None,
    optimized_score: float | None = None,
) -> str:
    frontmatter = {
        "id": agent.id,
        "description": agent.description,
        "model": agent.model,
        "max_output_tokens": agent.max_output_tokens,
        "max_iterations": agent.max_iterations,
        "tools": agent.tools,
    }
    if agent.recovery_iterations is not None:
        frontmatter["recovery_iterations"] = agent.recovery_iterations

    yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    score_info = ""
    if baseline_score is not None and optimized_score is not None:
        score_info = f" | baseline: {baseline_score:.4f} → optimized: {optimized_score:.4f}"

    parts = [
        "---",
        yaml_str.strip(),
        "---",
        "",
        f"<!-- DSPy Optimized | {version} | {timestamp}{score_info} -->",
        "",
        optimized_instructions,
    ]

    if demonstrations:
        parts.extend([
            "",
            "## Examples",
            "",
            demonstrations,
        ])

    return "\n".join(parts) + "\n"


def export_agent(
    agent: AgentSpec,
    optimized_instructions: str,
    demonstrations: str | None = None,
    output_dir: Path = Path("exports"),
    version: str = "v1",
    baseline_score: float | None = None,
    optimized_score: float | None = None,
) -> Path:
    if baseline_score is not None and optimized_score is not None:
        if optimized_score < baseline_score:
            raise ValueError(
                f"Export blocked: score regression for {agent.id} "
                f"(baseline={baseline_score:.4f}, optimized={optimized_score:.4f})"
            )

    md = format_agent_markdown(
        agent, optimized_instructions, demonstrations,
        version, baseline_score, optimized_score,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{agent.id}.md"
    output_file.write_text(md)
    return output_file
