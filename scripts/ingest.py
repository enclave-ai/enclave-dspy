from pathlib import Path

import click

from src.config import get_enclave_agents_dir, get_enclave_skills_dir
from src.ingestion.parser import parse_agent, parse_agents_dir, parse_skills_dir


@click.command()
@click.option(
    "--enclave-path", type=click.Path(exists=True),
    help="Path to the enclave repo root",
)
@click.option("--agent", type=str, default=None, help="Ingest a specific agent by ID")
def ingest(enclave_path: str | None, agent: str | None):
    """Parse Enclave agent definitions into DSPy-compatible specs."""
    if enclave_path:
        defs = Path(enclave_path) / "packages" / "ai-client" / "src" / "definitions"
        agents_dir = defs / "agents"
        skills_dir = defs / "skills"
    else:
        agents_dir = get_enclave_agents_dir()
        skills_dir = get_enclave_skills_dir()

    if not agents_dir.exists():
        click.echo(f"Error: agents directory not found at {agents_dir}", err=True)
        raise SystemExit(1)

    if agent:
        agent_file = agents_dir / f"{agent}.md"
        if not agent_file.exists():
            click.echo(f"Error: agent file not found: {agent_file}", err=True)
            raise SystemExit(1)
        raw = agent_file.read_text()
        spec = parse_agent(raw, str(agent_file))
        click.echo(f"Parsed agent: {spec.id} ({spec.model}, {len(spec.tools)} tools)")
    else:
        agent_specs = parse_agents_dir(agents_dir)
        click.echo(f"Parsed {len(agent_specs)} agents:")
        for spec in agent_specs:
            click.echo(f"  - {spec.id} ({spec.model}, {len(spec.tools)} tools)")

        if skills_dir.exists():
            skill_specs = parse_skills_dir(skills_dir)
            click.echo(f"\nParsed {len(skill_specs)} skills:")
            for spec in skill_specs:
                click.echo(f"  - {spec.id}: {spec.description}")
