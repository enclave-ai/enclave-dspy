from pathlib import Path

import click

from src.config import get_enclave_agents_dir
from src.ingestion.parser import parse_agent
from src.export.exporter import export_agent

OPTIMIZED_DIR = Path("optimized")
EXPORTS_DIR = Path("exports")


@click.command()
@click.option("--agent", type=str, default=None, help="Agent ID to export")
@click.option("--all-agents", "all_flag", is_flag=True, help="Export all agents")
@click.option("--version", type=str, default="latest", help="Version to export")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
def export(agent: str | None, all_flag: bool, version: str, dry_run: bool):
    """Export optimized programs to Enclave Markdown format."""
    agents_dir = get_enclave_agents_dir()

    if agent:
        _export_agent(agent, agents_dir, version, dry_run)
    elif all_flag:
        for agent_dir in sorted(OPTIMIZED_DIR.iterdir()):
            if agent_dir.is_dir():
                _export_agent(agent_dir.name, agents_dir, version, dry_run)
    else:
        click.echo("Specify --agent <name> or --all-agents")


def _export_agent(agent_id: str, agents_dir: Path, version: str, dry_run: bool):
    agent_file = agents_dir / f"{agent_id}.md"
    if not agent_file.exists():
        click.echo(f"Skipping {agent_id}: agent file not found")
        return

    raw = agent_file.read_text()
    spec = parse_agent(raw, str(agent_file))

    optimized_dir = OPTIMIZED_DIR / agent_id
    if version == "latest":
        versions = sorted(optimized_dir.glob("v*.json"))
        if not versions:
            click.echo(f"Skipping {agent_id}: no optimized versions found")
            return
        version_file = versions[-1]
        version_str = version_file.stem
    else:
        version_file = optimized_dir / f"{version}.json"
        version_str = version
        if not version_file.exists():
            click.echo(f"Skipping {agent_id}: version {version} not found")
            return

    optimized_instructions = spec.system_prompt

    if dry_run:
        from src.export.exporter import format_agent_markdown
        md = format_agent_markdown(spec, optimized_instructions, version=version_str)
        click.echo(f"\n--- {agent_id} ({version_str}) ---")
        click.echo(md)
    else:
        output = export_agent(
            agent=spec,
            optimized_instructions=optimized_instructions,
            output_dir=EXPORTS_DIR,
            version=version_str,
        )
        click.echo(f"Exported {agent_id} -> {output}")
