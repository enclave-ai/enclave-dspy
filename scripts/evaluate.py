from pathlib import Path

import click

from src.config import configure_dspy, get_enclave_agents_dir, load_config
from src.data.loader import load_all_splits
from src.ingestion.parser import parse_agent
from src.modules.factory import create_module
from src.optimizers.runner import build_metric, evaluate_program

DATASETS_DIR = Path("datasets")
OPTIMIZED_DIR = Path("optimized")


@click.command()
@click.option("--agent", type=str, default=None, help="Agent ID to evaluate")
@click.option("--all-agents", "all_flag", is_flag=True, help="Evaluate all agents")
@click.option("--version", type=str, default=None, help="Evaluate a specific optimized version")
@click.option("--model", type=str, default=None, help="Model override")
def evaluate(agent: str | None, all_flag: bool, version: str | None, model: str | None):
    """Evaluate agent performance on dev set."""
    config = load_config()
    configure_dspy(model or config.defaults.target_model)

    agents_dir = get_enclave_agents_dir()

    if agent:
        _evaluate_agent(agent, agents_dir, config, version)
    elif all_flag:
        for agent_dir in sorted(DATASETS_DIR.iterdir()):
            if agent_dir.is_dir() and agent_dir.name != ".gitkeep":
                _evaluate_agent(agent_dir.name, agents_dir, config, version)
    else:
        click.echo("Specify --agent <name> or --all-agents")


def _evaluate_agent(agent_id: str, agents_dir: Path, config, version: str | None):
    agent_file = agents_dir / f"{agent_id}.md"
    if not agent_file.exists():
        click.echo(f"Skipping {agent_id}: agent file not found")
        return

    raw = agent_file.read_text()
    spec = parse_agent(raw, str(agent_file))
    module = create_module(spec)

    if version:
        program_path = OPTIMIZED_DIR / agent_id / f"{version}.json"
        if program_path.exists():
            module.load(str(program_path))
            click.echo(f"Loaded optimized version: {version}")
        else:
            click.echo(f"Version {version} not found at {program_path}")
            return

    splits = load_all_splits(DATASETS_DIR / agent_id)
    devset = splits.get("dev", [])
    if not devset:
        click.echo(f"Skipping {agent_id}: no dev set found")
        return

    metric = build_metric(agent_id, config)
    score = evaluate_program(module, devset, metric)
    label = f"{agent_id} ({version})" if version else f"{agent_id} (baseline)"
    click.echo(f"{label}: {score:.4f}")
