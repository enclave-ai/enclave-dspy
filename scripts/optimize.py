from pathlib import Path

import click

from src.config import load_config, configure_dspy
from src.data.loader import load_all_splits
from src.ingestion.parser import parse_agent
from src.modules.factory import create_module
from src.optimizers.runner import run_optimization
from src.config import get_enclave_agents_dir

DATASETS_DIR = Path("datasets")
OPTIMIZED_DIR = Path("optimized")


@click.command()
@click.option("--agent", type=str, default=None, help="Agent ID to optimize")
@click.option("--all-agents", "all_flag", is_flag=True, help="Optimize all agents")
@click.option("--optimizer", type=str, default=None, help="Specific optimizer to run")
@click.option("--max-bootstrapped-demos", type=int, default=None)
@click.option("--max-labeled-demos", type=int, default=None)
@click.option("--teacher-model", type=str, default=None)
@click.option("--target-model", type=str, default=None)
@click.option("--threshold", type=float, default=None)
def optimize(
    agent: str | None,
    all_flag: bool,
    optimizer: str | None,
    max_bootstrapped_demos: int | None,
    max_labeled_demos: int | None,
    teacher_model: str | None,
    target_model: str | None,
    threshold: float | None,
):
    """Run prompt optimization pipeline."""
    config = load_config()

    if max_bootstrapped_demos is not None:
        config.defaults.max_bootstrapped_demos = max_bootstrapped_demos
    if max_labeled_demos is not None:
        config.defaults.max_labeled_demos = max_labeled_demos
    if teacher_model:
        config.defaults.teacher_model = teacher_model
    if target_model:
        config.defaults.target_model = target_model
    if threshold is not None:
        config.defaults.improvement_threshold = threshold

    configure_dspy(config.defaults.target_model)
    agents_dir = get_enclave_agents_dir()

    if agent:
        _optimize_agent(agent, agents_dir, config, optimizer)
    elif all_flag:
        for agent_dir in sorted(DATASETS_DIR.iterdir()):
            if agent_dir.is_dir() and agent_dir.name != ".gitkeep":
                _optimize_agent(agent_dir.name, agents_dir, config, optimizer)
    else:
        click.echo("Specify --agent <name> or --all-agents")


def _optimize_agent(agent_id: str, agents_dir: Path, config, optimizer: str | None):
    agent_file = agents_dir / f"{agent_id}.md"
    if not agent_file.exists():
        click.echo(f"Skipping {agent_id}: agent file not found")
        return

    raw = agent_file.read_text()
    spec = parse_agent(raw, str(agent_file))
    module = create_module(spec)

    splits = load_all_splits(DATASETS_DIR / agent_id)
    trainset = splits.get("train", [])
    devset = splits.get("dev", [])

    if not trainset:
        click.echo(f"Skipping {agent_id}: no training data")
        return

    click.echo(f"\nOptimizing {agent_id} ({len(trainset)} train, {len(devset)} dev)...")
    result = run_optimization(
        program=module,
        trainset=trainset,
        devset=devset,
        agent_id=agent_id,
        config=config,
        optimizer_name=optimizer,
        output_dir=OPTIMIZED_DIR,
    )

    click.echo(
        f"  Best: {result.optimizer_name} | "
        f"baseline={result.baseline_score:.4f} -> "
        f"optimized={result.optimized_score:.4f} "
        f"({result.improvement:+.2%})"
    )
