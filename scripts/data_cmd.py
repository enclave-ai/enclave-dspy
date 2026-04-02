from pathlib import Path

import click

from src.config import get_enclave_agents_dir, load_config
from src.data.loader import get_dataset_status
from src.data.synthetic import generate_synthetic_examples
from src.ingestion.parser import parse_agent

DATASETS_DIR = Path("datasets")


@click.group()
def data():
    """Manage evaluation datasets."""
    pass


@data.command()
@click.option("--agent", required=True, help="Agent ID to generate examples for")
@click.option("--count", default=20, help="Number of examples to generate")
@click.option("--teacher-model", default=None, help="Teacher model override")
def generate(agent: str, count: int, teacher_model: str | None):
    """Generate synthetic training examples."""
    agents_dir = get_enclave_agents_dir()
    agent_file = agents_dir / f"{agent}.md"
    if not agent_file.exists():
        click.echo(f"Error: agent file not found: {agent_file}", err=True)
        raise SystemExit(1)

    raw = agent_file.read_text()
    spec = parse_agent(raw, str(agent_file))

    config = load_config()
    model = teacher_model or config.defaults.teacher_model

    click.echo(f"Generating {count} synthetic examples for {agent} using {model}...")
    output_dir = DATASETS_DIR / agent
    examples = generate_synthetic_examples(spec, count, output_dir, model)
    click.echo(f"Generated {len(examples)} examples -> {output_dir}/synthetic.jsonl")


@data.command()
@click.option("--bucket", default="enclave-braintrust-logs", help="S3 bucket name")
@click.option("--prefix", default="logs/", help="S3 prefix")
@click.option("--profile", default="readonly", help="AWS profile")
@click.option("--local-dir", default="/tmp/bt-logs", help="Local cache directory")
@click.option("--agent", default=None, help="Extract specific agent (default: all)")
def pull(
    bucket: str,
    prefix: str,
    profile: str,
    local_dir: str,
    agent: str | None,
):
    """Pull production traces from Braintrust S3 logs and extract training data."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.data.braintrust_etl import extract_all_agents, sync_s3_logs

    logs_path = Path(local_dir)

    click.echo(f"Syncing s3://{bucket}/{prefix} → {logs_path}")
    sync_s3_logs(bucket, prefix, logs_path, profile)

    agents = [agent] if agent else None
    click.echo(f"\nExtracting {'agent: ' + agent if agent else 'all agents'}...")

    results = extract_all_agents(logs_path, DATASETS_DIR, agents)

    click.echo(f"\n{'Agent':<30} {'Train':>6} {'Dev':>6} {'Test':>6}")
    click.echo("-" * 54)
    for name, counts in sorted(results.items()):
        click.echo(
            f"{name:<30} {counts.get('train', 0):>6} "
            f"{counts.get('dev', 0):>6} {counts.get('test', 0):>6}"
        )


@data.command()
def status():
    """Show dataset counts for all agents."""
    if not DATASETS_DIR.exists():
        click.echo("No datasets directory found.")
        return

    click.echo(f"{'Agent':<30} {'Train':>6} {'Dev':>6} {'Test':>6}")
    click.echo("-" * 54)
    for agent_dir in sorted(DATASETS_DIR.iterdir()):
        if agent_dir.is_dir() and agent_dir.name != ".gitkeep":
            info = get_dataset_status(DATASETS_DIR, agent_dir.name)
            click.echo(
                f"{info.agent_id:<30} {info.train_count:>6} "
                f"{info.dev_count:>6} {info.test_count:>6}"
            )
