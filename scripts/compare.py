from pathlib import Path

import click

OPTIMIZED_DIR = Path("optimized")


@click.command()
@click.option("--agent", required=True, help="Agent ID to compare")
@click.option("--versions", multiple=True, help="Specific versions to compare")
def compare(agent: str, versions: tuple[str]):
    """Compare optimization results."""
    report_file = OPTIMIZED_DIR / agent / "report.md"
    if not report_file.exists():
        click.echo(f"No optimization report found for {agent}")
        return

    content = report_file.read_text()
    click.echo(content)
