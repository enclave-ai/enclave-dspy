import click

from scripts.ingest import ingest
from scripts.data_cmd import data
from scripts.evaluate import evaluate
from scripts.optimize import optimize
from scripts.compare import compare
from scripts.export_cmd import export


@click.group()
def cli():
    """Enclave-DSPy: prompt optimization pipeline for Enclave security agents."""
    pass


cli.add_command(ingest)
cli.add_command(data)
cli.add_command(evaluate)
cli.add_command(optimize)
cli.add_command(compare)
cli.add_command(export)


if __name__ == "__main__":
    cli()
