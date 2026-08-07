"""Command-line interface for the privacy-enhanced federated framework.

Subcommands
-----------
- ``pef train``      federated training (single dataset or ``--all``)
- ``pef ablation``   4-config component ablation
- ``pef extras``     supplementary experiments (scalability, privacy,
                     linkage, baselines)
- ``pef viz``        single-dataset figures
- ``pef viz-multi``  cross-dataset figures + summary
- ``pef pipeline``   train-all + extras + viz + viz-multi
- ``pef version``    print the package version

Every command accepts ``--results-dir`` to control where JSON / PDF / PNG
artefacts are written. The default is the path-relative ``results``
directory at the current working directory, so we recommend running
``pef`` from the repository root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pef import __version__

app = typer.Typer(
    name="pef",
    help="Privacy-Enhanced Federated Learning Framework — CLI.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

# Sub-app for the viz subcommands. Typer nests them as ``pef viz ...`` /
# ``pef viz-multi ...``.
viz_app = typer.Typer(help="Generate figures from experiment logs.")
app.add_typer(viz_app, name="viz")
app.add_typer(viz_app, name="viz-multi", help="Generate cross-dataset figures.")


def _ensure_results_dir(path: str) -> str:
    """Create the results directory if it does not exist and return it."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _print_summary_row(label: str, acc: float, verified: bool, rounds: int) -> None:
    console.print(
        f"  [bold]{label:16s}[/bold] | final acc=[green]{acc:.4f}[/green] | "
        f"rounds={rounds} | all-verified=[cyan]{verified}[/cyan]"
    )


# ---------------------------------------------------------------------------
# pef version
# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Print the installed pef version and exit."""
    console.print(f"pef [bold cyan]{__version__}[/bold cyan]")


# ---------------------------------------------------------------------------
# pef train
# ---------------------------------------------------------------------------
@app.command()
def train(
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="One of: cifar10, mnist, fashion_mnist. Ignored if --all is set.",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        help="Run on cifar10, mnist, fashion_mnist sequentially.",
    ),
    results_dir: str = typer.Option(
        "results",
        "--results-dir",
        help="Where to write JSON logs (created if missing).",
    ),
) -> None:
    """Run federated training on one or all supported datasets."""
    if not dataset and not all_datasets:
        typer.echo("error: provide --dataset NAME or --all", err=True)
        raise typer.Exit(code=2)

    from pef.federated_simulation import main as run_dataset

    _ensure_results_dir(results_dir)
    # federated_simulation writes relative to cwd; we chdir by overriding
    # the working directory at the top of the call stack.
    cwd = os.getcwd()
    try:
        if all_datasets:
            names = ["cifar10", "mnist", "fashion_mnist"]
        else:
            assert dataset is not None
            names = [dataset]

        console.rule("[bold]pef train[/bold]")
        summary = []
        for ds in names:
            console.print(f"\n[bold cyan]==> {ds}[/bold cyan]")
            log = run_dataset(ds)
            summary.append(
                (log["label"], log["accuracy"][-1], all(log["verification"]), len(log["accuracy"]))
            )

        if len(summary) > 1:
            console.rule("[bold]Cross-dataset summary[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Dataset")
            table.add_column("Final acc", justify="right")
            table.add_column("Rounds", justify="right")
            table.add_column("All verified", justify="center")
            for label, acc, verified, rounds in summary:
                table.add_row(label, f"{acc:.4f}", str(rounds), "✓" if verified else "✗")
            console.print(table)
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# pef ablation
# ---------------------------------------------------------------------------
@app.command()
def ablation(
    results_dir: str = typer.Option(
        "results", "--results-dir", help="Where to write ablation.json."
    ),
) -> None:
    """Run the 4-config component ablation (Shapley × Folding)."""
    _ensure_results_dir(results_dir)
    from pef.ablation import run_ablation

    console.rule("[bold]pef ablation[/bold]")
    run_ablation()


# ---------------------------------------------------------------------------
# pef extras
# ---------------------------------------------------------------------------
@app.command()
def extras(
    results_dir: str = typer.Option(
        "results", "--results-dir", help="Where to write supplementary JSON."
    ),
) -> None:
    """Run supplementary experiments: scalability, privacy-utility, linkage, baselines."""
    _ensure_results_dir(results_dir)
    from pef.experiments_extra import main as run_extras

    console.rule("[bold]pef extras[/bold]")
    run_extras()


# ---------------------------------------------------------------------------
# pef viz  /  pef viz-multi
# ---------------------------------------------------------------------------
@viz_app.command("single")
def viz_single(
    results_dir: str = typer.Option(
        "results", "--results-dir", help="Where to read experiment_log.json from."
    ),
) -> None:
    """Generate the 8 single-dataset figures from ``results/experiment_log.json``."""
    from pef.visualization import main as run_viz

    console.rule("[bold]pef viz single[/bold]")
    run_viz()


@viz_app.command("multi")
def viz_multi(
    results_dir: str = typer.Option(
        "results", "--results-dir", help="Where to read per-dataset JSON from."
    ),
) -> None:
    """Generate the cross-dataset figures + summary."""
    from pef.visualization_multi import main as run_viz_multi

    console.rule("[bold]pef viz multi[/bold]")
    run_viz_multi()


# Backwards-compatible flat subcommands (so users can type ``pef viz-multi``
# without the nested subcommand, since the README documents it that way).
@app.command("viz-multi", hidden=True)
def viz_multi_flat(
    results_dir: str = typer.Option(
        "results", "--results-dir", help="Where to read per-dataset JSON from."
    ),
) -> None:
    """Alias of ``pef viz multi`` for shell muscle-memory."""
    viz_multi(results_dir=results_dir)


# ---------------------------------------------------------------------------
# pef pipeline
# ---------------------------------------------------------------------------
@app.command()
def pipeline(
    results_dir: str = typer.Option(
        "results", "--results-dir", help="Where to write all artefacts."
    ),
) -> None:
    """Run the full pipeline: train-all + extras + viz + viz-multi.

    This is what the CI workflow runs.
    """
    _ensure_results_dir(results_dir)
    console.rule("[bold]pef pipeline[/bold]")
    console.print("[bold cyan]Step 1/4[/bold cyan]  train --all")
    train(all_datasets=True, results_dir=results_dir)
    console.print("\n[bold cyan]Step 2/4[/bold cyan]  extras")
    extras(results_dir=results_dir)
    console.print("\n[bold cyan]Step 3/4[/bold cyan]  viz")
    viz_single(results_dir=results_dir)
    console.print("\n[bold cyan]Step 4/4[/bold cyan]  viz multi")
    viz_multi(results_dir=results_dir)
    console.rule("[bold green]pipeline complete[/bold green]")


# ---------------------------------------------------------------------------
# pef — error if invoked as a module without -m
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the ``pef`` console script."""
    try:
        app()
    except SystemExit as e:
        # Don't dump a traceback for normal help / version exits.
        if e.code not in (0, None, 2):
            raise
        sys.exit(e.code)


if __name__ == "__main__":
    main()
