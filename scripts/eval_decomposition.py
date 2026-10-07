"""Run the decomposition eval against the bundled OHDSI Phenotype Library cohorts.

Makes a real Claude call per test case (``default_decomposer``) — needs
``ANTHROPIC_API_KEY`` and the ``agent`` extra. See
``phenoforge.eval.decomposition`` for what this measures and why.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from phenoforge.agent.nodes import default_decomposer
from phenoforge.engine.db import connect
from phenoforge.engine.dense import DenseRetriever
from phenoforge.eval.decomposition import score_decomposition

load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env, if present

app = typer.Typer(add_completion=False)


@app.command()
def main(
    db: Path = typer.Option(
        Path("data/vocab.duckdb"), "--db", help="Path to a built vocab.duckdb."
    ),
    library_dir: Path = typer.Option(
        Path("data/phenotype_library"),
        "--library-dir",
        help="Path to fetched OHDSI Phenotype Library cohorts.",
    ),
    index: Path | None = typer.Option(
        None,
        "--index",
        help="Path to a built LanceDB index. If omitted, retrieval is BM25-only.",
    ),
    model: str = typer.Option(
        "claude-sonnet-5", "--model", help="Anthropic model id for the decompose step."
    ),
) -> None:
    """Run the decomposition eval and print per-case scores plus the mean gap."""
    con = connect(db)
    try:
        dense = DenseRetriever(con, index_path=index) if index is not None else None
        decompose_fn = default_decomposer(model=model)
        report = score_decomposition(con, library_dir, decompose_fn, dense=dense)
    finally:
        con.close()

    typer.echo(f"Ran {len(report.results)} case(s)\n")
    for result in report.results:
        typer.echo(f"{result.cohort_name} ({result.cohort_id})")
        typer.echo(f"  description: {result.population_description!r}")
        typer.echo(f"  seed terms:  {result.seed_terms}")
        typer.echo(
            f"  baseline={result.baseline_hierarchical_score:.3f}  "
            f"decomposed={result.decomposed_hierarchical_score:.3f}  "
            f"gap={result.gap:+.3f}"
        )
        typer.echo("")

    typer.echo(f"Mean gap (baseline - decomposed): {report.mean_gap:+.3f}")


if __name__ == "__main__":
    app()
