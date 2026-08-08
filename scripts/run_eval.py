"""Run the eval harness against the bundled OHDSI Phenotype Library cohorts.

Scores each retrieval method (BM25, dense, hybrid, hierarchy expansion)
against curated ground truth and prints a per-method summary table. See
``phenoforge.eval`` for the metrics and ``phenoforge.eval.harness`` for the
orchestration this wraps.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import BaseModel

from phenoforge.engine.db import connect
from phenoforge.engine.dense import DenseRetriever
from phenoforge.eval.harness import run_benchmark
from phenoforge.eval.models import BenchmarkReport

app = typer.Typer(add_completion=False)


class EvalReport(BaseModel):
    """CLI-facing summary: wraps the benchmark report with the run's config echoed back.

    :ivar benchmark: The full per-cohort, per-method scoring results.
    :ivar cohort_ids: Cohort ids that were run.
    :ivar methods: Methods that were run.
    """

    benchmark: BenchmarkReport
    cohort_ids: list[str]
    methods: list[str]


def run_eval(
    db_path: Path,
    library_dir: Path,
    cohort_ids: list[str] | None = None,
    methods: list[str] | None = None,
    index_path: Path | None = None,
) -> EvalReport:
    """Open connections, run the benchmark, return a report.

    :param db_path: Path to a built ``vocab.duckdb``.
    :param library_dir: Path to fetched phenotype library cohorts
        (``scripts/fetch_phenotype_library.py`` output).
    :param cohort_ids: Cohort ids to run; defaults to every id found in
        ``library_dir``'s ``manifest.json``.
    :param methods: Methods to run; see
        :func:`~phenoforge.eval.harness.run_benchmark`.
    :param index_path: Optional built LanceDB index. If given, a real
        :class:`~phenoforge.engine.dense.DenseRetriever` is built (real
        BioLORD-2023 load) so dense/hybrid run for real; otherwise they're
        skipped, matching ``search_concepts``'s existing fallback.
    :returns: The run's report.
    :rtype: EvalReport
    """
    manifest = json.loads((library_dir / "manifest.json").read_text())
    resolved_cohort_ids = cohort_ids if cohort_ids is not None else list(manifest)

    con = connect(db_path)
    try:
        dense = DenseRetriever(con, index_path=index_path) if index_path is not None else None
        benchmark = run_benchmark(
            con,
            library_dir,
            cohort_ids=resolved_cohort_ids,
            methods=methods,
            dense=dense,
        )
    finally:
        con.close()

    return EvalReport(
        benchmark=benchmark,
        cohort_ids=resolved_cohort_ids,
        methods=methods if methods is not None else list(benchmark.mean_coverage_by_method),
    )


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
        help="Path to a built LanceDB index. If omitted, dense/hybrid methods are skipped.",
    ),
) -> None:
    """Run the eval harness across the bundled demo cohorts and print a summary table."""
    report = run_eval(db, library_dir, index_path=index)

    typer.echo(f"Ran {len(report.cohort_ids)} cohort(s)")
    typer.echo(f"{'method':<20}{'coverage':>12}{'over-inclusion':>18}{'hierarchical':>16}")
    for method in sorted(report.benchmark.mean_coverage_by_method):
        coverage = report.benchmark.mean_coverage_by_method[method]
        over_inclusion = report.benchmark.mean_over_inclusion_by_method[method]
        hierarchical = report.benchmark.mean_hierarchical_score_by_method[method]
        typer.echo(f"{method:<20}{coverage:>12.3f}{over_inclusion:>18.3f}{hierarchical:>16.3f}")


if __name__ == "__main__":
    app()
