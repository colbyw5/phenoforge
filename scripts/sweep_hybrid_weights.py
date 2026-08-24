"""Sweep hybrid_search's dense_weight and report which value scores best.

``engine/hybrid.hybrid_search`` fuses BM25 and dense results by reciprocal
rank fusion at a fixed, equal weight (``dense_weight=0.5``). This script
answers "is 0.5 actually the best point, or just the untuned default" by
re-running the eval harness's ``"hybrid"`` method at a range of weights
against the curated OHDSI Phenotype Library cohorts as ground truth, the
same ground truth ``scripts/run_eval.py`` scores against.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import BaseModel

from phenoforge.engine.db import connect
from phenoforge.engine.dense import DenseRetriever
from phenoforge.eval.harness import run_benchmark

app = typer.Typer(add_completion=False)


class WeightPoint(BaseModel):
    """One sweep step's scores at a given dense weight.

    :ivar dense_weight: Dense's share of the fused RRF score, in ``[0, 1]``.
    :ivar coverage: Mean coverage across cohorts at this weight.
    :ivar over_inclusion: Mean over-inclusion penalty (lower is better).
    :ivar hierarchical: Mean hierarchical score (higher is better) — the
        harmonic mean of coverage and precision, and the metric this sweep
        optimizes for.
    """

    dense_weight: float
    coverage: float
    over_inclusion: float
    hierarchical: float


def sweep_hybrid_weights(
    db_path: Path,
    library_dir: Path,
    index_path: Path,
    cohort_ids: list[str] | None = None,
    steps: int = 11,
    k: int = 25,
) -> list[WeightPoint]:
    """Score the ``"hybrid"`` method at evenly spaced dense weights.

    :param db_path: Path to a built ``vocab.duckdb``.
    :param library_dir: Path to fetched phenotype library cohorts.
    :param index_path: Path to a built LanceDB index (required — a sweep
        with no dense index is just BM25 at every step).
    :param cohort_ids: Cohort ids to score against; defaults to every id in
        ``library_dir``'s ``manifest.json``.
    :param steps: Number of weight points from ``0.0`` to ``1.0`` inclusive,
        e.g. ``11`` gives ``0.0, 0.1, ..., 1.0``.
    :param k: Result count passed to ``hybrid_search`` at each step.
    :returns: One :class:`WeightPoint` per step, in ascending dense-weight order.
    :rtype: list[WeightPoint]
    """
    manifest = json.loads((library_dir / "manifest.json").read_text())
    resolved_cohort_ids = cohort_ids if cohort_ids is not None else list(manifest)

    con = connect(db_path)
    try:
        dense = DenseRetriever(con, index_path=index_path)
        points = []
        for i in range(steps):
            dense_weight = i / (steps - 1)
            benchmark = run_benchmark(
                con,
                library_dir,
                cohort_ids=resolved_cohort_ids,
                methods=["hybrid"],
                dense=dense,
                k=k,
                dense_weight=dense_weight,
            )
            points.append(
                WeightPoint(
                    dense_weight=dense_weight,
                    coverage=benchmark.mean_coverage_by_method["hybrid"],
                    over_inclusion=benchmark.mean_over_inclusion_by_method["hybrid"],
                    hierarchical=benchmark.mean_hierarchical_score_by_method["hybrid"],
                )
            )
    finally:
        con.close()

    return points


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
    index: Path = typer.Option(
        Path("data/concept_index.lance"), "--index", help="Path to a built LanceDB index."
    ),
    steps: int = typer.Option(
        11, "--steps", help="Number of dense-weight points from 0.0 to 1.0 inclusive."
    ),
) -> None:
    """Sweep hybrid_search's dense_weight and report the best-scoring value."""
    points = sweep_hybrid_weights(db, library_dir, index, steps=steps)

    typer.echo(f"{'dense_weight':>14}{'coverage':>12}{'over-inclusion':>18}{'hierarchical':>16}")
    for p in points:
        typer.echo(
            f"{p.dense_weight:>14.2f}{p.coverage:>12.3f}{p.over_inclusion:>18.3f}{p.hierarchical:>16.3f}"
        )

    best = max(points, key=lambda p: p.hierarchical)
    typer.echo(
        f"\nBest dense_weight by hierarchical score: {best.dense_weight:.2f} "
        f"(hierarchical={best.hierarchical:.3f}, coverage={best.coverage:.3f}, "
        f"over_inclusion={best.over_inclusion:.3f})"
    )


if __name__ == "__main__":
    app()
