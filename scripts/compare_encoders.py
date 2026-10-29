"""Compare embedding models for dense/hybrid retrieval against curated ground truth.

``v0.6`` from the roadmap, deferred from the original plan (``engine/dense.py``'s
``EmbedFn`` was made injectable specifically so this comparison didn't need
to be designed in later). Builds a fresh LanceDB index per candidate
embedder (BioLORD-2023, the current default; SapBERT; MedCPT), scores
``"dense"`` and ``"hybrid"`` against the same 23 bundled OHDSI Phenotype
Library cohorts ``scripts/run_eval.py`` uses, and reports which embedder
actually scores best rather than assuming BioLORD is.

Each index build re-embeds every ICD-10-CM concept name for real (no
caching across embedders — different models produce different-dimension,
incomparable vectors), so this is slow (real CPU time, three full passes
over the vocabulary) but makes zero Claude API calls and costs nothing
beyond that compute time.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import typer
from pydantic import BaseModel

from phenoforge.engine.db import connect
from phenoforge.engine.dense import (
    DenseRetriever,
    EmbedFn,
    default_embedder,
    medcpt_embedders,
    sapbert_embedder,
)
from phenoforge.eval.harness import run_benchmark
from scripts.build_index import build_dense_index

app = typer.Typer(add_completion=False)


class EncoderResult(BaseModel):
    """One embedder's scores against the bundled benchmark.

    :ivar name: Short label for the embedder, e.g. ``"biolord"``.
    :ivar dense_coverage: Mean ``"dense"``-only coverage.
    :ivar dense_over_inclusion: Mean ``"dense"``-only over-inclusion penalty.
    :ivar dense_hierarchical: Mean ``"dense"``-only hierarchical score.
    :ivar hybrid_coverage: Mean ``"hybrid"`` coverage (BM25 + this embedder).
    :ivar hybrid_over_inclusion: Mean ``"hybrid"`` over-inclusion penalty.
    :ivar hybrid_hierarchical: Mean ``"hybrid"`` hierarchical score.
    """

    name: str
    dense_coverage: float
    dense_over_inclusion: float
    dense_hierarchical: float
    hybrid_coverage: float
    hybrid_over_inclusion: float
    hybrid_hierarchical: float


def compare_encoders(
    db_path: Path,
    library_dir: Path,
    cohort_ids: list[str] | None = None,
) -> list[EncoderResult]:
    """Build an index per candidate embedder and score ``dense``/``hybrid`` against ground truth.

    :param db_path: Path to a built ``vocab.duckdb``.
    :param library_dir: Path to fetched OHDSI Phenotype Library cohorts.
    :param cohort_ids: Cohort ids to score against; defaults to every id in
        ``library_dir``'s ``manifest.json``.
    :returns: One :class:`EncoderResult` per embedder, in the order built.
    :rtype: list[EncoderResult]
    """
    manifest = json.loads((library_dir / "manifest.json").read_text())
    resolved_cohort_ids = cohort_ids if cohort_ids is not None else list(manifest)

    con = connect(db_path)
    try:
        candidates: list[tuple[str, EmbedFn, EmbedFn | None]] = [
            ("biolord", default_embedder(), None),
            ("sapbert", sapbert_embedder(), None),
        ]
        medcpt_article, medcpt_query = medcpt_embedders()
        candidates.append(("medcpt", medcpt_article, medcpt_query))

        results: list[EncoderResult] = []
        for name, embed_fn, query_embed_fn in candidates:
            with tempfile.TemporaryDirectory() as tmp:
                index_path = Path(tmp) / f"{name}.lance"
                build_dense_index(con, index_path, embed_fn=embed_fn)
                dense = DenseRetriever(
                    con, embed_fn=embed_fn, index_path=index_path, query_embed_fn=query_embed_fn
                )
                benchmark = run_benchmark(
                    con,
                    library_dir,
                    cohort_ids=resolved_cohort_ids,
                    methods=["dense", "hybrid"],
                    dense=dense,
                )
            results.append(
                EncoderResult(
                    name=name,
                    dense_coverage=benchmark.mean_coverage_by_method["dense"],
                    dense_over_inclusion=benchmark.mean_over_inclusion_by_method["dense"],
                    dense_hierarchical=benchmark.mean_hierarchical_score_by_method["dense"],
                    hybrid_coverage=benchmark.mean_coverage_by_method["hybrid"],
                    hybrid_over_inclusion=benchmark.mean_over_inclusion_by_method["hybrid"],
                    hybrid_hierarchical=benchmark.mean_hierarchical_score_by_method["hybrid"],
                )
            )
    finally:
        con.close()

    return results


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
) -> None:
    """Compare BioLORD-2023, SapBERT, and MedCPT for dense/hybrid retrieval."""
    results = compare_encoders(db, library_dir)

    typer.echo(
        f"{'embedder':<10}"
        f"{'dense_cov':>11}{'dense_oi':>10}{'dense_hier':>12}"
        f"{'hybrid_cov':>12}{'hybrid_oi':>11}{'hybrid_hier':>13}"
    )
    for r in results:
        typer.echo(
            f"{r.name:<10}"
            f"{r.dense_coverage:>11.3f}{r.dense_over_inclusion:>10.3f}{r.dense_hierarchical:>12.3f}"
            f"{r.hybrid_coverage:>12.3f}{r.hybrid_over_inclusion:>11.3f}{r.hybrid_hierarchical:>13.3f}"
        )

    best = max(results, key=lambda r: r.hybrid_hierarchical)
    typer.echo(
        f"\nBest embedder by hybrid hierarchical score: {best.name} "
        f"({best.hybrid_hierarchical:.3f})"
    )


if __name__ == "__main__":
    app()
