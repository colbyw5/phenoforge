"""Zero-setup demo: search + hierarchy expansion over the bundled demo vocabulary.

No Athena account, no API key, no model download required — builds an
ephemeral DuckDB from the small, redistributable ``demo/`` vocabulary (see
``demo/README.md`` for why that one's safe to commit and this project's real
vocabulary isn't) and runs the same ``hybrid_search``/``expand_descendants``
engine calls ``scripts/run_agent.py`` and the MCP server use.

Demonstrates the ``generated`` tier only (BM25 + hierarchy expansion; dense
is opt-in since it downloads BioLORD-2023 on first use). The bundled
vocabulary has no SNOMED rows, so there is nothing for
``find_curated_definition`` to resolve against — the ``curated`` tier and
the full agent need the real Setup in the root README.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import typer

from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.expansion import expand_descendants
from phenoforge.engine.hybrid import hybrid_search
from phenoforge.engine.models import ConceptWithProvenance, UnmappableTerm
from phenoforge.engine.retrieval import BM25Retriever
from scripts.load_vocab import VocabPaths, build_vocab_db

app = typer.Typer(add_completion=False)

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


def _print_concept_set(
    concepts: list[ConceptWithProvenance], unmappable: list[UnmappableTerm]
) -> None:
    """Print concepts and unmappable terms in the same style as run_agent.py/run_eval.py.

    :param concepts: Concepts to print, one per line with tier and source.
    :param unmappable: Unresolved terms to print with their reason.
    """
    for concept in concepts:
        typer.echo(
            f"  {concept.concept_code:<10}{concept.concept_name:<60}"
            f"[{concept.tier.value}] {concept.source}"
        )
    for item in unmappable:
        typer.echo(f"  (unmapped) {item.term}: {item.reason}")


@app.command()
def main(
    query: str = typer.Argument(
        "diabetic nephropathy",
        help="Free-text search string against the demo vocabulary's concept names.",
    ),
    expand_from: str = typer.Option(
        "E11",
        "--expand-from",
        help="ICD-10-CM seed code to expand descendants from.",
    ),
    dense: bool = typer.Option(
        False,
        "--dense",
        help="Also build a real dense index (downloads BioLORD-2023 on first use).",
    ),
) -> None:
    """Run hybrid search and hierarchy expansion against the bundled demo vocabulary."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "demo_vocab.duckdb"
        build_vocab_db(VocabPaths(athena_dir=_DEMO_DIR), db_path)

        con = duckdb.connect(str(db_path))
        try:
            bm25 = BM25Retriever(con)
            dense_retriever = DenseRetriever(con) if dense else None

            typer.echo(f'search_concepts("{query}"):')
            results, unmappable = hybrid_search(bm25, dense_retriever, query)
            _print_concept_set(results, [unmappable] if unmappable else [])

            typer.echo(f"\nexpand_hierarchy({expand_from!r}):")
            descendants = expand_descendants(con, expand_from)
            _print_concept_set(descendants, [])
        finally:
            con.close()

    typer.echo(
        "\nThis demo only shows the generated tier over 22 bundled ICD-10-CM codes. "
        "See the root README's Setup section for the real vocabulary, curated-tier "
        "matching, and the full agent."
    )


if __name__ == "__main__":
    app()
