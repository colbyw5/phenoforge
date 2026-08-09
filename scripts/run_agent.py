"""Interactive CLI for the tiered-trust cohort-assembly agent.

Runs the graph built by ``phenoforge.agent.graph.build_graph`` against a
real population description, pausing at the terminal for human review
whenever ``generated``-tier candidates need confirmation (the
``confirm`` node's interrupt), then prints the final ``ConceptSet``. See
``phenoforge.agent`` for the tiered-trust flow this realizes.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import typer
from dotenv import load_dotenv
from langgraph.types import Command

from phenoforge.agent.graph import build_graph
from phenoforge.engine.db import connect
from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.models import ConceptSet

load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env, if present

app = typer.Typer(add_completion=False)


def _prompt_for_acceptance(pending: list[dict]) -> list[str]:
    """Print pending generated candidates and prompt for which to accept.

    :param pending: The ``confirm`` node's interrupt payload's ``"pending"``
        list — each item has ``term``, ``concept_code``, ``concept_name``,
        ``source``.
    :returns: ``concept_code`` values the human accepted.
    :rtype: list[str]
    """
    typer.echo("\nThe following generated (unverified) candidates need review:")
    for i, item in enumerate(pending, start=1):
        typer.echo(
            f"  [{i}] {item['concept_code']}  {item['concept_name']}"
            f"  (term: {item['term']!r}, source: {item['source']})"
        )
    raw = typer.prompt("Accept which? (comma-separated numbers, 'all', or 'none')", default="none")
    raw = raw.strip().lower()
    if raw == "none" or raw == "":
        return []
    if raw == "all":
        return [item["concept_code"] for item in pending]
    accepted = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token) - 1
        accepted.append(pending[index]["concept_code"])
    return accepted


def run_agent_interactive(
    population_description: str,
    db_path: Path,
    library_dir: Path,
    index_path: Path | None = None,
) -> ConceptSet:
    """Run the agent end-to-end, pausing at the terminal for human review.

    :param population_description: Free-text population description, e.g.
        "adults with type 2 diabetes and diabetic nephropathy".
    :param db_path: Path to a built ``vocab.duckdb``.
    :param library_dir: Path to fetched OHDSI Phenotype Library cohorts.
    :param index_path: Optional built LanceDB index. If given, a real
        :class:`~phenoforge.engine.dense.DenseRetriever` is built so
        ``generate`` runs hybrid search; otherwise it degrades to BM25-only,
        matching ``search_concepts``'s existing fallback.
    :returns: The assembled concept set.
    :rtype: ConceptSet
    """
    con = connect(db_path)
    try:
        dense = DenseRetriever(con, index_path=index_path) if index_path is not None else None
        graph = build_graph(con, library_dir, dense=dense)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = graph.invoke({"population_description": population_description}, config=config)
        while "__interrupt__" in result:
            pending = result["__interrupt__"][0].value["pending"]
            accepted_codes = _prompt_for_acceptance(pending)
            result = graph.invoke(Command(resume=accepted_codes), config=config)

        final: ConceptSet = result["final_concept_set"]
        return final
    finally:
        con.close()


@app.command()
def main(
    population_description: str = typer.Argument(
        ..., help='Free-text population description, e.g. "adults with type 2 diabetes".'
    ),
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
        help="Path to a built LanceDB index. If omitted, generated candidates come from BM25 only.",
    ),
) -> None:
    """Decompose a population description, resolve each term, and assemble a ConceptSet."""
    final = run_agent_interactive(population_description, db, library_dir, index_path=index)

    typer.echo("\nFinal concept set:")
    for concept in final.concepts:
        typer.echo(
            f"  {concept.concept_code:<10}{concept.concept_name:<60}"
            f"[{concept.tier.value}] {concept.source}"
        )
    if final.unmappable:
        typer.echo("\nUnmappable:")
        for item in final.unmappable:
            typer.echo(f"  {item.term}: {item.reason}")


if __name__ == "__main__":
    app()
