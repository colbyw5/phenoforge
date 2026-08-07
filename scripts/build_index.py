"""Build a LanceDB semantic index over ICD-10-CM concept names.

Embeds every ICD-10-CM concept name (BioLORD-2023 by default) and writes a
persisted LanceDB table, so ``phenoforge.engine.dense.DenseRetriever`` can
open it without re-embedding at every process start.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import lancedb
import typer
from pydantic import BaseModel

from phenoforge.engine.db import connect
from phenoforge.engine.dense import _TABLE_NAME, EmbedFn, _embed_concept_rows, default_embedder

app = typer.Typer(add_completion=False)


class BuildIndexReport(BaseModel):
    """Summary of a completed index build.

    :ivar rows_indexed: Number of ICD-10-CM concepts embedded and indexed.
    :ivar index_path: Where the LanceDB table was written.
    """

    rows_indexed: int
    index_path: Path


def build_dense_index(
    con: duckdb.DuckDBPyConnection,
    index_path: Path,
    embed_fn: EmbedFn | None = None,
    batch_size: int = 64,
) -> BuildIndexReport:
    """Embed every ICD-10-CM concept name and write a LanceDB table.

    :param con: Open connection to ``vocab.duckdb``.
    :param index_path: Output directory for the LanceDB table. Overwritten
        if it already exists.
    :param embed_fn: Embedding function. Defaults to real BioLORD-2023 via
        :func:`~phenoforge.engine.dense.default_embedder`.
    :param batch_size: Number of concept names embedded per batch.
    :returns: Summary of what was indexed.
    :rtype: BuildIndexReport
    """
    embed_fn = embed_fn or default_embedder()
    records = _embed_concept_rows(con, embed_fn, batch_size)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(index_path))
    db.create_table(_TABLE_NAME, data=records, mode="overwrite")

    return BuildIndexReport(rows_indexed=len(records), index_path=index_path)


@app.command()
def main(
    db_path: Path = typer.Option(
        Path("data/vocab.duckdb"), "--db", help="Path to a built vocab.duckdb."
    ),
    output: Path = typer.Option(
        Path("data/concept_index.lance"),
        "--output",
        "-o",
        help="Output directory for the LanceDB index.",
    ),
) -> None:
    """Build data/concept_index.lance from data/vocab.duckdb.

    :param db_path: Path to a built ``vocab.duckdb``.
    :param output: Output directory for the LanceDB index.
    """
    con = connect(db_path)
    try:
        report = build_dense_index(con, output)
    finally:
        con.close()

    typer.echo(f"Built {report.index_path}")
    typer.echo(f"  rows indexed: {report.rows_indexed:,}")


if __name__ == "__main__":
    app()
