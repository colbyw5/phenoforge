"""Tests for scripts/build_index.py, using the fake embedder (no model download)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import lancedb

from scripts.build_index import build_dense_index
from tests.engine.fake_embedder import fake_embed_fn


def test_build_dense_index_indexes_all_icd10cm_concepts(
    mini_vocab_con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    icd10cm_count = mini_vocab_con.execute(
        "SELECT COUNT(*) FROM concept WHERE vocabulary_id = 'ICD10CM'"
    ).fetchone()[0]

    index_path = tmp_path / "concept_index.lance"
    report = build_dense_index(mini_vocab_con, index_path, embed_fn=fake_embed_fn)

    assert report.rows_indexed == icd10cm_count
    assert report.index_path == index_path


def test_build_dense_index_table_is_queryable(
    mini_vocab_con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:

    index_path = tmp_path / "concept_index.lance"
    build_dense_index(mini_vocab_con, index_path, embed_fn=fake_embed_fn)

    db = lancedb.connect(str(index_path))
    table = db.open_table("concepts")
    query_vector = fake_embed_fn(["diabetic nephropathy"])[0]
    results = table.search(query_vector).limit(5).to_list()
    codes = [r["concept_code"] for r in results]
    assert "E11.21" in codes
