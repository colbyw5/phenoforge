"""Tests for phenoforge.engine.dense. Uses the fake embedder, no model download."""

from __future__ import annotations

from pathlib import Path

import duckdb

from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.models import ProvenanceTier
from tests.engine.fake_embedder import fake_embed_fn


def test_search_finds_lexically_overlapping_match(con: duckdb.DuckDBPyConnection) -> None:
    retriever = DenseRetriever(con, embed_fn=fake_embed_fn)
    results, unmappable = retriever.search("diabetic nephropathy")
    assert unmappable is None
    codes = [c.concept_code for c in results]
    assert "E11.21" in codes
    assert all(c.tier is ProvenanceTier.GENERATED for c in results)
    assert all(c.source == "dense:diabetic nephropathy" for c in results)


def test_search_no_overlap_reports_unmappable(con: duckdb.DuckDBPyConnection) -> None:
    retriever = DenseRetriever(con, embed_fn=fake_embed_fn)
    results, unmappable = retriever.search("xyzzy plugh quux", min_score=0.5)
    assert results == []
    assert unmappable is not None
    assert unmappable.term == "xyzzy plugh quux"


def test_search_respects_k(con: duckdb.DuckDBPyConnection) -> None:
    retriever = DenseRetriever(con, embed_fn=fake_embed_fn)
    results, _ = retriever.search("condition", k=3)
    assert len(results) <= 3


def test_search_only_returns_icd10cm(con: duckdb.DuckDBPyConnection) -> None:
    retriever = DenseRetriever(con, embed_fn=fake_embed_fn)
    results, _ = retriever.search("disorder")
    assert all(c.vocabulary_id == "ICD10CM" for c in results)


def test_search_normalizes_an_unnormalized_embedder(con: duckdb.DuckDBPyConnection) -> None:
    """A real regression: an embedder that returns large, unnormalized
    vectors (like raw sentence-transformers output without
    normalize_embeddings=True) must not silently break the similarity
    threshold and return nothing."""

    def unnormalized_embed_fn(texts: list[str]) -> list[list[float]]:
        return [[v * 100.0 for v in vec] for vec in fake_embed_fn(texts)]

    retriever = DenseRetriever(con, embed_fn=unnormalized_embed_fn)
    results, unmappable = retriever.search("diabetic nephropathy")
    assert unmappable is None
    codes = [c.concept_code for c in results]
    assert "E11.21" in codes


def test_query_embed_fn_used_for_search_not_indexing(con: duckdb.DuckDBPyConnection) -> None:
    """An asymmetric encoder pair (like MedCPT's article/query encoders)
    passes a separate query_embed_fn: indexing still uses embed_fn, but
    search must use query_embed_fn, not fall back to embed_fn. A
    query_embed_fn that returns an unrelated, fixed vector (disjoint from
    every real embed_fn vector) proves search didn't silently use embed_fn
    instead — if it had, "diabetic nephropathy" would still match E11.21.
    """

    def unrelated_query_embed_fn(texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 511 for _ in texts]

    retriever = DenseRetriever(con, embed_fn=fake_embed_fn, query_embed_fn=unrelated_query_embed_fn)
    results, _ = retriever.search("diabetic nephropathy", min_score=0.9)

    codes = [c.concept_code for c in results]
    assert "E11.21" not in codes


def test_query_embed_fn_defaults_to_embed_fn(con: duckdb.DuckDBPyConnection) -> None:
    """Without an explicit query_embed_fn, search must still work exactly
    as before (symmetric embedder, the BioLORD/SapBERT case) — a strict
    no-op for every existing caller."""
    retriever = DenseRetriever(con, embed_fn=fake_embed_fn)
    results, unmappable = retriever.search("diabetic nephropathy")
    assert unmappable is None
    assert "E11.21" in [c.concept_code for c in results]


def test_falls_back_to_in_memory_index_when_table_missing(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """An index_path directory that exists but has no valid table (e.g. an
    interrupted build_index.py run) must fall back gracefully, not raise."""
    empty_index_dir = tmp_path / "concept_index.lance"
    empty_index_dir.mkdir()

    retriever = DenseRetriever(con, embed_fn=fake_embed_fn, index_path=empty_index_dir)
    results, unmappable = retriever.search("diabetic nephropathy")

    assert unmappable is None
    codes = [c.concept_code for c in results]
    assert "E11.21" in codes
