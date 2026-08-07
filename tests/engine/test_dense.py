"""Tests for phenoforge.engine.dense. Uses the fake embedder, no model download."""

from __future__ import annotations

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
