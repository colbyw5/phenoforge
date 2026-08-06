"""Tests for phenoforge.engine.retrieval."""

from __future__ import annotations

import duckdb

from phenoforge.engine.models import ProvenanceTier
from phenoforge.engine.retrieval import BM25Retriever


def test_search_finds_lexical_match(con: duckdb.DuckDBPyConnection) -> None:
    retriever = BM25Retriever(con)
    results, unmappable = retriever.search("diabetic nephropathy")
    assert unmappable is None
    codes = [c.concept_code for c in results]
    assert "E11.21" in codes
    assert all(c.tier is ProvenanceTier.GENERATED for c in results)


def test_search_no_match_reports_unmappable(con: duckdb.DuckDBPyConnection) -> None:
    retriever = BM25Retriever(con)
    results, unmappable = retriever.search("xyzzy plugh quux")
    assert results == []
    assert unmappable is not None
    assert unmappable.term == "xyzzy plugh quux"


def test_search_respects_k(con: duckdb.DuckDBPyConnection) -> None:
    retriever = BM25Retriever(con)
    results, _ = retriever.search("condition", k=3)
    assert len(results) <= 3


def test_search_only_returns_icd10cm(con: duckdb.DuckDBPyConnection) -> None:
    retriever = BM25Retriever(con)
    results, _ = retriever.search("disorder")
    assert all(c.vocabulary_id == "ICD10CM" for c in results)
