"""Tests for phenoforge.engine.explain."""

from __future__ import annotations

from pathlib import Path

import duckdb

from phenoforge.engine.explain import explain_inclusion
from phenoforge.engine.models import ProvenanceTier
from phenoforge.engine.retrieval import BM25Retriever
from tests.scripts.conftest import MiniVocab

#: mini_vocab's E11.21 has no direct "Is a" edge to E11 (only to E11.2, which
#: itself "Is a" E11) — unlike the real Athena download, which has a
#: redundant direct edge, this fixture's hierarchy is exactly two levels.
E11_TO_E11_21_DEPTH = 2


def test_explain_inclusion_curated_path(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    bm25 = BM25Retriever(con)

    result = explain_inclusion(con, "E11.21", bm25, library_dir, query="diabetic nephropathy")

    assert result.concept_code == "E11.21"
    assert result.concept_name == "Type 2 diabetes mellitus with diabetic nephropathy"
    curated_paths = [p for p in result.paths if p.tier == ProvenanceTier.CURATED]
    assert len(curated_paths) == 1
    assert curated_paths[0].source == "ohdsi_pl:1:Diabetic nephropathy demo cohort"
    assert "cohort 1" in curated_paths[0].evidence


def test_explain_inclusion_hierarchy_path(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    bm25 = BM25Retriever(con)

    result = explain_inclusion(con, "E11.21", bm25, library_dir, seed_code="E11")

    hierarchy_paths = [p for p in result.paths if p.source.startswith("hierarchy_expansion")]
    assert len(hierarchy_paths) == 1
    assert hierarchy_paths[0].source == "hierarchy_expansion:E11"
    assert f"{E11_TO_E11_21_DEPTH} level(s) below" in hierarchy_paths[0].evidence


def test_explain_inclusion_checks_both_paths_additively(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    """Both query and seed_code given, and both paths actually produce the
    code — both show up, not just the first one checked."""
    bm25 = BM25Retriever(con)

    result = explain_inclusion(
        con, "E11.21", bm25, library_dir, query="diabetic nephropathy", seed_code="E11"
    )

    tiers_and_sources = {(p.tier, p.source.split(":")[0]) for p in result.paths}
    assert (ProvenanceTier.CURATED, "ohdsi_pl") in tiers_and_sources
    assert (ProvenanceTier.GENERATED, "hierarchy_expansion") in tiers_and_sources


def test_explain_inclusion_no_match_returns_empty_paths(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    bm25 = BM25Retriever(con)

    result = explain_inclusion(con, "E11.21", bm25, library_dir, query="completely unrelated xyz")

    assert result.concept_name == "Type 2 diabetes mellitus with diabetic nephropathy"
    assert result.paths == []


def test_explain_inclusion_nonexistent_code_returns_no_name(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    bm25 = BM25Retriever(con)

    result = explain_inclusion(con, "Z999.999", bm25, library_dir, query="diabetic nephropathy")

    assert result.concept_name is None
    assert result.paths == []


def test_explain_inclusion_neither_query_nor_seed_returns_empty_paths(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    bm25 = BM25Retriever(con)

    result = explain_inclusion(con, "E11.21", bm25, library_dir)

    assert result.paths == []
