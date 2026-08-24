"""Tests for phenoforge.eval.harness."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from phenoforge.engine.dense import DenseRetriever
from phenoforge.eval.harness import match_scores, run_benchmark, score_cohort
from tests.engine.fake_embedder import fake_embed_fn
from tests.scripts.conftest import MiniVocab


def test_match_scores_exact_match_scores_one(con: duckdb.DuckDBPyConnection) -> None:
    scores = match_scores(con, ["E11.21"], ["E11.21"])
    assert scores == [1.0]


def test_match_scores_near_miss_scores_partial_credit(
    con: duckdb.DuckDBPyConnection,
) -> None:
    scores = match_scores(con, ["E11.21"], ["E11.2"])
    assert 0.0 < scores[0] < 1.0


def test_match_scores_empty_to_codes_scores_zero(con: duckdb.DuckDBPyConnection) -> None:
    assert match_scores(con, ["E11.21"], []) == [0.0]


def test_match_scores_picks_best_of_multiple_candidates(
    con: duckdb.DuckDBPyConnection,
) -> None:
    scores = match_scores(con, ["E11.21"], ["E11", "E11.21"])
    assert scores == [1.0]


def test_score_cohort_perfect_match(con: duckdb.DuckDBPyConnection) -> None:
    result = score_cohort(con, "1", "test cohort", ["E11.21"], ["E11.21"], "test")
    assert result.coverage == 1.0
    assert result.over_inclusion_penalty == 0.0
    assert result.hierarchical_score == 1.0


def test_score_cohort_no_predictions(con: duckdb.DuckDBPyConnection) -> None:
    result = score_cohort(con, "1", "test cohort", ["E11.21"], [], "test")
    assert result.coverage == 0.0
    assert result.over_inclusion_penalty == 0.0  # vacuous - nothing predicted to penalize
    assert result.hierarchical_score == 0.0


def test_run_benchmark_bm25_and_expand_descendants(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    report = run_benchmark(
        con, library_dir, cohort_ids=["1"], methods=["bm25", "expand_descendants"]
    )
    methods = {r.method for r in report.results}
    assert methods == {"bm25", "expand_descendants"}
    assert set(report.mean_coverage_by_method) == {"bm25", "expand_descendants"}
    bm25_result = next(r for r in report.results if r.method == "bm25")
    assert "E11.21" in bm25_result.predicted_codes


def test_run_benchmark_hybrid_with_injected_fake_dense(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    dense = DenseRetriever(con, embed_fn=fake_embed_fn)
    report = run_benchmark(
        con, library_dir, cohort_ids=["1"], methods=["hybrid", "dense"], dense=dense
    )
    methods = {r.method for r in report.results}
    assert methods == {"hybrid", "dense"}


def test_run_benchmark_skips_dense_dependent_methods_without_dense(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    report = run_benchmark(con, library_dir, cohort_ids=["1"], methods=["dense"], dense=None)
    assert report.results == []


def test_run_benchmark_dense_weight_reaches_hybrid_search(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    """dense_weight=0.0 on "hybrid" ranks every BM25 hit ahead of every
    dense-only hit (a zero-weighted list still pads the fused results with
    its own items at score 0 — see reciprocal_rank_fusion — so this checks
    ranking, not list identity). Proves the parameter actually reaches
    hybrid_search rather than being ignored."""
    dense = DenseRetriever(con, embed_fn=fake_embed_fn)
    bm25_only = run_benchmark(con, library_dir, cohort_ids=["1"], methods=["bm25"], dense=dense)
    bm25_codes = next(r for r in bm25_only.results if r.method == "bm25").predicted_codes

    dense_zero_weight = run_benchmark(
        con, library_dir, cohort_ids=["1"], methods=["hybrid"], dense=dense, dense_weight=0.0
    )
    hybrid_codes_at_zero = next(
        r for r in dense_zero_weight.results if r.method == "hybrid"
    ).predicted_codes
    assert hybrid_codes_at_zero[: len(bm25_codes)] == bm25_codes
    assert len(hybrid_codes_at_zero) > len(bm25_codes)  # padded by score-0 dense-only items


def test_run_benchmark_unknown_method_raises(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        run_benchmark(con, library_dir, cohort_ids=["1"], methods=["nonexistent"])
