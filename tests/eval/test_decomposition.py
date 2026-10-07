"""Tests for phenoforge.eval.decomposition."""

from __future__ import annotations

from pathlib import Path

import duckdb

from phenoforge.eval.decomposition import DecompositionTestCase, score_decomposition
from tests.agent.conftest import RESOLVING_TERM
from tests.scripts.conftest import MiniVocab

#: Deliberately shares no token with mini_vocab's ~190 filler concepts, each
#: literally named "Filler condition {i}" — tests/agent/conftest.py's own
#: UNRESOLVING_TERM ("made up unrelated condition xyz") shares "condition"
#: with every one of them, which BM25 matches on and returns as noise. That
#: fixture was designed to test curated resolved/unresolved, not probed
#: against raw hybrid_search output content, so it's the wrong fit here.
GENUINELY_UNMATCHED_TERM = "zzqqxxnonexistentgibberish"


def test_score_decomposition_resolves_curated_term_perfectly(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    """The fake decomposer's RESOLVING_TERM curated-matches cohort "1" exactly
    (ground truth is E11.21 alone), so the decomposed path should score a
    perfect 1.0 regardless of how well the baseline (cohort-name-as-query)
    direct search happens to do."""

    def fake_decompose(population_description: str) -> list[str]:
        return [RESOLVING_TERM, GENUINELY_UNMATCHED_TERM]

    cases = [
        DecompositionTestCase(
            population_description="adults with type 2 diabetes and diabetic nephropathy",
            cohort_id="1",
            source="test",
        )
    ]

    report = score_decomposition(con, library_dir, fake_decompose, test_cases=cases)

    assert len(report.results) == 1
    result = report.results[0]
    assert result.seed_terms == [RESOLVING_TERM, GENUINELY_UNMATCHED_TERM]
    assert result.decomposed_hierarchical_score == 1.0
    assert result.gap == result.baseline_hierarchical_score - result.decomposed_hierarchical_score
    assert report.mean_gap == result.gap


def test_score_decomposition_empty_cases_returns_empty_report(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> None:
    report = score_decomposition(con, library_dir, lambda _: [], test_cases=[])

    assert report.results == []
    assert report.mean_gap == 0.0
