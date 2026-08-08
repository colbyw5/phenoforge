"""Tests for scripts/run_eval.py. No network, no model download."""

from __future__ import annotations

from pathlib import Path

from scripts.run_eval import run_eval
from tests.scripts.conftest import MiniVocab


def test_run_eval_bm25_and_expand_descendants(mini_vocab: MiniVocab, library_dir: Path) -> None:
    report = run_eval(
        mini_vocab.db_path,
        library_dir,
        methods=["bm25", "expand_descendants"],
    )
    assert report.cohort_ids == ["1"]
    assert set(report.methods) == {"bm25", "expand_descendants"}
    assert set(report.benchmark.mean_coverage_by_method) == {"bm25", "expand_descendants"}


def test_run_eval_defaults_cohort_ids_to_manifest(mini_vocab: MiniVocab, library_dir: Path) -> None:
    report = run_eval(mini_vocab.db_path, library_dir, methods=["bm25"])
    assert report.cohort_ids == ["1"]


def test_run_eval_no_index_skips_dense(mini_vocab: MiniVocab, library_dir: Path) -> None:
    report = run_eval(mini_vocab.db_path, library_dir, methods=["dense"])
    assert report.benchmark.results == []
