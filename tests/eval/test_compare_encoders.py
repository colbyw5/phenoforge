"""Tests for scripts/compare_encoders.py. No network, no model download.

Monkeypatches the three embedder factories with the fake embedder so the
script's own orchestration (building one index per embedder, scoring
dense/hybrid, reporting per-embedder results) is exercised without ever
reaching HuggingFace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.compare_encoders as compare_encoders_module
from scripts.compare_encoders import compare_encoders
from tests.engine.fake_embedder import fake_embed_fn
from tests.scripts.conftest import MiniVocab


@pytest.fixture(autouse=True)
def _fake_embedders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compare_encoders_module, "default_embedder", lambda: fake_embed_fn)
    monkeypatch.setattr(compare_encoders_module, "sapbert_embedder", lambda: fake_embed_fn)
    monkeypatch.setattr(
        compare_encoders_module, "medcpt_embedders", lambda: (fake_embed_fn, fake_embed_fn)
    )


def test_compare_encoders_runs_all_three_and_scores_dense_and_hybrid(
    mini_vocab: MiniVocab, library_dir: Path
) -> None:
    results = compare_encoders(mini_vocab.db_path, library_dir)

    assert [r.name for r in results] == ["biolord", "sapbert", "medcpt"]
    for r in results:
        assert 0.0 <= r.dense_hierarchical <= 1.0
        assert 0.0 <= r.hybrid_hierarchical <= 1.0


def test_compare_encoders_defaults_cohort_ids_to_manifest(
    mini_vocab: MiniVocab, library_dir: Path
) -> None:
    results = compare_encoders(mini_vocab.db_path, library_dir, cohort_ids=["1"])
    assert len(results) == 3
