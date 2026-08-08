"""Tests for phenoforge.eval.metrics. Pure functions, no DB fixture."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from phenoforge.eval.metrics import (
    distance_weighted_score,
    hierarchical_score,
    over_inclusion_penalty,
    set_coverage,
    tree_distance,
)


# Construct two chains sharing a known LCA at a controlled offset, so the
# resulting tree_distance is exactly (lift_a + lift_b) by construction —
# avoids reverse-engineering expected distance from arbitrary chains.
@st.composite
def _chain_pair_with_known_distance(draw: st.DrawFn) -> tuple[list[int], list[int], int]:
    lca_chain = draw(st.lists(st.integers(1000, 9999), min_size=1, max_size=5, unique=True))
    lift_a = draw(st.integers(0, 5))
    lift_b = draw(st.integers(0, 5))
    extra_a = draw(
        st.lists(st.integers(10000, 19999), min_size=lift_a, max_size=lift_a, unique=True)
    )
    extra_b = draw(
        st.lists(st.integers(20000, 29999), min_size=lift_b, max_size=lift_b, unique=True)
    )
    return extra_a + lca_chain, extra_b + lca_chain, lift_a + lift_b


@given(_chain_pair_with_known_distance())
def test_tree_distance_matches_construction(pair: tuple[list[int], list[int], int]) -> None:
    chain_a, chain_b, expected = pair
    assert tree_distance(chain_a, chain_b) == expected


def test_tree_distance_same_concept_is_zero() -> None:
    assert tree_distance([1, 2, 3], [1, 5, 6]) == 0


def test_tree_distance_no_common_ancestor_is_none() -> None:
    assert tree_distance([1, 2, 3], [4, 5, 6]) is None


@given(d1=st.integers(0, 20), d2=st.integers(0, 20))
def test_distance_weighted_score_monotonic_in_distance(d1: int, d2: int) -> None:
    """AGENTS.md-mandated invariant: score is monotonic (non-increasing) in tree distance."""
    if d1 <= d2:
        assert distance_weighted_score(d1) >= distance_weighted_score(d2)
    else:
        assert distance_weighted_score(d1) <= distance_weighted_score(d2)


@given(st.integers(0, 100))
def test_distance_weighted_score_bounded(d: int) -> None:
    score = distance_weighted_score(d)
    assert 0.0 <= score <= 1.0


def test_distance_weighted_score_exact_match_is_one() -> None:
    assert distance_weighted_score(0) == 1.0


def test_distance_weighted_score_none_is_zero() -> None:
    assert distance_weighted_score(None) == 0.0


def test_distance_weighted_score_floors_at_max_distance() -> None:
    assert distance_weighted_score(6, max_distance=6) == 0.0
    assert distance_weighted_score(100, max_distance=6) == 0.0


def test_set_coverage_mean() -> None:
    assert set_coverage([1.0, 0.5, 0.0], 3) == pytest.approx(0.5)


def test_set_coverage_empty_is_vacuously_one() -> None:
    assert set_coverage([], 0) == 1.0


def test_set_coverage_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="!="):
        set_coverage([1.0], 2)


def test_over_inclusion_penalty_mean() -> None:
    assert over_inclusion_penalty([1.0, 0.5, 0.0], 3) == pytest.approx(0.5)


def test_over_inclusion_penalty_empty_is_vacuously_zero() -> None:
    assert over_inclusion_penalty([], 0) == 0.0


def test_over_inclusion_penalty_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="!="):
        over_inclusion_penalty([1.0], 2)


def test_hierarchical_score_perfect() -> None:
    assert hierarchical_score(coverage=1.0, over_inclusion_penalty=0.0) == 1.0


def test_hierarchical_score_both_zero() -> None:
    assert hierarchical_score(coverage=0.0, over_inclusion_penalty=1.0) == 0.0


def test_hierarchical_score_harmonic_mean_punishes_imbalance() -> None:
    # Harmonic mean of (coverage=1.0, precision=0.0) should be 0.0, not 0.5
    # (which an arithmetic mean would give) - a method can't hide zero
    # precision behind perfect coverage.
    balanced = hierarchical_score(coverage=0.5, over_inclusion_penalty=0.5)
    imbalanced = hierarchical_score(coverage=1.0, over_inclusion_penalty=1.0)
    assert imbalanced == 0.0
    assert balanced > 0.0
