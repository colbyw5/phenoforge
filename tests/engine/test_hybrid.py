"""Tests for phenoforge.engine.hybrid.reciprocal_rank_fusion. Pure function, no DB."""

from __future__ import annotations

from phenoforge.engine.hybrid import reciprocal_rank_fusion
from phenoforge.engine.models import ConceptWithProvenance, ProvenanceTier


def _concept(concept_id: int, code: str, source: str) -> ConceptWithProvenance:
    return ConceptWithProvenance(
        concept_id=concept_id,
        concept_code=code,
        concept_name=f"Concept {code}",
        domain_id="Condition",
        vocabulary_id="ICD10CM",
        tier=ProvenanceTier.GENERATED,
        source=source,
    )


def test_fuses_disjoint_lists_by_rrf_score() -> None:
    """Both lists' rank-0 items (concepts 1 and 3) tie on RRF score — the
    stable sort resolves the tie by input order, so concept 1 (from the
    first list) sorts before concept 3. Concept 2, at rank 1 in bm25, scores
    lower than either rank-0 item and sorts last."""
    bm25 = [_concept(1, "A", "bm25:q"), _concept(2, "B", "bm25:q")]
    dense = [_concept(3, "C", "dense:q")]

    fused = reciprocal_rank_fusion([bm25, dense])

    assert [c.concept_id for c in fused] == [1, 3, 2]


def test_concept_in_multiple_lists_is_deduplicated_and_boosted() -> None:
    bm25 = [_concept(1, "A", "bm25:q"), _concept(2, "B", "bm25:q")]
    dense = [_concept(2, "B", "dense:q"), _concept(3, "C", "dense:q")]

    fused = reciprocal_rank_fusion([bm25, dense])

    ids = [c.concept_id for c in fused]
    assert ids.count(2) == 1
    assert ids[0] == 2


def test_deduplicated_concept_keeps_highest_ranked_source() -> None:
    bm25 = [_concept(1, "A", "bm25:q")]
    dense = [_concept(1, "A", "dense:q"), _concept(2, "B", "dense:q")]

    fused = reciprocal_rank_fusion([bm25, dense])

    concept_1 = next(c for c in fused if c.concept_id == 1)
    assert concept_1.source == "bm25:q"


def test_respects_k_truncation() -> None:
    bm25 = [_concept(i, str(i), "bm25:q") for i in range(20)]

    fused = reciprocal_rank_fusion([bm25], k=3)

    assert len(fused) == 3
    assert [c.concept_id for c in fused] == [0, 1, 2]


def test_empty_lists_return_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
