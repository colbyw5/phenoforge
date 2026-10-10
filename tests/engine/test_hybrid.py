"""Tests for phenoforge.engine.hybrid: reciprocal_rank_fusion (pure) and hybrid_search."""

from __future__ import annotations

from phenoforge.engine.hybrid import hybrid_search, reciprocal_rank_fusion
from phenoforge.engine.models import ConceptWithProvenance, ProvenanceTier, UnmappableTerm


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


def test_zero_weight_excludes_a_list_in_practice() -> None:
    """A list weighted to 0 still contributes 0 score to every concept it
    holds, so an item found only in that list never outranks one found in a
    positively-weighted list — equivalent to dropping the zero-weighted list."""
    bm25 = [_concept(1, "A", "bm25:q")]
    dense = [_concept(2, "B", "dense:q")]

    fused = reciprocal_rank_fusion([bm25, dense], weights=[1.0, 0.0])

    assert [c.concept_id for c in fused] == [1, 2]
    only_bm25 = reciprocal_rank_fusion([bm25])
    assert [c.concept_id for c in fused] == [c.concept_id for c in only_bm25] + [2]


def test_uniform_weights_preserve_unweighted_order() -> None:
    """Scaling every list's contribution by the same constant never changes
    relative order — the default (weights=None) must match this exactly, so
    existing unweighted callers see no behavior change."""
    bm25 = [_concept(1, "A", "bm25:q"), _concept(2, "B", "bm25:q")]
    dense = [_concept(2, "B", "dense:q"), _concept(3, "C", "dense:q")]

    unweighted = reciprocal_rank_fusion([bm25, dense])
    weighted = reciprocal_rank_fusion([bm25, dense], weights=[0.5, 0.5])

    assert [c.concept_id for c in weighted] == [c.concept_id for c in unweighted]


def test_higher_weight_promotes_that_lists_top_result() -> None:
    """Concept 3 ranks last in dense (rank 1) but dense is weighted heavily
    enough to still outscore concept 2, which only bm25 ranks at all."""
    bm25 = [_concept(1, "A", "bm25:q"), _concept(2, "B", "bm25:q")]
    dense = [_concept(4, "D", "dense:q"), _concept(3, "C", "dense:q")]

    fused = reciprocal_rank_fusion([bm25, dense], weights=[0.1, 0.9])

    ids = [c.concept_id for c in fused]
    assert ids.index(3) < ids.index(2)


class _StubRetriever:
    """Duck-typed stand-in for BM25Retriever/DenseRetriever: returns a fixed
    ranked list, sliced to whatever ``k`` it's asked for, and records every
    ``k`` it was called with so tests can assert on it directly."""

    def __init__(self, items: list[ConceptWithProvenance], source_prefix: str) -> None:
        self._items = items
        self._source_prefix = source_prefix
        self.calls: list[int] = []

    def search(
        self, query: str, k: int = 10, min_score: float = 0.0
    ) -> tuple[list[ConceptWithProvenance], UnmappableTerm | None]:
        self.calls.append(k)
        return self._items[:k], None


def test_hybrid_search_default_candidate_k_couples_pool_to_output() -> None:
    """Undocumented-until-now original behavior, kept as the default for
    backward compatibility: with candidate_k omitted, both sub-retrievers
    are searched exactly k-deep — the same k controls both "how many
    results come back" and "how deep each retriever searches"."""
    bm25 = _StubRetriever([_concept(i, str(i), "bm25:q") for i in range(20)], "bm25")
    dense = _StubRetriever([_concept(i, str(i), "dense:q") for i in range(20)], "dense")

    hybrid_search(bm25, dense, "q", k=7)  # type: ignore[arg-type]

    assert bm25.calls == [7]
    assert dense.calls == [7]


def test_hybrid_search_explicit_candidate_k_decouples_pool_from_output() -> None:
    """Real case this was found from: the same concept's fused rank shifted
    depending on k alone, because widening k also silently widened the
    candidate pool. Concept 50 ranks only #15 in bm25 but #0 in dense — at
    a shallow k=5 pool it never enters bm25's considered set at all, so it
    scores on dense alone; once candidate_k is widened past 15, bm25's
    contribution kicks in too and its *combined* score changes. Output
    length still respects k, proving the two are now independent."""
    bm25_items = [_concept(i, str(i), "bm25:q") for i in range(15)] + [_concept(50, "50", "bm25:q")]
    dense_items = [_concept(50, "50", "dense:q")] + [
        _concept(i, str(i), "dense:q") for i in range(15)
    ]
    bm25 = _StubRetriever(bm25_items, "bm25")
    dense = _StubRetriever(dense_items, "dense")

    shallow, _ = hybrid_search(bm25, dense, "q", k=5)  # type: ignore[arg-type]
    assert bm25.calls == [5] and dense.calls == [5]
    shallow_score_source = next(c for c in shallow if c.concept_id == 50).source
    assert shallow_score_source == "dense:q"  # bm25 never saw it at this depth

    bm25.calls.clear()
    dense.calls.clear()
    deep, _ = hybrid_search(bm25, dense, "q", k=5, candidate_k=20)  # type: ignore[arg-type]
    assert bm25.calls == [20] and dense.calls == [20]
    assert len(deep) == 5  # output size still respects k, not candidate_k


def test_hybrid_search_bm25_only_respects_k_despite_deeper_candidate_pool() -> None:
    """dense=None with an explicit candidate_k > k must not leak extra
    results past k — the pre-fix code had no truncation here because
    bm25.search(k=k) and the final output size were always identical."""
    bm25 = _StubRetriever([_concept(i, str(i), "bm25:q") for i in range(20)], "bm25")

    fused, _ = hybrid_search(bm25, None, "q", k=5, candidate_k=20)  # type: ignore[arg-type]

    assert bm25.calls == [20]
    assert len(fused) == 5
