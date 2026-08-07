"""Fuse ranked concept lists from multiple retrieval components.

Pure function, no DB or network dependency — used to combine BM25
(retrieval.py) and dense (dense.py) results into a single ranked list.
"""

from __future__ import annotations

from phenoforge.engine.models import ConceptWithProvenance

_DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[ConceptWithProvenance]],
    k: int = 10,
    rrf_k: int = _DEFAULT_RRF_K,
) -> list[ConceptWithProvenance]:
    """Fuse ranked concept lists by reciprocal rank.

    Score for a concept is the sum of ``1 / (rrf_k + rank)`` across every
    input list it appears in (rank is 0-indexed position within that list).
    ``rrf_k=60`` is the standard default from the original RRF literature —
    it sidesteps needing to normalize scores across retrievers whose raw
    scores aren't comparable (BM25 term-frequency scores vs. cosine
    similarity), since RRF only uses rank, not magnitude.

    A concept appearing in multiple lists is deduplicated by ``concept_id``,
    keeping the :class:`~phenoforge.engine.models.ConceptWithProvenance`
    instance from whichever list ranked it highest — its ``source`` therefore
    names the specific retriever that surfaced it best (e.g. ``"dense:..."``
    or ``"bm25:..."``), never a synthesized "hybrid" label, so provenance
    stays traceable to a real mechanism.

    :param result_lists: Ranked concept lists to fuse, each already ordered
        best-first.
    :param k: Maximum number of fused results to return.
    :param rrf_k: RRF rank-damping constant.
    :returns: Top-``k`` concepts by fused score, descending.
    :rtype: list[ConceptWithProvenance]
    """
    scores: dict[int, float] = {}
    best_instance: dict[int, tuple[float, ConceptWithProvenance]] = {}

    for result_list in result_lists:
        for rank, concept in enumerate(result_list):
            contribution = 1.0 / (rrf_k + rank)
            scores[concept.concept_id] = scores.get(concept.concept_id, 0.0) + contribution

            existing = best_instance.get(concept.concept_id)
            if existing is None or contribution > existing[0]:
                best_instance[concept.concept_id] = (contribution, concept)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [best_instance[cid][1] for cid in ranked_ids[:k]]
