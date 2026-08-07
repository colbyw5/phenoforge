"""Fuse ranked concept lists from multiple retrieval components.

``reciprocal_rank_fusion`` is a pure function, no DB or network dependency.
``hybrid_search`` is the orchestration layer on top of it — deciding
whether to query a dense retriever at all and building the fallback
``unmappable`` entry — which belongs here rather than in the MCP transport
per AGENTS.md's thin-server rule (``mcp/`` is tool registration only, no
business logic).
"""

from __future__ import annotations

from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.models import ConceptWithProvenance, UnmappableTerm
from phenoforge.engine.retrieval import BM25Retriever

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


def hybrid_search(
    bm25: BM25Retriever,
    dense: DenseRetriever | None,
    query: str,
    k: int = 10,
) -> tuple[list[ConceptWithProvenance], UnmappableTerm | None]:
    """Search using BM25 and, if available, dense retrieval, fused by RRF.

    ``dense`` is optional so a caller (the MCP server) can pass ``None``
    when no dense index has been built yet, degrading to BM25-only rather
    than requiring one. Same return contract as
    :meth:`~phenoforge.engine.retrieval.BM25Retriever.search` and
    :meth:`~phenoforge.engine.dense.DenseRetriever.search`.

    :param bm25: A built BM25 retriever.
    :param dense: A built dense retriever, or ``None`` to search lexically only.
    :param query: Free-text search string.
    :param k: Maximum number of results to return.
    :returns: A tuple of (fused, deduplicated results ordered by combined
        relevance; an :class:`~phenoforge.engine.models.UnmappableTerm` if
        nothing matched, else ``None``).
    :rtype: tuple[list[ConceptWithProvenance], UnmappableTerm | None]
    """
    bm25_results, bm25_unmappable = bm25.search(query, k=k)
    if dense is not None:
        dense_results, _ = dense.search(query, k=k)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=k)
    else:
        fused = bm25_results

    if not fused:
        return [], bm25_unmappable or UnmappableTerm(term=query, reason="no match above threshold")
    return fused, None
