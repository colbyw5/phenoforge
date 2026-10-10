"""Explain why a concept code would be included in a result, via which path.

MCP tool calls are stateless — nothing persists a previously-returned
``ConceptSet`` server-side, so "why is this code in my set" has to be
answered by re-deriving it: re-running the same curated/generated/
hierarchy resolution a caller would have gotten, and reporting which of
those paths actually produce ``concept_code``, with evidence specific to
each path (a cohort citation, a search rank, a hierarchy distance). A code
can legitimately be reachable by more than one path at once — e.g. both
curated and hierarchy expansion — so every path checked is reported, not
just the first hit.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from pydantic import BaseModel, Field

from phenoforge.engine.curated import find_curated_definition
from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.expansion import ancestor_chain_ids, expand_descendants
from phenoforge.engine.hybrid import hybrid_search
from phenoforge.engine.lookup import lookup_by_code
from phenoforge.engine.models import ProvenanceTier
from phenoforge.engine.retrieval import BM25Retriever


class InclusionPath(BaseModel):
    """One way ``concept_code`` was found for the query/seed checked.

    :ivar tier: :class:`~phenoforge.engine.models.ProvenanceTier` this path
        produces.
    :ivar source: The same provenance string a real result would carry,
        e.g. ``"bm25:diabetic nephropathy"`` or
        ``"ohdsi_pl:503:Type 2 diabetes mellitus"``.
    :ivar evidence: Human-readable detail specific to this path — a
        cohort citation, a search rank, or a hierarchy distance.
    """

    tier: ProvenanceTier
    source: str
    evidence: str


class InclusionExplanation(BaseModel):
    """Every way ``concept_code`` was found, across every path checked.

    :ivar concept_code: The code that was explained.
    :ivar concept_name: Its vocabulary display name, or ``None`` if the
        code doesn't exist in the loaded ICD-10-CM vocabulary at all.
    :ivar paths: One entry per path that actually produced this code;
        empty if none did (including if the code doesn't exist).
    """

    concept_code: str
    concept_name: str | None = None
    paths: list[InclusionPath] = Field(default_factory=list)


def explain_inclusion(
    con: duckdb.DuckDBPyConnection,
    concept_code: str,
    bm25: BM25Retriever,
    library_dir: Path,
    dense: DenseRetriever | None = None,
    query: str | None = None,
    seed_code: str | None = None,
    fanout_threshold: int = 100,
    k: int = 25,
) -> InclusionExplanation:
    """Check every applicable path for whether it produces ``concept_code``.

    Checks are independent and additive, mirroring the two distinct ways a
    caller actually reaches a code through this project's tool surface —
    pass whichever parameters apply to what's being explained:

    - ``query`` given: checks :func:`~phenoforge.engine.curated.find_curated_definition`
      (curated) and :func:`~phenoforge.engine.hybrid.hybrid_search` (generated),
      the same two calls ``find_curated_definition``/``search_concepts``
      (the MCP tools) make.
    - ``seed_code`` given: checks :func:`~phenoforge.engine.expansion.expand_descendants`
      (generated, via hierarchy), the same call ``expand_hierarchy`` (the
      MCP tool) makes.

    :param con: Open connection to ``vocab.duckdb``.
    :param concept_code: ICD-10-CM code to explain.
    :param bm25: Shared BM25 index.
    :param library_dir: Directory of fetched OHDSI Phenotype Library cohorts.
    :param dense: Pre-built dense retriever, or ``None`` to degrade the
        generated-tier check to BM25-only, matching ``search_concepts``'s
        existing fallback.
    :param query: Free-text query to check curated/generated paths against.
    :param seed_code: ICD-10-CM seed code to check the hierarchy-expansion
        path against.
    :param fanout_threshold: Passed through to curated resolution.
    :param k: Result count passed to curated/generated retrieval calls.
    :returns: Every path that produced ``concept_code``; empty ``paths`` if
        none did, including if the code isn't in the vocabulary at all.
    :rtype: InclusionExplanation
    """
    concept = lookup_by_code(con, concept_code)
    paths: list[InclusionPath] = []

    if query is not None:
        curated = find_curated_definition(
            con, query, library_dir, bm25, dense, fanout_threshold=fanout_threshold, k=k
        )
        for c in curated.concepts:
            if c.concept_code == concept_code:
                _, cohort_id, cohort_name = c.source.split(":", 2)
                paths.append(
                    InclusionPath(
                        tier=ProvenanceTier.CURATED,
                        source=c.source,
                        evidence=(
                            f"from the bundled OHDSI Phenotype Library cohort "
                            f"{cohort_id} ({cohort_name})"
                        ),
                    )
                )

        generated, _ = hybrid_search(bm25, dense, query, k=k)
        for rank, c in enumerate(generated, start=1):
            if c.concept_code == concept_code:
                paths.append(
                    InclusionPath(
                        tier=ProvenanceTier.GENERATED,
                        source=c.source,
                        evidence=f"ranked #{rank} of {len(generated)} for query {query!r}",
                    )
                )

    if seed_code is not None:
        descendants = expand_descendants(con, seed_code)
        for c in descendants:
            if c.concept_code == concept_code:
                chain = ancestor_chain_ids(con, concept_code)
                seed = lookup_by_code(con, seed_code)
                depth = chain.index(seed.concept_id) if seed is not None else None
                depth_note = f"{depth} level(s) below" if depth is not None else "below"
                paths.append(
                    InclusionPath(
                        tier=ProvenanceTier.GENERATED,
                        source=c.source,
                        evidence=f"{depth_note} seed code {seed_code!r} in the hierarchy",
                    )
                )

    return InclusionExplanation(
        concept_code=concept_code,
        concept_name=concept.concept_name if concept is not None else None,
        paths=paths,
    )
