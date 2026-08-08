"""Pydantic state schema for the cohort-assembly LangGraph.

Fields multiple nodes append to across steps (``seed_terms``, ``term_results``,
``pending_candidates``) use ``Annotated[..., operator.add]`` reducers.
Without one, LangGraph replaces a Pydantic state field wholesale on each
node's partial-update return rather than merging, so a later node's return
would silently clobber an earlier node's contribution instead of
accumulating. Each of these fields is single-writer in this graph (one node
returns the whole list in one shot) — the reducer costs nothing now and
keeps the door open for a later per-term fan-out (LangGraph's ``Send`` API)
without a state-shape change. ``confirmed_codes`` and ``final_concept_set``
are set exactly once each, by a single node, so they need no reducer.
"""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from phenoforge.engine.models import ConceptSet, ConceptWithProvenance, UnmappableTerm


class SeedTermResult(BaseModel):
    """One seed clinical term's curated-lookup outcome.

    :ivar term: The seed term as decomposed from the population description.
    :ivar curated: Result of checking this term against the curated OHDSI
        Phenotype Library — set by
        :func:`~phenoforge.agent.nodes.check_curated`.
    :ivar resolved: ``True`` only if ``curated.concepts`` is non-empty. A
        curated miss (even one carrying its own explanatory
        :class:`~phenoforge.engine.models.UnmappableTerm`) still leaves this
        ``False`` so the term falls through to
        :func:`~phenoforge.agent.nodes.generate`.
    """

    term: str
    curated: ConceptSet | None = None
    resolved: bool = False


class PendingCandidate(BaseModel):
    """One ``generated``-tier concept awaiting human confirmation.

    :ivar term: The seed term this candidate was generated for.
    :ivar concept: The candidate concept itself.
    """

    term: str
    concept: ConceptWithProvenance


class CohortAssemblyState(BaseModel):
    """Full state threaded through the cohort-assembly graph.

    :ivar population_description: The original free-text input, e.g.
        "adults with type 2 diabetes and diabetic nephropathy". Set once by
        the caller.
    :ivar seed_terms: Seed clinical terms extracted by
        :func:`~phenoforge.agent.nodes.decompose`.
    :ivar term_results: Per-term curated-lookup results, written by
        :func:`~phenoforge.agent.nodes.check_curated`.
    :ivar pending_candidates: ``generated``-tier concepts awaiting
        confirmation, staged by :func:`~phenoforge.agent.nodes.generate`.
    :ivar confirmed_codes: ``concept_code`` values the human accepted during
        :func:`~phenoforge.agent.nodes.confirm` — cross-referenced against
        ``pending_candidates`` by :func:`~phenoforge.agent.nodes.assemble`.
    :ivar unmappable: Terms/candidates that ended up unresolved anywhere in
        the pipeline, accumulated across nodes.
    :ivar final_concept_set: The assembled result, set once by
        :func:`~phenoforge.agent.nodes.assemble`.
    """

    population_description: str
    seed_terms: Annotated[list[str], operator.add] = Field(default_factory=list)
    term_results: Annotated[list[SeedTermResult], operator.add] = Field(default_factory=list)
    pending_candidates: Annotated[list[PendingCandidate], operator.add] = Field(
        default_factory=list
    )
    confirmed_codes: list[str] = Field(default_factory=list)
    unmappable: Annotated[list[UnmappableTerm], operator.add] = Field(default_factory=list)
    final_concept_set: ConceptSet | None = None
