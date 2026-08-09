"""Graph node functions for cohort assembly: decompose, curated-check, generate, confirm, assemble.

Every node is a plain function taking
:class:`~phenoforge.agent.state.CohortAssemblyState` and returning a partial
state update (a dict of the fields it changed), per LangGraph's node
contract. DB/retriever/LLM dependencies are bound at graph-build time via
``functools.partial`` (see :mod:`phenoforge.agent.graph`) rather than
threaded through state, since state must stay Pydantic-serializable for the
checkpointer — a ``duckdb.DuckDBPyConnection`` can't cross it.

The decomposition LLM call is injectable (``DecomposeFn``), mirroring
``engine/dense.py``'s ``EmbedFn`` pattern, so tests never make a real
Anthropic API call. Unlike ``dense.py``'s ``sentence_transformers``,
``langgraph`` itself is a hard import here (this module *is* graph node
code, always needed) — only ``langchain_anthropic`` is deferred into
:func:`default_decomposer`, since that's the piece a caller can avoid
entirely by supplying their own ``decompose_fn``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb
from langgraph.types import interrupt
from pydantic import BaseModel

from phenoforge.agent.state import CohortAssemblyState, PendingCandidate, SeedTermResult
from phenoforge.engine.curated import find_curated_definition
from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.hybrid import hybrid_search
from phenoforge.engine.models import ConceptSet, ConceptWithProvenance, UnmappableTerm
from phenoforge.engine.retrieval import BM25Retriever

DecomposeFn = Callable[[str], list[str]]

_DECOMPOSE_SYSTEM_PROMPT = (
    "You decompose a clinical population description into distinct seed clinical "
    "terms, one per condition/diagnosis mentioned, suitable for lookup against a "
    "medical vocabulary. Do not include demographic qualifiers (e.g. 'adults', "
    "'over 65') or study-design language (e.g. 'with', 'and') as terms. "
    'Example: "adults with type 2 diabetes and diabetic nephropathy" -> '
    '["type 2 diabetes", "diabetic nephropathy"].'
)


class DecomposedTerms(BaseModel):
    """Structured-output schema the decomposition LLM call populates.

    :ivar terms: Distinct seed clinical terms extracted from the population
        description, each independently searchable.
    """

    terms: list[str]


def default_decomposer(model: str = "claude-sonnet-5") -> DecomposeFn:
    """Build a decomposition function backed by a real Claude call.

    Imports ``langchain_anthropic`` on call, not at module import time, so
    ``import phenoforge.agent.nodes`` never requires the ``agent`` extra to
    be installed (mirrors
    :func:`~phenoforge.engine.dense.default_embedder`). Defaults to a
    quality-over-cost model since decomposition accuracy is the capability
    this version exists to add (see :mod:`phenoforge.agent`).

    :param model: Anthropic model id.
    :returns: A function mapping a population description to a list of seed
        clinical terms (see :data:`DecomposeFn`).
    :rtype: DecomposeFn
    :raises ImportError: If ``langchain-anthropic`` is not installed.
    """
    from langchain_anthropic import ChatAnthropic

    # temperature is deprecated for claude-sonnet-5 and newer — omitted
    # rather than pinned to a value that would raise a 400.
    llm = ChatAnthropic(model=model).with_structured_output(DecomposedTerms)

    def decompose(population_description: str) -> list[str]:
        result = llm.invoke(
            [
                ("system", _DECOMPOSE_SYSTEM_PROMPT),
                ("human", population_description),
            ]
        )
        return result.terms  # type: ignore[union-attr]

    return decompose


def decompose(state: CohortAssemblyState, *, decompose_fn: DecomposeFn) -> dict[str, object]:
    """Extract distinct seed clinical terms from the population description.

    :param state: Current graph state; reads ``population_description``.
    :param decompose_fn: Injected decomposition function (real or fake).
    :returns: Partial state update: ``{"seed_terms": [...]}``.
    :rtype: dict[str, object]
    """
    return {"seed_terms": decompose_fn(state.population_description)}


def check_curated(
    state: CohortAssemblyState, *, con: duckdb.DuckDBPyConnection, library_dir: Path
) -> dict[str, object]:
    """Check every seed term against the curated OHDSI Phenotype Library.

    Delegates to :func:`~phenoforge.engine.curated.find_curated_definition`
    per term. A term resolves (``SeedTermResult.resolved = True``) only if
    the curated lookup returns at least one concept; a curated miss (even
    one carrying its own explanatory
    :class:`~phenoforge.engine.models.UnmappableTerm`) leaves it unresolved
    for :func:`generate` to pick up.

    :param state: Current graph state; reads ``seed_terms``.
    :param con: Open connection to ``vocab.duckdb``.
    :param library_dir: Fetched phenotype library directory.
    :returns: Partial state update: ``{"term_results": [...]}``.
    :rtype: dict[str, object]
    """
    results = []
    for term in state.seed_terms:
        curated = find_curated_definition(con, term, library_dir)
        results.append(SeedTermResult(term=term, curated=curated, resolved=bool(curated.concepts)))
    return {"term_results": results}


def generate(
    state: CohortAssemblyState,
    *,
    bm25: BM25Retriever,
    dense: DenseRetriever | None,
    k: int = 10,
) -> dict[str, object]:
    """Generate candidate concepts (hybrid search) for every unresolved seed term.

    Only runs for terms :func:`check_curated` left unresolved. Every
    candidate is ``generated``-tier by construction — that is
    :func:`~phenoforge.engine.hybrid.hybrid_search`'s contract — and is
    staged into ``pending_candidates`` for human confirmation rather than
    folded directly into the result, since that's exactly the tier that
    requires it.

    :param state: Current graph state; reads ``term_results``.
    :param bm25: Built BM25 retriever.
    :param dense: Built dense retriever, or ``None`` to degrade to BM25-only.
    :param k: Candidates requested per term.
    :returns: Partial state update: ``{"pending_candidates": [...], "unmappable": [...]}``.
    :rtype: dict[str, object]
    """
    candidates: list[PendingCandidate] = []
    unmappable: list[UnmappableTerm] = []
    for result in state.term_results:
        if result.resolved:
            continue
        concepts, unmapped = hybrid_search(bm25, dense, result.term, k=k)
        if unmapped is not None:
            unmappable.append(unmapped)
        candidates.extend(PendingCandidate(term=result.term, concept=c) for c in concepts)
    return {"pending_candidates": candidates, "unmappable": unmappable}


def confirm(state: CohortAssemblyState) -> dict[str, object]:
    """Pause the graph for human review of pending generated candidates.

    Calls :func:`langgraph.types.interrupt` with the pending payload (see
    :mod:`phenoforge.agent.graph`'s module docstring for the exact
    payload/resume shape); the caller resumes with
    ``Command(resume=<list of accepted concept_codes>)``. A no-op if there
    is nothing pending — lets :func:`~phenoforge.agent.graph.build_graph`'s
    conditional edge route straight to ``assemble`` in the all-curated case
    without ever reaching this node, but staying defensive here too.

    :param state: Current graph state; reads ``pending_candidates``.
    :returns: Partial state update: ``{"confirmed_codes": [...]}``.
    :rtype: dict[str, object]
    """
    if not state.pending_candidates:
        return {"confirmed_codes": []}
    payload = {
        "pending": [
            {
                "term": c.term,
                "concept_code": c.concept.concept_code,
                "concept_name": c.concept.concept_name,
                "source": c.concept.source,
            }
            for c in state.pending_candidates
        ]
    }
    accepted_codes: list[str] = interrupt(payload)
    return {"confirmed_codes": accepted_codes}


def assemble(state: CohortAssemblyState) -> dict[str, object]:
    """Assemble the final ``ConceptSet`` from curated results and confirmed candidates.

    Curated concepts from every resolved term are included unconditionally,
    along with every curated per-term ``unmappable`` entry — a resolved
    term's curated match can itself carry unmappable SNOMED items (e.g.
    high-fanout-skipped), and those must not be silently dropped just
    because the term overall resolved. Generated candidates are included
    only if their ``concept_code`` is in ``confirmed_codes``; rejected
    candidates become :class:`~phenoforge.engine.models.UnmappableTerm`
    entries. Deduplicated by ``concept_id``.

    :param state: Current graph state; reads ``term_results``,
        ``pending_candidates``, ``confirmed_codes``, ``unmappable``.
    :returns: Partial state update: ``{"final_concept_set": ConceptSet(...)}``.
    :rtype: dict[str, object]
    """
    concepts: list[ConceptWithProvenance] = []
    unmappable = list(state.unmappable)
    for result in state.term_results:
        if result.curated is not None:
            concepts.extend(result.curated.concepts)
            unmappable.extend(result.curated.unmappable)

    confirmed = set(state.confirmed_codes)
    seen_ids = {c.concept_id for c in concepts}
    for candidate in state.pending_candidates:
        if candidate.concept.concept_code in confirmed:
            if candidate.concept.concept_id not in seen_ids:
                concepts.append(candidate.concept)
                seen_ids.add(candidate.concept.concept_id)
        else:
            unmappable.append(
                UnmappableTerm(term=candidate.term, reason="generated candidate rejected on review")
            )

    return {"final_concept_set": ConceptSet(concepts=concepts, unmappable=unmappable)}
