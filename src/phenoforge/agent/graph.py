"""LangGraph wiring for tiered-trust cohort assembly.

Node functions in ``agent/nodes.py`` are pure-ish (state in, partial state
out); this module is the only place DB/retriever/LLM dependencies get bound
to them (via ``functools.partial``) and the only place graph topology lives.

Topology: ``decompose -> check_curated -> [generate -> confirm] -> assemble``,
with the bracketed segment skipped via conditional edges when every seed
term resolved via curated matching.

Interrupt/resume payload shape (the contract between
:func:`~phenoforge.agent.nodes.confirm` and any caller, e.g.
``scripts/run_agent.py``):

- Outbound (what ``graph.invoke(...)`` returns via
  ``result["__interrupt__"][0].value`` when paused):
  ``{"pending": [{"term": str, "concept_code": str, "concept_name": str,
  "source": str}, ...]}``.
- Inbound (what ``Command(resume=...)`` carries to resume): ``list[str]`` —
  the ``concept_code`` values the human accepted, a subset of the outbound
  payload's codes. Chosen over a per-concept accept/reject dict because the
  CLI's natural input (a comma-separated list of accepted codes, or
  "all"/"none") maps directly to it with no intermediate structure, and
  ``assemble`` only needs to know "accepted or not" per candidate.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import duckdb
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from phenoforge.agent.nodes import (
    DecomposeFn,
    assemble,
    check_curated,
    confirm,
    decompose,
    default_decomposer,
    generate,
)
from phenoforge.agent.state import CohortAssemblyState, PendingCandidate, SeedTermResult
from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.models import (
    Concept,
    ConceptSet,
    ConceptWithProvenance,
    ProvenanceTier,
    UnmappableTerm,
)
from phenoforge.engine.retrieval import BM25Retriever

# Our own Pydantic types that cross the checkpointer's msgpack boundary
# (state fields, and ConceptSet nested inside SeedTermResult) — explicitly
# allowlisted so InMemorySaver's default serializer doesn't warn/eventually
# block on them. Discovered by actually running the graph through a full
# interrupt/resume cycle, not assumed.
_CHECKPOINT_ALLOWED_TYPES = (
    Concept,
    ConceptSet,
    ConceptWithProvenance,
    ProvenanceTier,
    UnmappableTerm,
    CohortAssemblyState,
    SeedTermResult,
    PendingCandidate,
)


def _has_unresolved_terms(state: CohortAssemblyState) -> str:
    """Route to ``generate`` if any seed term missed the curated library, else ``assemble``.

    :param state: Current graph state; reads ``term_results``.
    :returns: ``"generate"`` or ``"assemble"``, an edge-routing key.
    :rtype: str
    """
    if any(not r.resolved for r in state.term_results):
        return "generate"
    return "assemble"


def _has_pending_candidates(state: CohortAssemblyState) -> str:
    """Route to ``confirm`` if :func:`~phenoforge.agent.nodes.generate` staged anything.

    :param state: Current graph state; reads ``pending_candidates``.
    :returns: ``"confirm"`` or ``"assemble"``, an edge-routing key.
    :rtype: str
    """
    if state.pending_candidates:
        return "confirm"
    return "assemble"


def build_graph(
    con: duckdb.DuckDBPyConnection,
    library_dir: Path,
    decompose_fn: DecomposeFn | None = None,
    dense: DenseRetriever | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph:
    """Build and compile the cohort-assembly graph.

    :param con: Open connection to ``vocab.duckdb``, bound into
        :func:`~phenoforge.agent.nodes.check_curated` and (via a fresh
        :class:`~phenoforge.engine.retrieval.BM25Retriever`)
        :func:`~phenoforge.agent.nodes.generate`.
    :param library_dir: Fetched phenotype library directory, bound into
        :func:`~phenoforge.agent.nodes.check_curated`.
    :param decompose_fn: Decomposition function; defaults to
        :func:`~phenoforge.agent.nodes.default_decomposer` (real Claude
        call). Tests inject a fake returning deterministic terms.
    :param dense: Pre-built dense retriever, or ``None`` to degrade
        ``generate`` to BM25-only, matching ``search_concepts``'s existing
        fallback behavior.
    :param checkpointer: Checkpointer backing ``interrupt``/
        ``Command(resume=...)``. Required for the ``confirm`` node's
        pause/resume to work at all — defaults to an in-memory
        :class:`~langgraph.checkpoint.memory.InMemorySaver` (with this
        project's Pydantic types allowlisted for its serializer — see
        :data:`_CHECKPOINT_ALLOWED_TYPES`) if not given, fine for a
        single-process CLI run but does not persist across process
        restarts.
    :returns: A compiled graph, ready for ``.invoke(state,
        config={"configurable": {"thread_id": ...}})``.
    :rtype: CompiledStateGraph
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES)
        )

    resolved_decompose_fn = decompose_fn or default_decomposer()
    bm25 = BM25Retriever(con)

    builder = StateGraph(CohortAssemblyState)
    builder.add_node("decompose", partial(decompose, decompose_fn=resolved_decompose_fn))
    builder.add_node("check_curated", partial(check_curated, con=con, library_dir=library_dir))
    builder.add_node("generate", partial(generate, bm25=bm25, dense=dense))
    builder.add_node("confirm", confirm)
    builder.add_node("assemble", assemble)

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "check_curated")
    builder.add_conditional_edges(
        "check_curated", _has_unresolved_terms, {"generate": "generate", "assemble": "assemble"}
    )
    builder.add_conditional_edges(
        "generate", _has_pending_candidates, {"confirm": "confirm", "assemble": "assemble"}
    )
    builder.add_edge("confirm", "assemble")
    builder.add_edge("assemble", END)

    return builder.compile(checkpointer=checkpointer)
