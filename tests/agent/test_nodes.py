"""Unit tests for agent node functions.

``confirm`` is only tested here for its no-pending no-op path — calling it
with pending candidates outside a compiled graph raises, since
``langgraph.types.interrupt`` needs the LangGraph runtime (covered instead
by ``test_graph.py``'s interrupt/resume path).
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from phenoforge.agent.nodes import assemble, check_curated, confirm, decompose, generate
from phenoforge.agent.state import CohortAssemblyState, PendingCandidate, SeedTermResult
from phenoforge.engine.models import (
    ConceptSet,
    ConceptWithProvenance,
    ProvenanceTier,
    UnmappableTerm,
)
from phenoforge.engine.retrieval import BM25Retriever
from tests.agent.conftest import RESOLVING_TERM, UNRESOLVING_TERM
from tests.scripts.conftest import MiniVocab


def test_decompose_populates_seed_terms(fake_decompose_fn):
    state = CohortAssemblyState(population_description="adults with diabetic nephropathy")
    result = decompose(state, decompose_fn=fake_decompose_fn)
    assert result == {"seed_terms": [RESOLVING_TERM, UNRESOLVING_TERM]}


def test_check_curated_splits_resolved_and_unresolved(
    con: duckdb.DuckDBPyConnection, library_dir: Path
):
    state = CohortAssemblyState(
        population_description="irrelevant", seed_terms=[RESOLVING_TERM, UNRESOLVING_TERM]
    )
    result = check_curated(state, con=con, library_dir=library_dir)
    results = result["term_results"]
    assert len(results) == 2
    resolving, unresolving = results
    assert resolving.term == RESOLVING_TERM
    assert resolving.resolved is True
    assert resolving.curated is not None
    assert resolving.curated.concepts
    assert unresolving.term == UNRESOLVING_TERM
    assert unresolving.resolved is False


def test_generate_only_covers_unresolved_terms(con: duckdb.DuckDBPyConnection):
    bm25 = BM25Retriever(con)
    resolved_result = SeedTermResult(
        term=RESOLVING_TERM, curated=ConceptSet(concepts=[]), resolved=True
    )
    unresolved_result = SeedTermResult(term="nephropathy", curated=None, resolved=False)
    state = CohortAssemblyState(
        population_description="irrelevant", term_results=[resolved_result, unresolved_result]
    )
    result = generate(state, bm25=bm25, dense=None)
    candidates = result["pending_candidates"]
    assert all(c.term == "nephropathy" for c in candidates)
    assert candidates  # the mini vocab has ICD-10-CM concepts matching "nephropathy"


def test_generate_reports_unmappable_when_nothing_matches(con: duckdb.DuckDBPyConnection):
    bm25 = BM25Retriever(con)
    unresolved_result = SeedTermResult(
        term="zzz nonexistent gibberish", curated=None, resolved=False
    )
    state = CohortAssemblyState(
        population_description="irrelevant", term_results=[unresolved_result]
    )
    result = generate(state, bm25=bm25, dense=None)
    assert result["pending_candidates"] == []
    assert len(result["unmappable"]) == 1


def test_confirm_no_pending_is_a_noop():
    state = CohortAssemblyState(population_description="irrelevant", pending_candidates=[])
    assert confirm(state) == {"confirmed_codes": []}


def _concept(code: str, concept_id: int) -> ConceptWithProvenance:
    return ConceptWithProvenance(
        concept_id=concept_id,
        concept_code=code,
        concept_name=f"concept {code}",
        domain_id="Condition",
        vocabulary_id="ICD10CM",
        tier=ProvenanceTier.GENERATED,
        source="bm25:test",
    )


def test_assemble_includes_curated_and_confirmed_excludes_rejected(mini_vocab: MiniVocab):
    curated_concept = ConceptWithProvenance(
        concept_id=mini_vocab.e11_21_id,
        concept_code="E11.21",
        concept_name="Type 2 diabetes mellitus with diabetic nephropathy",
        domain_id="Condition",
        vocabulary_id="ICD10CM",
        tier=ProvenanceTier.CURATED,
        source="ohdsi_pl:1:test",
    )
    curated_unmappable = UnmappableTerm(term="90721000", reason="fan-out exceeds threshold")
    accepted = _concept("Z001", 2001)
    rejected = _concept("Z002", 2002)

    term_result = SeedTermResult(
        term=RESOLVING_TERM,
        curated=ConceptSet(concepts=[curated_concept], unmappable=[curated_unmappable]),
        resolved=True,
    )
    state = CohortAssemblyState(
        population_description="irrelevant",
        term_results=[term_result],
        pending_candidates=[
            PendingCandidate(term="other", concept=accepted),
            PendingCandidate(term="other", concept=rejected),
        ],
        confirmed_codes=["Z001"],
    )
    result = assemble(state)
    final = result["final_concept_set"]
    assert {c.concept_code for c in final.concepts} == {"E11.21", "Z001"}
    reasons = {u.term: u.reason for u in final.unmappable}
    assert reasons["90721000"] == "fan-out exceeds threshold"
    assert reasons["other"] == "generated candidate rejected on review"


def test_assemble_dedupes_by_concept_id(mini_vocab: MiniVocab):
    curated_concept = ConceptWithProvenance(
        concept_id=mini_vocab.e11_21_id,
        concept_code="E11.21",
        concept_name="Type 2 diabetes mellitus with diabetic nephropathy",
        domain_id="Condition",
        vocabulary_id="ICD10CM",
        tier=ProvenanceTier.CURATED,
        source="ohdsi_pl:1:test",
    )
    duplicate_candidate = ConceptWithProvenance(
        concept_id=mini_vocab.e11_21_id,
        concept_code="E11.21",
        concept_name="Type 2 diabetes mellitus with diabetic nephropathy",
        domain_id="Condition",
        vocabulary_id="ICD10CM",
        tier=ProvenanceTier.GENERATED,
        source="bm25:test",
    )
    term_result = SeedTermResult(
        term=RESOLVING_TERM, curated=ConceptSet(concepts=[curated_concept]), resolved=True
    )
    state = CohortAssemblyState(
        population_description="irrelevant",
        term_results=[term_result],
        pending_candidates=[PendingCandidate(term="other", concept=duplicate_candidate)],
        confirmed_codes=["E11.21"],
    )
    result = assemble(state)
    assert len(result["final_concept_set"].concepts) == 1
