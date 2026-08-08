"""End-to-end graph tests: no real LLM/network — ``fake_decompose_fn``/``dense=None`` throughout."""

from __future__ import annotations

import uuid
from pathlib import Path

import duckdb

from phenoforge.agent.graph import build_graph
from phenoforge.agent.nodes import DecomposeFn
from phenoforge.engine.models import ProvenanceTier
from tests.agent.conftest import RESOLVING_TERM, UNRESOLVING_TERM


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_all_curated_path_never_interrupts(con: duckdb.DuckDBPyConnection, library_dir: Path):
    def decompose_fn(population_description: str) -> list[str]:
        return [RESOLVING_TERM]

    graph = build_graph(con, library_dir, decompose_fn=decompose_fn, dense=None)
    result = graph.invoke(
        {"population_description": "adults with diabetic nephropathy"}, config=_config()
    )
    assert "__interrupt__" not in result
    final = result["final_concept_set"]
    assert final.concepts
    assert all(c.tier == ProvenanceTier.CURATED for c in final.concepts)


def test_interrupt_resume_confirms_and_rejects_candidates(
    con: duckdb.DuckDBPyConnection, library_dir: Path, fake_decompose_fn: DecomposeFn
):
    from langgraph.types import Command

    graph = build_graph(con, library_dir, decompose_fn=fake_decompose_fn, dense=None)
    config = _config()

    paused = graph.invoke(
        {"population_description": "adults with diabetic nephropathy and some other condition"},
        config=config,
    )
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    pending = payload["pending"]
    assert pending
    assert all(p["term"] == UNRESOLVING_TERM for p in pending)

    accepted_code = pending[0]["concept_code"]
    result = graph.invoke(Command(resume=[accepted_code]), config=config)

    final = result["final_concept_set"]
    codes = {c.concept_code for c in final.concepts}
    assert "E11.21" in codes  # the curated match for RESOLVING_TERM
    assert accepted_code in codes
    tiers = {c.concept_code: c.tier for c in final.concepts}
    assert tiers[accepted_code] == ProvenanceTier.GENERATED

    rejected_codes = {p["concept_code"] for p in pending} - {accepted_code}
    unmappable_terms = {u.term for u in final.unmappable}
    if rejected_codes:
        assert "generated candidate rejected on review" in {u.reason for u in final.unmappable}
    assert unmappable_terms  # at least the rejection reasons carried through


def test_generate_falls_back_to_bm25_without_error(
    con: duckdb.DuckDBPyConnection, library_dir: Path, fake_decompose_fn: DecomposeFn
):
    graph = build_graph(con, library_dir, decompose_fn=fake_decompose_fn, dense=None)
    result = graph.invoke(
        {"population_description": "adults with diabetic nephropathy and something unmapped"},
        config=_config(),
    )
    assert "__interrupt__" in result
