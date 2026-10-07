"""Evaluate the agent's ``decompose`` step, isolated from retrieval quality.

``run_benchmark`` (``harness.py``) measures retrieval given an already-correct
search term. Nothing measures the step before that: given a free-text
population description, does ``decompose`` (a real Claude call) extract the
right seed terms? Before this module, that had been exercised exactly once,
by hand, in ``notebooks/agent_walkthrough.ipynb``.

Scoring term strings directly would need a new, somewhat arbitrary
similarity metric and a hand-labeled "correct terms" dataset — and would
measure something beside the point, since nobody cares about the terms
themselves, only the codes that come out the other end. Instead, this
reuses the bundled cohorts' existing ground truth and ``score_cohort``
unchanged: for each test case, compare "hybrid search using the cohort's
own name directly" (decomposition skipped entirely) against "the real
decompose step, then per-term curated/generated resolution" (mirroring
``agent/nodes.py``'s ``check_curated``/``generate``, minus the human
confirmation gate — this measures decomposition and retrieval together,
not human review). The gap between those two scores, against the *same*
ground truth, is what decomposition costs — or saves, if splitting a
description into multiple terms does better than one query.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
from pydantic import BaseModel

from phenoforge.agent.nodes import DecomposeFn
from phenoforge.engine.curated import find_curated_definition, load_curated_concept_set
from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.hybrid import hybrid_search
from phenoforge.engine.models import ConceptWithProvenance
from phenoforge.engine.retrieval import BM25Retriever
from phenoforge.eval.harness import score_cohort
from phenoforge.eval.models import DecompositionCaseResult, DecompositionReport


class DecompositionTestCase(BaseModel):
    """A free-text population description mapped to the cohort it should resolve to.

    :ivar population_description: Phrased like a real study population
        description, not copied from the cohort's own bundled display
        name — otherwise decomposition would be trivially unnecessary to
        get right.
    :ivar cohort_id: Bundled OHDSI Phenotype Library cohort id this
        description should resolve to; its curated concept set is ground
        truth.
    :ivar source: Where the phrasing pattern came from, for traceability —
        not used by scoring, just documentation.
    """

    population_description: str
    cohort_id: str
    source: str


#: Phrasing patterns drawn from real ClinicalTrials.gov study titles and
#: inclusion-criteria language for each bundled cohort's condition — not the
#: cohort's own bundled display name, and not invented from scratch, so
#: these test decomposition against how population descriptions are
#: actually phrased in practice, not against this project's own wording
#: conventions.
DEFAULT_TEST_CASES: list[DecompositionTestCase] = [
    DecompositionTestCase(
        population_description="adults with type 2 diabetes mellitus",
        cohort_id="503",
        source='ClinicalTrials.gov NCT03985293, "...in Adults With Type 2 Diabetes Mellitus"',
    ),
    DecompositionTestCase(
        population_description=(
            "patients with type 2 diabetes or a documented history of diabetes"
        ),
        cohort_id="40",
        source=(
            "common trial inclusion phrasing combining current diagnosis with history of diabetes"
        ),
    ),
    DecompositionTestCase(
        population_description=(
            "type 2 diabetes mellitus confirmed by diagnosis, treatment, or laboratory testing"
        ),
        cohort_id="288",
        source=(
            "phrasing pattern from diagnosis/treatment/lab-indexed diabetes ascertainment "
            "algorithms"
        ),
    ),
    DecompositionTestCase(
        population_description="adult patients with type 1 diabetes mellitus",
        cohort_id="499",
        source='ClinicalTrials.gov NCT01421147, "A Study in Adults With Type 1 Diabetes"',
    ),
    DecompositionTestCase(
        population_description=(
            "patients presenting to the emergency department with diabetic ketoacidosis"
        ),
        cohort_id="611",
        source="common DKA trial inclusion phrasing (ED presentation with primary DKA diagnosis)",
    ),
    DecompositionTestCase(
        population_description="pregnant women diagnosed with gestational diabetes mellitus",
        cohort_id="619",
        source="common GDM trial inclusion phrasing, e.g. ClinicalTrials.gov NCT02244814",
    ),
    DecompositionTestCase(
        population_description="adults with diabetic retinopathy",
        cohort_id="647",
        source="common diabetic-retinopathy trial inclusion phrasing",
    ),
    DecompositionTestCase(
        population_description="patients with chronic kidney disease",
        cohort_id="687",
        source="common CKD trial inclusion phrasing framed around a CKD diagnosis",
    ),
]


def _resolve_terms(
    con: duckdb.DuckDBPyConnection,
    terms: list[str],
    library_dir: Path,
    bm25: BM25Retriever,
    dense: DenseRetriever | None,
    fanout_threshold: int,
    k: int,
) -> list[str]:
    """Resolve each seed term via curated-first, generated-fallback — no confirmation gate.

    Mirrors ``agent/nodes.py``'s ``check_curated``/``generate`` logic
    without the ``confirm`` interrupt: this evaluates decomposition and
    retrieval together, auto-including every generated candidate, since
    there is no human in an eval loop.

    :returns: Deduplicated ICD-10-CM codes across every term, in
        first-seen order.
    :rtype: list[str]
    """
    codes: list[str] = []
    seen_ids: set[int] = set()

    def _extend(concepts: list[ConceptWithProvenance]) -> None:
        for concept in concepts:
            if concept.concept_id not in seen_ids:
                codes.append(concept.concept_code)
                seen_ids.add(concept.concept_id)

    for term in terms:
        curated = find_curated_definition(
            con, term, library_dir, bm25, dense, fanout_threshold=fanout_threshold, k=k
        )
        if curated.concepts:
            _extend(curated.concepts)
            continue
        generated, _ = hybrid_search(bm25, dense, term, k=k)
        _extend(generated)

    return codes


def score_decomposition(
    con: duckdb.DuckDBPyConnection,
    library_dir: Path,
    decompose_fn: DecomposeFn,
    dense: DenseRetriever | None = None,
    k: int = 25,
    fanout_threshold: int = 100,
    test_cases: list[DecompositionTestCase] | None = None,
) -> DecompositionReport:
    """Score ``decompose_fn`` against each test case's bundled cohort ground truth.

    :param con: Open connection to ``vocab.duckdb``.
    :param library_dir: Directory of fetched OHDSI Phenotype Library cohorts.
    :param decompose_fn: Decomposition function under test — pass
        :func:`~phenoforge.agent.nodes.default_decomposer`'s result for a
        real Claude call, or a fake for testing.
    :param dense: Pre-built dense retriever, or ``None`` to degrade to
        BM25-only, matching ``search_concepts``'s existing fallback.
    :param k: Result count passed to each retrieval call.
    :param fanout_threshold: Passed through to curated ground-truth loading
        and curated resolution.
    :param test_cases: Cases to run; defaults to :data:`DEFAULT_TEST_CASES`.
    :returns: Per-case scores and the mean gap.
    :rtype: DecompositionReport
    """
    cases = test_cases if test_cases is not None else DEFAULT_TEST_CASES
    manifest = json.loads((library_dir / "manifest.json").read_text())
    bm25 = BM25Retriever(con)

    results: list[DecompositionCaseResult] = []
    for case in cases:
        cohort_name = manifest.get(case.cohort_id, case.cohort_id)
        ground_truth = load_curated_concept_set(
            con, case.cohort_id, library_dir, fanout_threshold=fanout_threshold
        )
        ground_truth_codes = [c.concept_code for c in ground_truth.concepts]

        baseline_results, _ = hybrid_search(bm25, dense, cohort_name, k=k)
        baseline_codes = [c.concept_code for c in baseline_results]
        baseline_score = score_cohort(
            con, case.cohort_id, cohort_name, ground_truth_codes, baseline_codes, "baseline"
        ).hierarchical_score

        seed_terms = decompose_fn(case.population_description)
        decomposed_codes = _resolve_terms(
            con, seed_terms, library_dir, bm25, dense, fanout_threshold, k
        )
        decomposed_score = score_cohort(
            con, case.cohort_id, cohort_name, ground_truth_codes, decomposed_codes, "decomposed"
        ).hierarchical_score

        results.append(
            DecompositionCaseResult(
                cohort_id=case.cohort_id,
                cohort_name=cohort_name,
                population_description=case.population_description,
                seed_terms=seed_terms,
                baseline_hierarchical_score=baseline_score,
                decomposed_hierarchical_score=decomposed_score,
                gap=baseline_score - decomposed_score,
            )
        )

    mean_gap = sum(r.gap for r in results) / len(results) if results else 0.0
    return DecompositionReport(results=results, mean_gap=mean_gap)
