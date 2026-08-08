"""Benchmark harness: score generated retrieval against curated ground truth.

Orchestration layer with DB I/O — the counterpart to the pure functions in
:mod:`phenoforge.eval.metrics`. For each bundled OHDSI Phenotype Library
cohort, loads the curated concept set as ground truth (free — no
hand-annotation, per DESIGN.md's Evaluation section) and scores each
retrieval method's output against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from phenoforge.engine.curated import load_curated_concept_set
from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.expansion import ancestor_chain_ids, expand_descendants
from phenoforge.engine.hybrid import hybrid_search
from phenoforge.engine.retrieval import BM25Retriever
from phenoforge.eval.metrics import (
    distance_weighted_score,
    hierarchical_score,
    over_inclusion_penalty,
    set_coverage,
    tree_distance,
)
from phenoforge.eval.models import BenchmarkReport, CohortEvalResult

_DEFAULT_METHODS = ("bm25", "hybrid", "expand_descendants")


def match_scores(
    con: duckdb.DuckDBPyConnection, from_codes: list[str], to_codes: list[str]
) -> list[float]:
    """Best distance-weighted score from each of ``from_codes`` to any of ``to_codes``.

    For each code in ``from_codes``, computes
    :func:`~phenoforge.eval.metrics.tree_distance` (via
    :func:`~phenoforge.engine.expansion.ancestor_chain_ids`) against every
    code in ``to_codes`` and keeps the best (lowest-distance) match, then
    applies :func:`~phenoforge.eval.metrics.distance_weighted_score`.
    Ancestor chains are fetched once per unique code and memoized for the
    call's duration — cohorts run to the tens of codes, so simple O(n*m)
    with memoized lookups beats a precomputed global distance matrix at
    this scale.

    :param con: Open connection to ``vocab.duckdb``.
    :param from_codes: Codes to score. Caller decides direction: ground
        truth codes here (matched against predicted) computes coverage;
        predicted codes here (matched against ground truth) computes
        over-inclusion.
    :param to_codes: Codes to match against.
    :returns: One score per ``from_codes`` entry, same order. ``0.0`` per
        entry if ``to_codes`` is empty.
    :rtype: list[float]
    """
    if not to_codes:
        return [0.0] * len(from_codes)

    chain_cache: dict[str, list[int]] = {}

    def chain_for(code: str) -> list[int]:
        if code not in chain_cache:
            chain_cache[code] = ancestor_chain_ids(con, code)
        return chain_cache[code]

    to_chains = [chain_for(code) for code in to_codes]

    scores = []
    for code in from_codes:
        chain = chain_for(code)
        best_distance: int | None = None
        for other_chain in to_chains:
            distance = tree_distance(chain, other_chain)
            if distance is not None and (best_distance is None or distance < best_distance):
                best_distance = distance
        scores.append(distance_weighted_score(best_distance))
    return scores


def score_cohort(
    con: duckdb.DuckDBPyConnection,
    cohort_id: str,
    cohort_name: str,
    ground_truth_codes: list[str],
    predicted_codes: list[str],
    method: str,
) -> CohortEvalResult:
    """Score one method's predictions for one cohort against curated ground truth.

    Computes coverage (ground-truth -> predicted direction) and
    over-inclusion (predicted -> ground-truth direction) via two
    :func:`match_scores` calls, then
    :func:`~phenoforge.eval.metrics.set_coverage`,
    :func:`~phenoforge.eval.metrics.over_inclusion_penalty`, and
    :func:`~phenoforge.eval.metrics.hierarchical_score`.

    :param con: Open connection to ``vocab.duckdb``.
    :param cohort_id: OHDSI Phenotype Library cohort id.
    :param cohort_name: Cohort display name, echoed into the result.
    :param ground_truth_codes: ICD-10-CM codes from the curated concept set.
    :param predicted_codes: ICD-10-CM codes from the method under test.
    :param method: Label for the method under test, e.g. ``"hybrid"``.
    :returns: The scored result.
    :rtype: CohortEvalResult
    """
    coverage_scores = match_scores(con, ground_truth_codes, predicted_codes)
    over_inclusion_scores = match_scores(con, predicted_codes, ground_truth_codes)

    coverage = set_coverage(coverage_scores, len(ground_truth_codes))
    penalty = over_inclusion_penalty(over_inclusion_scores, len(predicted_codes))

    return CohortEvalResult(
        cohort_id=cohort_id,
        cohort_name=cohort_name,
        method=method,
        predicted_codes=predicted_codes,
        ground_truth_codes=ground_truth_codes,
        coverage=coverage,
        over_inclusion_penalty=penalty,
        hierarchical_score=hierarchical_score(coverage, penalty),
        unresolvable_ground_truth=[],
    )


def _predict(
    con: duckdb.DuckDBPyConnection,
    method: str,
    cohort_name: str,
    ground_truth_codes: list[str],
    bm25: BM25Retriever,
    dense: DenseRetriever | None,
    k: int,
) -> list[str] | None:
    """Generate predicted codes for one method, or ``None`` if the method is unavailable."""
    if method == "bm25":
        results, _ = bm25.search(cohort_name, k=k)
        return [c.concept_code for c in results]
    if method == "dense":
        if dense is None:
            return None
        results, _ = dense.search(cohort_name, k=k)
        return [c.concept_code for c in results]
    if method == "hybrid":
        results, _ = hybrid_search(bm25, dense, cohort_name, k=k)
        return [c.concept_code for c in results]
    if method == "expand_descendants":
        if not ground_truth_codes:
            return []
        seed = min(ground_truth_codes, key=len)
        return [c.concept_code for c in expand_descendants(con, seed)]
    raise ValueError(f"unknown method: {method}")


def run_benchmark(
    con: duckdb.DuckDBPyConnection,
    library_dir: Path,
    cohort_ids: list[str],
    methods: list[str] | None = None,
    dense: DenseRetriever | None = None,
    k: int = 25,
    fanout_threshold: int = 100,
) -> BenchmarkReport:
    """Run every requested method against every cohort, aggregate the results.

    Per cohort id: loads curated ground truth via
    :func:`~phenoforge.engine.curated.load_curated_concept_set`, generates
    predictions per requested method (``cohort_name`` — the cohort's own
    display name — is the query fed to ``"bm25"``/``"dense"``/``"hybrid"``;
    ``"expand_descendants"`` seeds from the shortest ground-truth code, a
    deliberately weak baseline rather than a competitive method — the
    least-specific curated code is the natural entry point a human would
    expand from), scores each with :func:`score_cohort`, and aggregates.

    :param con: Open connection to ``vocab.duckdb``.
    :param library_dir: Directory of fetched OHDSI Phenotype Library
        cohorts (``scripts/fetch_phenotype_library.py`` output).
    :param cohort_ids: Cohort ids to benchmark.
    :param methods: Methods to run; defaults to
        ``["bm25", "hybrid", "expand_descendants"]``. ``"dense"`` only runs
        if ``dense`` is given; a requested method needing ``dense`` when
        none is given is skipped for that cohort, not errored — matching
        ``search_concepts``'s existing graceful-degradation behavior.
    :param dense: Pre-built dense retriever, or ``None`` to skip
        dense-dependent methods. Tests inject one built with the fake
        embedder from ``tests/engine/fake_embedder.py`` so this function's
        own tests never download a real model.
    :param k: Result count passed to each retrieval method.
    :param fanout_threshold: Passed through to curated ground-truth loading.
    :returns: Full benchmark report across all cohorts and methods.
    :rtype: BenchmarkReport
    """
    methods = list(methods) if methods is not None else list(_DEFAULT_METHODS)
    manifest = json.loads((library_dir / "manifest.json").read_text())
    bm25 = BM25Retriever(con)

    results: list[CohortEvalResult] = []
    for cohort_id in cohort_ids:
        cohort_name = manifest.get(cohort_id, cohort_id)
        ground_truth = load_curated_concept_set(
            con, cohort_id, library_dir, fanout_threshold=fanout_threshold
        )
        ground_truth_codes = [c.concept_code for c in ground_truth.concepts]

        for method in methods:
            predicted_codes = _predict(con, method, cohort_name, ground_truth_codes, bm25, dense, k)
            if predicted_codes is None:
                continue
            results.append(
                score_cohort(
                    con, cohort_id, cohort_name, ground_truth_codes, predicted_codes, method
                )
            )

    return BenchmarkReport(
        results=results,
        mean_coverage_by_method=_mean_by_method(results, "coverage"),
        mean_over_inclusion_by_method=_mean_by_method(results, "over_inclusion_penalty"),
        mean_hierarchical_score_by_method=_mean_by_method(results, "hierarchical_score"),
    )


def _mean_by_method(results: list[CohortEvalResult], field: str) -> dict[str, float]:
    by_method: dict[str, list[float]] = {}
    for result in results:
        by_method.setdefault(result.method, []).append(getattr(result, field))
    return {method: sum(values) / len(values) for method, values in by_method.items()}
