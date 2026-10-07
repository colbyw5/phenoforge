"""Pure result types for the eval harness. No DB dependency.

Gives ``metrics.py`` and ``harness.py`` a shared vocabulary of typed results
without either depending on the other's I/O — ``metrics.py`` stays a pure,
DB-free module that property-based tests can target cheaply.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CohortEvalResult(BaseModel):
    """Per-cohort scoring result for one retrieval method against curated ground truth.

    :ivar cohort_id: OHDSI Phenotype Library cohort id.
    :ivar cohort_name: Cohort display name — also the query fed to
        retrieval methods that generated ``predicted_codes``.
    :ivar method: Label for the method under test, e.g. ``"bm25"``,
        ``"dense"``, ``"hybrid"``, ``"expand_descendants"``.
    :ivar predicted_codes: ICD-10-CM codes the method produced.
    :ivar ground_truth_codes: ICD-10-CM codes from the curated cohort.
    :ivar coverage: Ground-truth -> predicted direction score, ``[0, 1]`` —
        did the whole curated set get assembled.
    :ivar over_inclusion_penalty: Predicted -> ground-truth direction
        score, ``[0, 1]`` — how much of what was predicted has no good
        match anywhere in the curated set.
    :ivar hierarchical_score: Harmonic mean of ``coverage`` and
        ``1 - over_inclusion_penalty``; a single combined quality number.
    :ivar unresolvable_ground_truth: Curated codes not found in the loaded
        vocabulary (a data-quality escape hatch, not expected to be
        nonempty in normal operation).
    """

    cohort_id: str
    cohort_name: str
    method: str
    predicted_codes: list[str] = Field(default_factory=list)
    ground_truth_codes: list[str] = Field(default_factory=list)
    coverage: float
    over_inclusion_penalty: float
    hierarchical_score: float
    unresolvable_ground_truth: list[str] = Field(default_factory=list)


class DecompositionCaseResult(BaseModel):
    """One test case's scores: retrieval alone vs. decomposition + retrieval.

    Both scores are measured against the same ``cohort_id``'s curated
    ground truth, isolating what the ``decompose`` step costs (or doesn't)
    on top of retrieval quality that's already measured separately by
    :class:`~phenoforge.eval.harness.run_benchmark`.

    :ivar cohort_id: Bundled OHDSI Phenotype Library cohort id used as
        ground truth.
    :ivar cohort_name: Cohort display name.
    :ivar population_description: The free-text input fed to ``decompose``.
    :ivar seed_terms: What ``decompose`` actually extracted.
    :ivar baseline_hierarchical_score: Hybrid search's score using the
        cohort's own display name as the query directly — "if decomposition
        were skipped entirely."
    :ivar decomposed_hierarchical_score: Score from running the real
        ``decompose`` → per-term curated/generated resolution → assemble
        pipeline, starting from ``population_description``.
    :ivar gap: ``baseline_hierarchical_score - decomposed_hierarchical_score``.
        Positive means decomposition cost accuracy relative to querying
        with the right term directly; negative means decomposition (e.g.
        splitting into multiple terms) did better than the single-query
        baseline.
    """

    cohort_id: str
    cohort_name: str
    population_description: str
    seed_terms: list[str] = Field(default_factory=list)
    baseline_hierarchical_score: float
    decomposed_hierarchical_score: float
    gap: float


class DecompositionReport(BaseModel):
    """Aggregate report across every decomposition test case.

    :ivar results: One result per test case.
    :ivar mean_gap: Mean :attr:`DecompositionCaseResult.gap` across cases —
        decomposition's average accuracy cost on this test set.
    """

    results: list[DecompositionCaseResult] = Field(default_factory=list)
    mean_gap: float = 0.0


class BenchmarkReport(BaseModel):
    """Aggregate report across all cohorts and methods.

    :ivar results: Every per-cohort, per-method result.
    :ivar mean_coverage_by_method: Mean :attr:`CohortEvalResult.coverage`,
        grouped by method.
    :ivar mean_over_inclusion_by_method: Mean
        :attr:`CohortEvalResult.over_inclusion_penalty`, grouped by method.
    :ivar mean_hierarchical_score_by_method: Mean
        :attr:`CohortEvalResult.hierarchical_score`, grouped by method.
    """

    results: list[CohortEvalResult] = Field(default_factory=list)
    mean_coverage_by_method: dict[str, float] = Field(default_factory=dict)
    mean_over_inclusion_by_method: dict[str, float] = Field(default_factory=dict)
    mean_hierarchical_score_by_method: dict[str, float] = Field(default_factory=dict)
