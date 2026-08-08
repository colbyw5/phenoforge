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
