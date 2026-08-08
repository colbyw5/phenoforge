"""Ontology-aware evaluation metrics and benchmark harness.

Standard exact-match recall is the wrong metric for this task: retrieving
``E11.21`` when the target is ``E11.9`` is a near-miss under the same
parent, while retrieving a circulatory-chapter code is a total miss — both
score zero under exact match. Every metric here is instead derived from
tree distance between concepts (:mod:`phenoforge.eval.metrics`), so
near-misses earn partial credit and total misses don't.

Implemented, of the metrics named in the project's evaluation notes:

- **Hierarchical distance-weighted scoring** — :func:`~phenoforge.eval.metrics.tree_distance`
  and :func:`~phenoforge.eval.metrics.distance_weighted_score`.
- **Set-level coverage** (did the whole curated set get assembled) —
  :func:`~phenoforge.eval.metrics.set_coverage`.
- **Over-inclusion penalty** (cohort definitions fail on false positives too) —
  :func:`~phenoforge.eval.metrics.over_inclusion_penalty`.

**Decomposition accuracy is deliberately not implemented here.** Nothing in
this codebase decomposes a population description into seed concepts or
sub-queries yet — that's the not-yet-built LangGraph agent's job. Inventing
a throwaway decomposition step just to score it would be scope creep
disguised as fidelity; this is a stated omission, not a silent gap.

Ground truth comes from the bundled OHDSI Phenotype Library demo cohorts
(``scripts/fetch_phenotype_library.py``) — free, no hand-annotation needed.
It is a **lower bound**, not an exact set: see
:func:`phenoforge.engine.curated.resolve_curated_concepts` for the
exclusion/fan-out/non-SNOMED caveats that shape it. Every score in this
package accounts for that by scoring through distance-weighting in both
directions rather than set-equality, so a predicted code near — but not
literally inside — the curated set still earns partial credit instead of
being flatly penalized.

Orchestration (:mod:`phenoforge.eval.harness`) runs each retrieval method
against every bundled cohort and scores it; ``scripts/run_eval.py`` is the
CLI entry point.
"""
