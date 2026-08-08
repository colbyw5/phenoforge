"""Ontology-aware scoring: tree distance, partial credit, coverage, over-inclusion.

Pure functions, no ``duckdb`` import anywhere in this file. Everything takes
already-computed ancestor chains or scores as plain ``list[int]`` /
``float`` inputs, so the property-based tests in
``tests/eval/test_metrics.py`` never touch a database — chain construction
lives in :func:`phenoforge.engine.expansion.ancestor_chain_ids`, and
DB-backed orchestration lives in :mod:`phenoforge.eval.harness`.

Standard exact-match recall is the wrong metric for this task: retrieving
``E11.21`` when the target is ``E11.9`` is a near-miss under the same
parent, while retrieving a circulatory-chapter code is a total miss — both
score zero under exact match today. Every score here is instead derived
from :func:`tree_distance`, so near-misses earn partial credit and total
misses don't.
"""

from __future__ import annotations


def tree_distance(chain_a: list[int], chain_b: list[int]) -> int | None:
    """Distance between two concepts via their lowest common ancestor.

    ``chain_a``/``chain_b`` are each ``[self_id, parent_id, ..., root_id]``,
    self-inclusive and nearest-first, as returned by
    :func:`phenoforge.engine.expansion.ancestor_chain_ids`. Walking
    ``chain_a`` outward from self, the first id also present in ``chain_b``
    is the lowest common ancestor — the closest shared node, since
    ``chain_a`` is ordered nearest-to-farthest.

    :param chain_a: Ancestor chain of the first concept.
    :param chain_b: Ancestor chain of the second concept.
    :returns: Tree-edge distance (sum of each concept's distance to the
        LCA), or ``None`` if the chains share no common ancestor (different
        chapters or vocabularies entirely — a true miss, not a near-miss).
    :rtype: int | None
    """
    index_in_b = {concept_id: i for i, concept_id in enumerate(chain_b)}
    for i, concept_id in enumerate(chain_a):
        j = index_in_b.get(concept_id)
        if j is not None:
            return i + j
    return None


def distance_weighted_score(distance: int | None, max_distance: int = 6) -> float:
    """Convert a tree distance into partial credit in ``[0, 1]``.

    ``distance=0`` (exact match) scores ``1.0``. Score decays linearly to
    ``0.0`` at ``distance >= max_distance``. ``None`` (no common ancestor —
    a total miss) scores ``0.0``. Linear decay is chosen over exponential
    for interpretability at prototype scale: a near-miss like ``E11.21`` vs.
    ``E11.9`` (distance 2, common parent ``E11``) should land clearly above
    zero and clearly below 1.0, not be swamped by a steep curve.

    :param distance: Tree distance from :func:`tree_distance`, or ``None``.
    :param max_distance: Distance at or beyond which score floors to ``0.0``.
    :returns: Score in ``[0.0, 1.0]``.
    :rtype: float
    """
    if distance is None:
        return 0.0
    if distance <= 0:
        return 1.0
    if distance >= max_distance:
        return 0.0
    return 1.0 - (distance / max_distance)


def set_coverage(predicted_scores: list[float], n_ground_truth: int) -> float:
    """Mean of each ground-truth code's best-match score against predicted codes.

    The caller has already computed one best :func:`distance_weighted_score`
    per ground-truth code (``0.0`` if nothing predicted is even a
    near-miss) — see :func:`phenoforge.eval.harness.match_scores`.

    :param predicted_scores: One best-match score per ground-truth code,
        each in ``[0, 1]``.
    :param n_ground_truth: Number of ground-truth codes; must equal
        ``len(predicted_scores)``.
    :returns: Mean coverage in ``[0, 1]``. ``1.0`` (vacuously) if
        ``n_ground_truth == 0``.
    :rtype: float
    :raises ValueError: If ``len(predicted_scores) != n_ground_truth``.
    """
    if len(predicted_scores) != n_ground_truth:
        raise ValueError(
            f"len(predicted_scores)={len(predicted_scores)} != n_ground_truth={n_ground_truth}"
        )
    if n_ground_truth == 0:
        return 1.0
    return sum(predicted_scores) / n_ground_truth


def over_inclusion_penalty(predicted_scores: list[float], n_predicted: int) -> float:
    """Fraction of predicted codes with no good match anywhere in ground truth.

    The caller has already computed one best :func:`distance_weighted_score`
    per predicted code against ground truth. A predicted code close to
    *something* in ground truth — a legitimately-missing sibling, given the
    curated set is a lower bound (see
    :func:`phenoforge.engine.curated.resolve_curated_concepts`) — earns
    partial credit rather than being flatly penalized as a false positive.

    :param predicted_scores: One best-match score per predicted code, each
        in ``[0, 1]``.
    :param n_predicted: Number of predicted codes; must equal
        ``len(predicted_scores)``.
    :returns: Penalty in ``[0, 1]`` — ``mean(1 - score)`` over predicted
        codes. ``0.0`` (vacuously) if ``n_predicted == 0``.
    :rtype: float
    :raises ValueError: If ``len(predicted_scores) != n_predicted``.
    """
    if len(predicted_scores) != n_predicted:
        raise ValueError(
            f"len(predicted_scores)={len(predicted_scores)} != n_predicted={n_predicted}"
        )
    if n_predicted == 0:
        return 0.0
    return sum(1.0 - score for score in predicted_scores) / n_predicted


def hierarchical_score(coverage: float, over_inclusion_penalty: float) -> float:
    """Harmonic mean of coverage and precision (``1 - over_inclusion_penalty``).

    An F1-style single quality number: ``coverage`` plays the recall role
    (did we get the whole curated set), ``1 - over_inclusion_penalty`` plays
    the precision role (is what we got actually relevant). Harmonic rather
    than arithmetic mean so a method can't hide a bad score on one axis by
    being good on the other — cohort definitions fail on false positives
    too, not just missed codes.

    :param coverage: :func:`set_coverage` output, ``[0, 1]``.
    :param over_inclusion_penalty: :func:`over_inclusion_penalty` output,
        ``[0, 1]``.
    :returns: Harmonic mean in ``[0, 1]``. ``0.0`` if both coverage and
        precision are ``0.0``.
    :rtype: float
    """
    precision = 1.0 - over_inclusion_penalty
    denominator = coverage + precision
    if denominator == 0.0:
        return 0.0
    return 2.0 * coverage * precision / denominator
