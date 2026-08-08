"""Hierarchy expansion over ``CONCEPT_RELATIONSHIP`` ``Is a`` edges.

OMOP's precomputed transitive-closure table, ``CONCEPT_ANCESTOR``, has zero
rows for ICD-10-CM, so expansion instead walks direct ``Is a``/``Subsumes``
edges via a recursive CTE. The real Athena download contains redundant
direct edges (e.g. a code's grandparent is also a direct parent) — not
cycles, but multiple paths to the same concept — so every query here carries
a visited-path guard and the result is deduplicated to each concept's
shortest depth.

Two directions: :func:`expand_descendants` walks down (seed -> more specific
codes), :func:`expand_ancestors` walks up (seed -> less specific codes, to
the root). The latter powers hierarchy-distance scoring in
:mod:`phenoforge.eval`.
"""

from __future__ import annotations

import duckdb

from phenoforge.engine.lookup import lookup_by_code
from phenoforge.engine.models import ConceptWithProvenance, ProvenanceTier

_ICD10CM = "ICD10CM"

_DESCENDANTS_QUERY = """
WITH RECURSIVE descendants(concept_id, depth, path) AS (
    SELECT ?, 0, [?]
    UNION ALL
    SELECT r.concept_id_1, d.depth + 1, list_append(d.path, r.concept_id_1)
    FROM concept_relationship r
    JOIN descendants d ON d.concept_id = r.concept_id_2
    WHERE r.relationship_id = 'Is a'
      AND NOT list_contains(d.path, r.concept_id_1)
)
SELECT c.concept_id, c.concept_code, c.concept_name, c.domain_id, c.vocabulary_id,
       MIN(d.depth) AS depth
FROM descendants d
JOIN concept c ON c.concept_id = d.concept_id
WHERE d.depth > 0
GROUP BY c.concept_id, c.concept_code, c.concept_name, c.domain_id, c.vocabulary_id
ORDER BY depth, c.concept_id
"""


def expand_descendants(
    con: duckdb.DuckDBPyConnection, seed_code: str
) -> list[ConceptWithProvenance]:
    """Expand an ICD-10-CM seed code to all of its hierarchy descendants.

    A descendant of ``concept_id_2`` is any concept ``concept_id_1`` with an
    ``Is a`` edge to it (direct or transitive) — i.e. traversal follows
    ``Is a`` edges backwards from the seed, equivalently ``Subsumes`` forward
    from it.

    :param con: Open connection to ``vocab.duckdb``.
    :param seed_code: ICD-10-CM code to expand from, e.g. ``"E11"``.
    :returns: Descendant concepts tagged with
        :attr:`~phenoforge.engine.models.ProvenanceTier.GENERATED`
        provenance, ordered by ascending tree depth. Empty if ``seed_code``
        does not exist or has no descendants.
    :rtype: list[ConceptWithProvenance]
    """
    seed = lookup_by_code(con, seed_code)
    if seed is None:
        return []

    rows = con.execute(_DESCENDANTS_QUERY, [seed.concept_id, seed.concept_id]).fetchall()
    return [
        ConceptWithProvenance(
            concept_id=row[0],
            concept_code=row[1],
            concept_name=row[2],
            domain_id=row[3],
            vocabulary_id=row[4],
            tier=ProvenanceTier.GENERATED,
            source=f"hierarchy_expansion:{seed_code}",
        )
        for row in rows
    ]


_ANCESTORS_QUERY = """
WITH RECURSIVE ancestors(concept_id, depth, path) AS (
    SELECT ?, 0, [?]
    UNION ALL
    SELECT r.concept_id_2, a.depth + 1, list_append(a.path, r.concept_id_2)
    FROM concept_relationship r
    JOIN ancestors a ON a.concept_id = r.concept_id_1
    WHERE r.relationship_id = 'Is a'
      AND NOT list_contains(a.path, r.concept_id_2)
)
SELECT c.concept_id, c.concept_code, c.concept_name, c.domain_id, c.vocabulary_id,
       MIN(a.depth) AS depth
FROM ancestors a
JOIN concept c ON c.concept_id = a.concept_id
WHERE a.depth > 0
GROUP BY c.concept_id, c.concept_code, c.concept_name, c.domain_id, c.vocabulary_id
ORDER BY depth, c.concept_id
"""


def expand_ancestors(con: duckdb.DuckDBPyConnection, seed_code: str) -> list[ConceptWithProvenance]:
    """Walk an ICD-10-CM concept's ``Is a`` ancestor chain to the root.

    Counterpart to :func:`expand_descendants`, walking the same edges in the
    opposite direction (``concept_id_1`` "Is a" ``concept_id_2`` means
    ``concept_id_2`` is the parent).

    :param con: Open connection to ``vocab.duckdb``.
    :param seed_code: ICD-10-CM code to walk ancestors from, e.g. ``"E11.21"``.
    :returns: Ancestor concepts tagged with
        :attr:`~phenoforge.engine.models.ProvenanceTier.GENERATED`
        provenance, ordered by ascending distance from the seed. Empty if
        ``seed_code`` does not exist or is already a root.
    :rtype: list[ConceptWithProvenance]
    """
    seed = lookup_by_code(con, seed_code)
    if seed is None:
        return []

    rows = con.execute(_ANCESTORS_QUERY, [seed.concept_id, seed.concept_id]).fetchall()
    return [
        ConceptWithProvenance(
            concept_id=row[0],
            concept_code=row[1],
            concept_name=row[2],
            domain_id=row[3],
            vocabulary_id=row[4],
            tier=ProvenanceTier.GENERATED,
            source=f"hierarchy_ancestors:{seed_code}",
        )
        for row in rows
    ]


def ancestor_chain_ids(con: duckdb.DuckDBPyConnection, seed_code: str) -> list[int]:
    """Return ``seed_code``'s own concept id followed by its ancestor ids, nearest first.

    Thin wrapper around :func:`expand_ancestors` returning bare ids
    (self-inclusive, at distance 0) — the shape hierarchy-distance scoring
    in :mod:`phenoforge.eval` needs, without forcing every caller through
    the full :class:`~phenoforge.engine.models.ConceptWithProvenance` model.

    Concepts with multiple direct parents (real in the Athena data — see
    :func:`expand_ancestors`) make this a flattened traversal of what's
    actually a DAG, not a strict linear chain; ties in depth are broken by
    ``concept_id`` in the underlying query so the same seed always produces
    the same chain (needed for :func:`phenoforge.eval.metrics.tree_distance`
    to be reproducible run to run — an earlier version of this query had no
    tiebreaker and silently returned differently-ordered chains across
    identical calls).

    :param con: Open connection to ``vocab.duckdb``.
    :param seed_code: ICD-10-CM code.
    :returns: ``[seed_concept_id, parent_id, grandparent_id, ..., root_id]``.
        Empty if ``seed_code`` does not exist.
    :rtype: list[int]
    """
    seed = lookup_by_code(con, seed_code)
    if seed is None:
        return []
    ancestors = expand_ancestors(con, seed_code)
    return [seed.concept_id, *(a.concept_id for a in ancestors)]
