"""Hierarchy (descendant) expansion over ``CONCEPT_RELATIONSHIP`` ``Is a`` edges.

DESIGN.md D1: ``CONCEPT_ANCESTOR`` has zero ICD-10-CM coverage, so expansion
walks direct ``Is a``/``Subsumes`` edges via a recursive CTE instead of a
precomputed closure. The real Athena download contains redundant direct
edges (e.g. a code's grandparent is also a direct parent) — not cycles, but
multiple paths to the same concept — so the CTE carries a visited-path guard
and the result is deduplicated to each concept's shortest depth.
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
ORDER BY depth
"""


def expand_descendants(
    con: duckdb.DuckDBPyConnection, seed_code: str
) -> list[ConceptWithProvenance]:
    """Expand an ICD-10-CM seed code to all of its hierarchy descendants.

    A descendant of ``concept_id_2`` is any concept ``concept_id_1`` with an
    ``Is a`` edge to it (direct or transitive) — i.e. traversal follows
    ``Is a`` edges backwards from the seed, equivalently ``Subsumes`` forward
    from it (DESIGN.md D1).

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
