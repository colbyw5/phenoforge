"""Direct concept_code lookup against the ICD-10-CM vocabulary."""

from __future__ import annotations

import duckdb

from phenoforge.engine.models import Concept

_ICD10CM = "ICD10CM"


def lookup_by_code(con: duckdb.DuckDBPyConnection, concept_code: str) -> Concept | None:
    """Look up a single ICD-10-CM concept by its code.

    Only ICD-10-CM is queryable here: SNOMED is loaded locally for future
    curated-set resolution but is never a user-facing lookup target.

    :param con: Open connection to ``vocab.duckdb``.
    :param concept_code: ICD-10-CM code, e.g. ``"E11.21"``.
    :returns: The matching concept, or ``None`` if no ICD-10-CM concept has
        that code.
    :rtype: Concept | None
    """
    row = con.execute(
        """
        SELECT concept_id, concept_code, concept_name, domain_id, vocabulary_id
        FROM concept
        WHERE vocabulary_id = ? AND concept_code = ?
        """,
        [_ICD10CM, concept_code],
    ).fetchone()
    if row is None:
        return None
    return Concept(
        concept_id=row[0],
        concept_code=row[1],
        concept_name=row[2],
        domain_id=row[3],
        vocabulary_id=row[4],
    )
