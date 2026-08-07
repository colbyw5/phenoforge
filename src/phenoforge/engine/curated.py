"""Curated OHDSI Phenotype Library resolution to ICD-10-CM.

Extracts concept-set (not temporal/inclusion-rule) content from Circe-shaped
cohort JSON, resolves each SNOMED concept id to ICD-10-CM via the
``Mapped from`` edges loaded by ``scripts/load_vocab.py``, and applies a
fan-out guard so a broad SNOMED grouper in a cohort definition doesn't
silently expand into thousands of unrelated ICD-10-CM codes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
from pydantic import BaseModel

from phenoforge.engine.models import (
    ConceptSet,
    ConceptWithProvenance,
    ProvenanceTier,
    UnmappableTerm,
)

_ICD10CM = "ICD10CM"
_SNOMED = "SNOMED"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def find_high_fanout_snomed_concepts(
    con: duckdb.DuckDBPyConnection, fanout_threshold: int
) -> list[int]:
    """Flag SNOMED concepts whose ``Mapped from`` fan-out into ICD-10-CM is excessive.

    Verified against the v2026 Athena download: fan-out ranges 1..3029, with
    a median of 2. The high tail is structural (a high-level SNOMED grouper
    mapping to thousands of unrelated ICD-10-CM codes), not clinical
    specificity, and should be surfaced for review rather than expanded
    blindly at resolution time. Used both by the vocabulary loader (to flag
    broad mappings at build time) and by curated-set resolution (to skip
    them at resolution time).

    :param con: Open connection with ``concept`` and ``concept_relationship`` loaded.
    :param fanout_threshold: Concepts mapping to more than this many
        ICD-10-CM codes are flagged.
    :returns: SNOMED ``concept_id`` values exceeding the threshold.
    :rtype: list[int]
    """
    # For relationship_id='Mapped from', concept_id_1 is the SNOMED concept
    # and concept_id_2 is the ICD-10-CM concept it was mapped from (verified
    # against the v2026 download — the "SNOMED -> ICD10CM" framing describes
    # the mapping direction, not the column order).
    rows = con.execute(
        """
        SELECT r.concept_id_1 AS snomed_concept_id, COUNT(*) AS n
        FROM concept_relationship r
        JOIN concept sn ON sn.concept_id = r.concept_id_1 AND sn.vocabulary_id = ?
        JOIN concept icd ON icd.concept_id = r.concept_id_2 AND icd.vocabulary_id = ?
        WHERE r.relationship_id = 'Mapped from'
        GROUP BY r.concept_id_1
        HAVING COUNT(*) > ?
        ORDER BY n DESC
        """,
        [_SNOMED, _ICD10CM, fanout_threshold],
    ).fetchall()
    return [int(row[0]) for row in rows]


class CuratedCohortConceptItem(BaseModel):
    """One item within a cohort's concept set, extracted from Circe JSON.

    :ivar concept_id: Source-vocabulary OMOP concept id (as recorded in the
        cohort's own vocabulary snapshot — not necessarily present in the
        locally loaded ``concept`` table).
    :ivar concept_code: Vocabulary-native code, e.g. a SNOMED code.
    :ivar vocabulary_id: Source vocabulary of the item, e.g. ``"SNOMED"``.
    :ivar is_excluded: Circe's ``isExcluded`` flag. Excluded items are
        dropped entirely during resolution (see :func:`resolve_curated_concepts`).
    :ivar include_descendants: Circe's ``includeDescendants`` flag. Not
        expanded during resolution (see :func:`resolve_curated_concepts`).
    """

    concept_id: int
    concept_code: str
    vocabulary_id: str
    is_excluded: bool
    include_descendants: bool


def load_cohort_concept_items(cohort_json_path: Path) -> list[CuratedCohortConceptItem]:
    """Parse a Circe cohort JSON file's ``ConceptSets`` into flat items.

    Extracts only the concept-set portion of the cohort definition — Circe
    cohort JSON also carries temporal windows and inclusion-rule logic
    (``PrimaryCriteria``, ``InclusionRules``, etc.), which is out of scope
    and not read here.

    :param cohort_json_path: Path to one cohort's JSON file.
    :returns: Flattened concept-set items across all of the cohort's concept
        sets.
    :rtype: list[CuratedCohortConceptItem]
    """
    data = json.loads(cohort_json_path.read_text())
    items: list[CuratedCohortConceptItem] = []
    for concept_set in data.get("ConceptSets", []):
        for item in concept_set.get("expression", {}).get("items", []):
            concept = item["concept"]
            items.append(
                CuratedCohortConceptItem(
                    concept_id=concept["CONCEPT_ID"],
                    concept_code=concept["CONCEPT_CODE"],
                    vocabulary_id=concept["VOCABULARY_ID"],
                    is_excluded=item.get("isExcluded", False),
                    include_descendants=item.get("includeDescendants", False),
                )
            )
    return items


def resolve_curated_concepts(
    con: duckdb.DuckDBPyConnection,
    cohort_id: str,
    cohort_name: str,
    items: list[CuratedCohortConceptItem],
    fanout_threshold: int = 100,
) -> tuple[list[ConceptWithProvenance], list[UnmappableTerm]]:
    """Resolve a cohort's SNOMED concept-set items to ICD-10-CM.

    Per non-excluded item:

    - ``vocabulary_id != 'SNOMED'``: reported as unmappable — RxNorm/LOINC
      items in a cohort definition are out of scope (only the SNOMED to
      ICD-10-CM mapping is loaded).
    - fan-out over ``fanout_threshold`` (see
      :func:`find_high_fanout_snomed_concepts`): reported as unmappable
      rather than resolved, so a broad SNOMED grouper doesn't silently
      expand the curated set into thousands of unrelated codes.
    - otherwise: resolved via ``concept_relationship`` ``'Mapped from'``
      edges (``concept_id_1`` = SNOMED, ``concept_id_2`` = ICD-10-CM),
      tagged :attr:`~phenoforge.engine.models.ProvenanceTier.CURATED`.

    ``is_excluded`` items are dropped entirely — ``ConceptSet`` has no
    exclusion concept, so silently omitting them is more honest than
    forcing them into ``unmappable``. ``include_descendants`` is not
    expanded: doing so would require a SNOMED-side hierarchy traversal,
    which is out of scope; this is a conservative simplification (it
    under-counts rather than over-includes).

    :param con: Open connection to ``vocab.duckdb``.
    :param cohort_id: The cohort's OHDSI Phenotype Library id, used in provenance.
    :param cohort_name: The cohort's display name, used in provenance.
    :param items: Concept-set items from :func:`load_cohort_concept_items`.
    :param fanout_threshold: Passed through to :func:`find_high_fanout_snomed_concepts`.
    :returns: Resolved ICD-10-CM concepts tagged ``curated``, and unmappable
        items with their reasons.
    :rtype: tuple[list[ConceptWithProvenance], list[UnmappableTerm]]
    """
    high_fanout = set(find_high_fanout_snomed_concepts(con, fanout_threshold))
    source = f"ohdsi_pl:{cohort_id}:{cohort_name}"

    resolved: list[ConceptWithProvenance] = []
    unmappable: list[UnmappableTerm] = []

    for item in items:
        if item.is_excluded:
            continue
        if item.vocabulary_id != _SNOMED:
            unmappable.append(
                UnmappableTerm(
                    term=item.concept_code,
                    reason=f"unsupported source vocabulary {item.vocabulary_id}",
                )
            )
            continue
        if item.concept_id in high_fanout:
            unmappable.append(
                UnmappableTerm(
                    term=item.concept_code,
                    reason=(
                        f"fan-out exceeds threshold {fanout_threshold}, skipped to "
                        "avoid an overly broad grouper silently expanding the set"
                    ),
                )
            )
            continue

        rows = con.execute(
            """
            SELECT icd.concept_id, icd.concept_code, icd.concept_name,
                   icd.domain_id, icd.vocabulary_id
            FROM concept_relationship r
            JOIN concept sn ON sn.concept_id = r.concept_id_1 AND sn.vocabulary_id = ?
            JOIN concept icd ON icd.concept_id = r.concept_id_2 AND icd.vocabulary_id = ?
            WHERE r.relationship_id = 'Mapped from' AND r.concept_id_1 = ?
            """,
            [_SNOMED, _ICD10CM, item.concept_id],
        ).fetchall()

        if not rows:
            unmappable.append(
                UnmappableTerm(
                    term=item.concept_code,
                    reason="no ICD-10-CM mapping found for this SNOMED concept",
                )
            )
            continue

        for row in rows:
            resolved.append(
                ConceptWithProvenance(
                    concept_id=row[0],
                    concept_code=row[1],
                    concept_name=row[2],
                    domain_id=row[3],
                    vocabulary_id=row[4],
                    tier=ProvenanceTier.CURATED,
                    source=source,
                )
            )

    return resolved, unmappable


def load_curated_concept_set(
    con: duckdb.DuckDBPyConnection,
    cohort_id: str,
    library_dir: Path,
    fanout_threshold: int = 100,
) -> ConceptSet:
    """Load one cohort's JSON + manifest name from ``library_dir`` and resolve it.

    :param con: Open connection to ``vocab.duckdb``.
    :param cohort_id: The cohort id, e.g. ``"503"``. Expects
        ``{library_dir}/{cohort_id}.json`` and a ``manifest.json`` mapping
        ids to display names, both written by
        ``scripts/fetch_phenotype_library.py``.
    :param library_dir: Directory containing fetched cohort JSON files.
    :param fanout_threshold: Passed through to :func:`resolve_curated_concepts`.
    :returns: The resolved concept set.
    :rtype: ConceptSet
    :raises FileNotFoundError: If the cohort JSON or manifest is missing.
    """
    manifest = json.loads((library_dir / "manifest.json").read_text())
    cohort_name = manifest.get(cohort_id, cohort_id)
    items = load_cohort_concept_items(library_dir / f"{cohort_id}.json")
    resolved, unmappable = resolve_curated_concepts(
        con, cohort_id, cohort_name, items, fanout_threshold
    )
    return ConceptSet(concepts=resolved, unmappable=unmappable)


def find_matching_cohort(query: str, manifest: dict[str, str]) -> str | None:
    """Find the best-matching cohort id for a free-text query by token overlap.

    The bundled demo library has only a handful of cohorts, which doesn't
    warrant a BM25 index (see :class:`~phenoforge.engine.retrieval.BM25Retriever`
    for that approach at ICD-10-CM's ~98k-concept scale) — simple token
    overlap against cohort display names is sufficient here.

    :param query: Free-text population description.
    :param manifest: Cohort id -> display name, as loaded from ``manifest.json``.
    :returns: The cohort id with the most overlapping tokens, or ``None`` if
        no cohort shares any token with the query.
    :rtype: str | None
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return None

    best_id: str | None = None
    best_overlap = 0
    for cohort_id, name in manifest.items():
        overlap = len(query_tokens & set(_tokenize(name)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = cohort_id
    return best_id
