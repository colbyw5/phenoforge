"""Curated OHDSI Phenotype Library resolution to ICD-10-CM.

Extracts concept-set (not temporal/inclusion-rule) content from Circe-shaped
cohort JSON, resolves each SNOMED concept id to ICD-10-CM via the
``Mapped from`` edges loaded by ``scripts/load_vocab.py``, and applies a
fan-out guard so a broad SNOMED grouper in a cohort definition doesn't
silently expand into thousands of unrelated ICD-10-CM codes.

A free-text query is matched to a bundled cohort via
:func:`find_curated_definition`, which searches the same BM25/dense
retrieval used for ``generated`` results against the full ICD-10-CM
vocabulary and checks whether the best hit is actually a member of a
cohort's *resolved code list* — not, as an earlier version of this did, by
comparing the query against cohort *display names* (which let one
incidental shared word confidently mismatch an unrelated cohort).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
from pydantic import BaseModel

from phenoforge.engine.dense import DenseRetriever
from phenoforge.engine.hybrid import hybrid_search
from phenoforge.engine.models import (
    ConceptSet,
    ConceptWithProvenance,
    ProvenanceTier,
    UnmappableTerm,
)
from phenoforge.engine.retrieval import BM25Retriever

_ICD10CM = "ICD10CM"
_SNOMED = "SNOMED"
_REQUIRED_CONCEPT_KEYS = ("CONCEPT_ID", "CONCEPT_CODE", "VOCABULARY_ID")


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

    Items missing ``concept`` or any of its required fields are skipped
    rather than raising — this file is fetched from an external source
    (``scripts/fetch_phenotype_library.py``) with no schema guarantee, and a
    single malformed item should degrade gracefully, not crash the whole
    lookup.

    :param cohort_json_path: Path to one cohort's JSON file.
    :returns: Flattened concept-set items across all of the cohort's concept
        sets. Malformed items are silently omitted.
    :rtype: list[CuratedCohortConceptItem]
    """
    data = json.loads(cohort_json_path.read_text())
    items: list[CuratedCohortConceptItem] = []
    for concept_set in data.get("ConceptSets", []):
        for item in concept_set.get("expression", {}).get("items", []):
            concept = item.get("concept")
            if not isinstance(concept, dict):
                continue
            if any(key not in concept for key in _REQUIRED_CONCEPT_KEYS):
                continue
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
    :returns: Resolved ICD-10-CM concepts tagged ``curated``, deduplicated by
        ``concept_id`` (two cohort items can map to the same code), and
        unmappable items with their reasons.
    :rtype: tuple[list[ConceptWithProvenance], list[UnmappableTerm]]
    """
    high_fanout = set(find_high_fanout_snomed_concepts(con, fanout_threshold))
    source = f"ohdsi_pl:{cohort_id}:{cohort_name}"

    resolved: list[ConceptWithProvenance] = []
    seen_concept_ids: set[int] = set()
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
            if row[0] in seen_concept_ids:
                continue
            seen_concept_ids.add(row[0])
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


def _build_cohort_membership(
    con: duckdb.DuckDBPyConnection,
    library_dir: Path,
    manifest: dict[str, str],
    fanout_threshold: int,
) -> dict[int, set[str]]:
    """Resolve every bundled cohort and index which cohorts contain each concept.

    Re-resolves all cohorts on every call rather than caching — at the
    bundled demo library's scale (8 cohorts, small JSON files) this is
    cheap, and caching would need invalidation the moment
    ``fetch_phenotype_library.py`` re-runs. Revisit if the library grows.

    :param con: Open connection to ``vocab.duckdb``.
    :param library_dir: Directory of fetched OHDSI Phenotype Library cohorts.
    :param manifest: Cohort id -> display name, as loaded from ``manifest.json``.
    :param fanout_threshold: Passed through to :func:`resolve_curated_concepts`.
    :returns: ICD-10-CM ``concept_id`` -> the set of cohort ids whose
        resolved definition includes it. Cohorts with a missing JSON file
        (stale manifest entry) are silently skipped, matching
        :func:`load_curated_concept_set`'s tolerance for that case elsewhere.
    :rtype: dict[int, set[str]]
    """
    membership: dict[int, set[str]] = {}
    for cohort_id, cohort_name in manifest.items():
        cohort_json_path = library_dir / f"{cohort_id}.json"
        if not cohort_json_path.exists():
            continue
        items = load_cohort_concept_items(cohort_json_path)
        resolved, _ = resolve_curated_concepts(con, cohort_id, cohort_name, items, fanout_threshold)
        for concept in resolved:
            membership.setdefault(concept.concept_id, set()).add(cohort_id)
    return membership


def find_curated_definition(
    con: duckdb.DuckDBPyConnection,
    query: str,
    library_dir: Path,
    bm25: BM25Retriever,
    dense: DenseRetriever | None = None,
    fanout_threshold: int = 100,
    k: int = 10,
) -> ConceptSet:
    """Find and resolve the OHDSI Phenotype Library cohort whose own codes best match a query.

    Runs the same BM25 (+ dense, if given) search used for ``generated``
    retrieval against the full ICD-10-CM vocabulary, then walks the ranked
    results looking for the best-ranked one that is actually a member of a
    bundled cohort's resolved code list — matching on real curated code
    content rather than a cohort's display name, which
    (found via a real end-to-end run) let a single incidental shared word
    like "diabetic" curated-match "diabetic nephropathy" to an unrelated
    "Diabetic ketoacidosis" cohort. A hit belonging to more than one cohort
    is deliberately treated as no match rather than guessed between them,
    for the same reason: a ``curated`` result claims to be safe to use
    as-is, and an ambiguous one would misrepresent that.

    :param con: Open connection to ``vocab.duckdb``.
    :param query: Free-text population description or seed clinical term.
    :param library_dir: Directory of fetched OHDSI Phenotype Library cohorts
        (``scripts/fetch_phenotype_library.py`` output).
    :param bm25: A built BM25 retriever over ICD-10-CM concept names —
        shared with :func:`~phenoforge.engine.hybrid.hybrid_search`'s other
        callers rather than built fresh here.
    :param dense: A built dense retriever, or ``None`` to search lexically
        only, matching :func:`~phenoforge.engine.hybrid.hybrid_search`'s
        existing fallback.
    :param fanout_threshold: Passed through to cohort resolution.
    :param k: How many top hybrid-search results to check for cohort
        membership before giving up.
    :returns: The matched cohort's resolved ``curated`` concepts, or an
        empty set with one explanatory :class:`~phenoforge.engine.models.UnmappableTerm`
        if the library hasn't been fetched, nothing in the top ``k`` results
        belongs to a cohort, or the best hit is claimed by more than one
        cohort. A cohort with a stale manifest entry (missing JSON file)
        simply can never be matched — see :func:`_build_cohort_membership`.
    :rtype: ConceptSet
    """
    manifest_path = library_dir / "manifest.json"
    if not manifest_path.exists():
        return ConceptSet(
            unmappable=[
                UnmappableTerm(
                    term=query,
                    reason=(
                        f"no phenotype library at {library_dir} — run "
                        "scripts/fetch_phenotype_library.py first"
                    ),
                )
            ]
        )

    manifest = json.loads(manifest_path.read_text())
    membership = _build_cohort_membership(con, library_dir, manifest, fanout_threshold)

    candidates, _ = hybrid_search(bm25, dense, query, k=k)
    cohort_id: str | None = None
    for candidate in candidates:
        cohort_ids = membership.get(candidate.concept_id)
        if not cohort_ids:
            continue
        if len(cohort_ids) > 1:
            return ConceptSet(
                unmappable=[
                    UnmappableTerm(
                        term=query,
                        reason=(
                            f"best match ({candidate.concept_code}) belongs to multiple "
                            f"curated cohorts ({', '.join(sorted(cohort_ids))}) — ambiguous"
                        ),
                    )
                ]
            )
        cohort_id = next(iter(cohort_ids))
        break

    if cohort_id is None:
        return ConceptSet(
            unmappable=[UnmappableTerm(term=query, reason="no bundled cohort matches this query")]
        )
    # cohort_id only ever comes from membership, which already required
    # successfully reading this exact JSON file — no FileNotFoundError
    # handling needed here (a stale manifest entry just means that cohort
    # can never be matched; see _build_cohort_membership).
    return load_curated_concept_set(con, cohort_id, library_dir, fanout_threshold)
