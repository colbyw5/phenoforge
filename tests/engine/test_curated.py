"""Tests for phenoforge.engine.curated."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from phenoforge.engine.curated import (
    CuratedCohortConceptItem,
    find_matching_cohort,
    load_cohort_concept_items,
    load_curated_concept_set,
    resolve_curated_concepts,
)
from phenoforge.engine.models import ProvenanceTier
from tests.scripts.conftest import MiniVocab


def _circe_json(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "ConceptSets": [
            {
                "id": 0,
                "name": "test concept set",
                "expression": {"items": items},
            }
        ]
    }


def _circe_item(
    concept_id: int,
    concept_code: str,
    vocabulary_id: str = "SNOMED",
    is_excluded: bool = False,
    include_descendants: bool = False,
) -> dict[str, object]:
    return {
        "concept": {
            "CONCEPT_ID": concept_id,
            "CONCEPT_CODE": concept_code,
            "VOCABULARY_ID": vocabulary_id,
            "CONCEPT_NAME": "irrelevant for parsing",
            "DOMAIN_ID": "Condition",
        },
        "isExcluded": is_excluded,
        "includeDescendants": include_descendants,
    }


def test_load_cohort_concept_items_parses_circe_json(tmp_path: Path) -> None:
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(_circe_json([_circe_item(90721000, "90721000")])))
    items = load_cohort_concept_items(path)
    assert items == [
        CuratedCohortConceptItem(
            concept_id=90721000,
            concept_code="90721000",
            vocabulary_id="SNOMED",
            is_excluded=False,
            include_descendants=False,
        )
    ]


def test_resolve_curated_concepts_maps_to_icd10cm(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection
) -> None:
    items = [
        CuratedCohortConceptItem(
            concept_id=mini_vocab.sn_normal_id,
            concept_code="90721000",
            vocabulary_id="SNOMED",
            is_excluded=False,
            include_descendants=False,
        )
    ]
    resolved, unmappable = resolve_curated_concepts(con, "1", "test cohort", items)
    assert unmappable == []
    assert len(resolved) == 1
    assert resolved[0].concept_code == "E11.21"
    assert resolved[0].tier is ProvenanceTier.CURATED
    assert resolved[0].source == "ohdsi_pl:1:test cohort"


def test_resolve_curated_concepts_drops_excluded_items(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection
) -> None:
    items = [
        CuratedCohortConceptItem(
            concept_id=mini_vocab.sn_normal_id,
            concept_code="90721000",
            vocabulary_id="SNOMED",
            is_excluded=True,
            include_descendants=False,
        )
    ]
    resolved, unmappable = resolve_curated_concepts(con, "1", "test cohort", items)
    assert resolved == []
    assert unmappable == []


def test_resolve_curated_concepts_skips_high_fanout(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection
) -> None:
    items = [
        CuratedCohortConceptItem(
            concept_id=mini_vocab.high_fanout_snomed_id,
            concept_code="64572001",
            vocabulary_id="SNOMED",
            is_excluded=False,
            include_descendants=False,
        )
    ]
    resolved, unmappable = resolve_curated_concepts(con, "1", "test cohort", items)
    assert resolved == []
    assert len(unmappable) == 1
    assert "fan-out" in unmappable[0].reason


def test_resolve_curated_concepts_skips_non_snomed(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection
) -> None:
    items = [
        CuratedCohortConceptItem(
            concept_id=999999,
            concept_code="12345",
            vocabulary_id="RxNorm",
            is_excluded=False,
            include_descendants=False,
        )
    ]
    resolved, unmappable = resolve_curated_concepts(con, "1", "test cohort", items)
    assert resolved == []
    assert len(unmappable) == 1
    assert "unsupported source vocabulary RxNorm" in unmappable[0].reason


def test_load_cohort_concept_items_skips_malformed_items(tmp_path: Path) -> None:
    good = _circe_item(90721000, "90721000")
    missing_concept = {"isExcluded": False, "includeDescendants": False}
    missing_field = {
        "concept": {"CONCEPT_ID": 1, "CONCEPT_CODE": "X", "DOMAIN_ID": "Condition"},
        "isExcluded": False,
        "includeDescendants": False,
    }
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(_circe_json([good, missing_concept, missing_field])))

    items = load_cohort_concept_items(path)

    assert len(items) == 1
    assert items[0].concept_code == "90721000"


def test_resolve_curated_concepts_dedupes_by_concept_id(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection
) -> None:
    item = CuratedCohortConceptItem(
        concept_id=mini_vocab.sn_normal_id,
        concept_code="90721000",
        vocabulary_id="SNOMED",
        is_excluded=False,
        include_descendants=False,
    )
    resolved, unmappable = resolve_curated_concepts(con, "1", "test cohort", [item, item])
    assert unmappable == []
    assert len(resolved) == 1


def test_load_curated_concept_set(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    library_dir = tmp_path / "phenotype_library"
    library_dir.mkdir()
    (library_dir / "manifest.json").write_text(json.dumps({"1": "Diabetic nephropathy demo"}))
    (library_dir / "1.json").write_text(
        json.dumps(_circe_json([_circe_item(mini_vocab.sn_normal_id, "90721000")]))
    )

    concept_set = load_curated_concept_set(con, "1", library_dir)
    assert len(concept_set.concepts) == 1
    assert concept_set.concepts[0].concept_code == "E11.21"
    assert concept_set.concepts[0].source == "ohdsi_pl:1:Diabetic nephropathy demo"


def test_find_matching_cohort() -> None:
    manifest = {
        "1": "Type 2 diabetes mellitus",
        "2": "Chronic kidney disease",
    }
    assert find_matching_cohort("adults with type 2 diabetes", manifest) == "1"
    assert find_matching_cohort("chronic kidney disease patients", manifest) == "2"


def test_find_matching_cohort_no_overlap_returns_none() -> None:
    manifest = {"1": "Type 2 diabetes mellitus"}
    assert find_matching_cohort("xyzzy plugh quux", manifest) is None


def test_find_matching_cohort_empty_query_returns_none() -> None:
    manifest = {"1": "Type 2 diabetes mellitus"}
    assert find_matching_cohort("", manifest) is None


def test_find_matching_cohort_tie_returns_none() -> None:
    manifest = {
        "40": "Diabetes Mellitus Type 2 or history of diabetes",
        "503": "Type 2 diabetes mellitus",
    }
    assert find_matching_cohort("type 2 diabetes", manifest) is None


def test_resolve_curated_concepts_no_mapping_found(
    mini_vocab: MiniVocab, con: duckdb.DuckDBPyConnection
) -> None:
    unmapped_snomed_id = 9999999999
    items = [
        CuratedCohortConceptItem(
            concept_id=unmapped_snomed_id,
            concept_code="00000000",
            vocabulary_id="SNOMED",
            is_excluded=False,
            include_descendants=False,
        )
    ]
    resolved, unmappable = resolve_curated_concepts(con, "1", "test cohort", items)
    assert resolved == []
    assert len(unmappable) == 1
    assert "no ICD-10-CM mapping found" in unmappable[0].reason
