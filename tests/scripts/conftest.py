"""Fixture: a ~200-concept miniature Athena vocabulary for scripts/load_vocab.py tests.

Builds synthetic ``CONCEPT.csv``, ``CONCEPT_RELATIONSHIP.csv``, and a CCSR
category CSV on disk in Athena's actual export shape (tab-delimited, no
quoting for the OMOP tables; single/double mixed quoting for CCSR), then runs
the real loader against them. Tests exercise the built DuckDB, not mocks.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pytest

from scripts.load_vocab import LoadReport, VocabPaths, build_vocab_db

_OMOP_COLUMNS = [
    "concept_id",
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
    "concept_code",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
]

_REL_COLUMNS = [
    "concept_id_1",
    "concept_id_2",
    "relationship_id",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
]


def _concept_row(
    concept_id: int,
    name: str,
    code: str,
    vocabulary_id: str = "ICD10CM",
    domain_id: str = "Condition",
    concept_class_id: str = "ICD10CM code",
    standard_concept: str = "",
    invalid_reason: str = "",
) -> dict[str, str]:
    return {
        "concept_id": str(concept_id),
        "concept_name": name,
        "domain_id": domain_id,
        "vocabulary_id": vocabulary_id,
        "concept_class_id": concept_class_id,
        "standard_concept": standard_concept,
        "concept_code": code,
        "valid_start_date": "19700101",
        "valid_end_date": "20991231",
        "invalid_reason": invalid_reason,
    }


def _rel_row(concept_id_1: int, concept_id_2: int, relationship_id: str) -> dict[str, str]:
    return {
        "concept_id_1": str(concept_id_1),
        "concept_id_2": str(concept_id_2),
        "relationship_id": relationship_id,
        "valid_start_date": "19700101",
        "valid_end_date": "20991231",
        "invalid_reason": "",
    }


@dataclass(frozen=True)
class MiniVocab:
    """Handles to the built fixture database and known concept ids.

    :ivar db_path: Path to the built DuckDB database.
    :ivar report: The :class:`~scripts.load_vocab.LoadReport` from the build.
    :ivar e11_id: ``concept_id`` for E11 (Type 2 diabetes mellitus).
    :ivar e11_2_id: ``concept_id`` for E11.2 (with kidney complications).
    :ivar e11_21_id: ``concept_id`` for E11.21 (diabetic nephropathy).
    :ivar high_fanout_snomed_id: SNOMED ``concept_id`` engineered to exceed
        the default fan-out threshold.
    """

    db_path: Path
    report: LoadReport
    e11_id: int
    e11_2_id: int
    e11_21_id: int
    high_fanout_snomed_id: int


@pytest.fixture
def mini_vocab(tmp_path: Path) -> Iterator[MiniVocab]:
    """Build a ~200-concept miniature vocabulary and load it with the real loader.

    Layout: an ICD-10-CM diabetes hierarchy (``E11`` -> ``E11.2`` ->
    ``E11.21``, matching the shape verified against the real Athena
    download) plus ~190 filler ICD-10-CM leaf concepts scattered across
    unrelated chapters, a handful of SNOMED concepts with ``Mapped
    from``/``Maps to`` edges into ICD-10-CM (including one deliberately
    high-fanout concept to exercise the fan-out guard), and a small CCSR
    category file.

    :param tmp_path: pytest-provided temporary directory.
    :yields: :class:`MiniVocab` with the built database path and known ids.
    """
    concept_id = 1000
    concepts: list[dict[str, str]] = []
    relationships: list[dict[str, str]] = []

    def next_id() -> int:
        nonlocal concept_id
        concept_id += 1
        return concept_id

    e11_id = next_id()
    e11_2_id = next_id()
    e11_21_id = next_id()
    concepts.append(_concept_row(e11_id, "Type 2 diabetes mellitus", "E11"))
    concepts.append(
        _concept_row(e11_2_id, "Type 2 diabetes mellitus with kidney complications", "E11.2")
    )
    concepts.append(
        _concept_row(e11_21_id, "Type 2 diabetes mellitus with diabetic nephropathy", "E11.21")
    )
    relationships.append(_rel_row(e11_21_id, e11_2_id, "Is a"))
    relationships.append(_rel_row(e11_2_id, e11_21_id, "Subsumes"))
    relationships.append(_rel_row(e11_2_id, e11_id, "Is a"))
    relationships.append(_rel_row(e11_id, e11_2_id, "Subsumes"))

    # ~190 unrelated filler ICD-10-CM leaves so the vocabulary is ~200 concepts.
    filler_ids = []
    for i in range(190):
        cid = next_id()
        filler_ids.append(cid)
        concepts.append(_concept_row(cid, f"Filler condition {i}", f"Z{i:03d}"))

    # A handful of SNOMED concepts mapping into ICD-10-CM.
    sn_normal_id = next_id()
    concepts.append(
        _concept_row(
            sn_normal_id,
            "Diabetic nephropathy (disorder)",
            "90721000",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
        )
    )
    relationships.append(_rel_row(sn_normal_id, e11_21_id, "Mapped from"))
    relationships.append(_rel_row(e11_21_id, sn_normal_id, "Maps to"))

    high_fanout_snomed_id = next_id()
    concepts.append(
        _concept_row(
            high_fanout_snomed_id,
            "Disease (disorder)",
            "64572001",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
        )
    )
    for fid in filler_ids[:150]:
        relationships.append(_rel_row(high_fanout_snomed_id, fid, "Mapped from"))
        relationships.append(_rel_row(fid, high_fanout_snomed_id, "Maps to"))

    # An invalid/deprecated row that must be filtered out.
    deprecated_id = next_id()
    concepts.append(_concept_row(deprecated_id, "Deprecated code", "Q999", invalid_reason="D"))

    athena_dir = tmp_path / "athena"
    athena_dir.mkdir()
    _write_tsv(athena_dir / "CONCEPT.csv", _OMOP_COLUMNS, concepts)
    _write_tsv(athena_dir / "CONCEPT_RELATIONSHIP.csv", _REL_COLUMNS, relationships)

    ccsr_csv = tmp_path / "DXCCSR.csv"
    _write_ccsr(ccsr_csv, [("E11", "END002", "Diabetes mellitus without complication")])

    db_path = tmp_path / "vocab.duckdb"
    paths = VocabPaths(athena_dir=athena_dir, ccsr_csv=ccsr_csv)
    report = build_vocab_db(paths, db_path, fanout_threshold=100)

    yield MiniVocab(
        db_path=db_path,
        report=report,
        e11_id=e11_id,
        e11_2_id=e11_2_id,
        e11_21_id=e11_21_id,
        high_fanout_snomed_id=high_fanout_snomed_id,
    )


def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_ccsr(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", newline="") as f:
        f.write("'ICD-10-CM CODE','CCSR CATEGORY 1','CCSR CATEGORY 1 DESCRIPTION'\n")
        for code, category, desc in rows:
            f.write(f"'{code}','{category}',\"{desc}\"\n")


@pytest.fixture
def mini_vocab_con(mini_vocab: MiniVocab) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a read-only connection to the built miniature vocabulary database.

    :param mini_vocab: The built fixture database.
    :yields: A DuckDB connection, closed on teardown.
    """
    con = duckdb.connect(str(mini_vocab.db_path), read_only=True)
    try:
        yield con
    finally:
        con.close()
