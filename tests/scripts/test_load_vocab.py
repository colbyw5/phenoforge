"""Tests for scripts/load_vocab.py against the miniature fixture vocabulary."""

from __future__ import annotations

import duckdb
import pytest

from tests.scripts.conftest import MiniVocab


def test_loads_icd10cm_and_snomed_concepts(mini_vocab: MiniVocab) -> None:
    assert mini_vocab.report.concept_counts["ICD10CM"] > 0
    assert mini_vocab.report.concept_counts["SNOMED"] > 0


def test_deprecated_concepts_are_excluded(mini_vocab_con: duckdb.DuckDBPyConnection) -> None:
    row = mini_vocab_con.execute(
        "SELECT COUNT(*) FROM concept WHERE concept_code = 'Q999'"
    ).fetchone()
    assert row is not None
    assert row[0] == 0


def test_concept_code_lookup(mini_vocab_con: duckdb.DuckDBPyConnection) -> None:
    row = mini_vocab_con.execute(
        "SELECT concept_id, concept_name FROM concept "
        "WHERE vocabulary_id = 'ICD10CM' AND concept_code = 'E11.21'"
    ).fetchone()
    assert row is not None
    assert row[1] == "Type 2 diabetes mellitus with diabetic nephropathy"


def test_descendant_expansion_recursive_cte(
    mini_vocab: MiniVocab, mini_vocab_con: duckdb.DuckDBPyConnection
) -> None:
    """Verifies the D1 traversal shape: E11.21 -> {E11.2, E11} via 'Is a'.

    The real Athena download has a redundant direct E11.21 -> E11 edge
    alongside the E11.21 -> E11.2 -> E11 chain (multiple direct parents, not
    a cycle) — the visited-path guard is required so a concept reachable by
    more than one path is reported once, at its shortest depth, per D1.
    """
    rows = mini_vocab_con.execute(
        """
        WITH RECURSIVE ancestors(concept_id, depth, path) AS (
            SELECT ?, 0, [?]
            UNION ALL
            SELECT r.concept_id_2, a.depth + 1, list_append(a.path, r.concept_id_2)
            FROM concept_relationship r
            JOIN ancestors a ON a.concept_id = r.concept_id_1
            WHERE r.relationship_id = 'Is a'
              AND NOT list_contains(a.path, r.concept_id_2)
        )
        SELECT c.concept_code, MIN(a.depth) AS depth
        FROM ancestors a
        JOIN concept c ON c.concept_id = a.concept_id
        WHERE a.depth > 0
        GROUP BY c.concept_code
        ORDER BY depth
        """,
        [mini_vocab.e11_21_id, mini_vocab.e11_21_id],
    ).fetchall()
    codes = [r[0] for r in rows]
    assert codes == ["E11.2", "E11"]


def test_ccsr_loaded(mini_vocab: MiniVocab, mini_vocab_con: duckdb.DuckDBPyConnection) -> None:
    assert mini_vocab.report.ccsr_rows == 1
    row = mini_vocab_con.execute(
        "SELECT cc.category_name FROM concept_ccsr c "
        "JOIN ccsr_category cc ON cc.category_id = c.category_id "
        "WHERE c.concept_code = 'E11'"
    ).fetchone()
    assert row is not None
    assert row[0] == "Diabetes mellitus without complication"


def test_high_fanout_snomed_concept_is_flagged(mini_vocab: MiniVocab) -> None:
    assert mini_vocab.high_fanout_snomed_id in mini_vocab.report.high_fanout_snomed_concepts


def test_missing_concept_csv_raises(tmp_path: object) -> None:
    from pathlib import Path

    from scripts.load_vocab import VocabPaths, build_vocab_db

    athena_dir = Path(str(tmp_path)) / "empty"
    athena_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_vocab_db(VocabPaths(athena_dir=athena_dir), Path(str(tmp_path)) / "out.duckdb")
