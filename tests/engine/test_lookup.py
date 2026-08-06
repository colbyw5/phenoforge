"""Tests for phenoforge.engine.lookup."""

from __future__ import annotations

import duckdb

from phenoforge.engine.lookup import lookup_by_code


def test_lookup_known_code(con: duckdb.DuckDBPyConnection) -> None:
    concept = lookup_by_code(con, "E11.21")
    assert concept is not None
    assert concept.concept_name == "Type 2 diabetes mellitus with diabetic nephropathy"
    assert concept.vocabulary_id == "ICD10CM"


def test_lookup_unknown_code_returns_none(con: duckdb.DuckDBPyConnection) -> None:
    assert lookup_by_code(con, "NOPE") is None


def test_lookup_does_not_return_snomed(con: duckdb.DuckDBPyConnection) -> None:
    assert lookup_by_code(con, "90721000") is None
