"""Tests for phenoforge.engine.expansion."""

from __future__ import annotations

import duckdb

from phenoforge.engine.expansion import expand_descendants
from phenoforge.engine.models import ProvenanceTier


def test_expand_descendants_from_root(con: duckdb.DuckDBPyConnection) -> None:
    results = expand_descendants(con, "E11")
    codes = {c.concept_code for c in results}
    assert codes == {"E11.2", "E11.21"}
    assert all(c.tier is ProvenanceTier.GENERATED for c in results)
    assert all(c.source == "hierarchy_expansion:E11" for c in results)


def test_expand_descendants_ordered_by_depth(con: duckdb.DuckDBPyConnection) -> None:
    results = expand_descendants(con, "E11")
    codes_in_order = [c.concept_code for c in results]
    assert codes_in_order.index("E11.2") < codes_in_order.index("E11.21")


def test_expand_leaf_has_no_descendants(con: duckdb.DuckDBPyConnection) -> None:
    assert expand_descendants(con, "E11.21") == []


def test_expand_unknown_code_returns_empty(con: duckdb.DuckDBPyConnection) -> None:
    assert expand_descendants(con, "NOPE") == []
