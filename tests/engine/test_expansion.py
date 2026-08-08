"""Tests for phenoforge.engine.expansion."""

from __future__ import annotations

import duckdb

from phenoforge.engine.expansion import ancestor_chain_ids, expand_ancestors, expand_descendants
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


def test_expand_ancestors_from_leaf(con: duckdb.DuckDBPyConnection) -> None:
    results = expand_ancestors(con, "E11.21")
    codes = [c.concept_code for c in results]
    assert codes == ["E11.2", "E11"]
    assert all(c.tier is ProvenanceTier.GENERATED for c in results)
    assert all(c.source == "hierarchy_ancestors:E11.21" for c in results)


def test_expand_ancestors_ordered_by_depth(con: duckdb.DuckDBPyConnection) -> None:
    results = expand_ancestors(con, "E11.21")
    codes_in_order = [c.concept_code for c in results]
    assert codes_in_order.index("E11.2") < codes_in_order.index("E11")


def test_expand_ancestors_root_has_none(con: duckdb.DuckDBPyConnection) -> None:
    assert expand_ancestors(con, "E11") == []


def test_expand_ancestors_unknown_code_returns_empty(con: duckdb.DuckDBPyConnection) -> None:
    assert expand_ancestors(con, "NOPE") == []


def test_ancestor_chain_ids_includes_self_then_ancestors(
    con: duckdb.DuckDBPyConnection,
) -> None:
    chain = ancestor_chain_ids(con, "E11.21")
    codes = [
        con.execute("SELECT concept_code FROM concept WHERE concept_id = ?", [cid]).fetchone()[0]
        for cid in chain
    ]
    assert codes == ["E11.21", "E11.2", "E11"]


def test_ancestor_chain_ids_unknown_code_returns_empty(con: duckdb.DuckDBPyConnection) -> None:
    assert ancestor_chain_ids(con, "NOPE") == []
