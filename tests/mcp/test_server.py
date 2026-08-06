"""Smoke tests: each MCP tool delegates to the engine correctly.

Not testing the transport itself (stdio/SSE plumbing is FastMCP's job) —
only that the thin wrappers in phenoforge.mcp.server call through to the
right engine function and shape the result as documented in AGENTS.md
(never a bare code list).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import phenoforge.mcp.server as server
from tests.scripts.conftest import MiniVocab


@pytest.fixture(autouse=True)
def _use_mini_vocab(mini_vocab: MiniVocab) -> Iterator[None]:
    server._con = None
    server._retriever = None
    import phenoforge.engine.db as db_module

    original = db_module.connect

    def fake_connect(_path: object = None) -> object:
        return original(mini_vocab.db_path)

    db_module.connect = fake_connect  # type: ignore[assignment]
    server.connect = fake_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        db_module.connect = original
        server.connect = original
        server._con = None
        server._retriever = None


def test_lookup_concept_tool(mini_vocab: MiniVocab) -> None:
    concept = server.lookup_concept("E11.21")
    assert concept is not None
    assert concept.concept_code == "E11.21"


def test_expand_hierarchy_tool(mini_vocab: MiniVocab) -> None:
    result = server.expand_hierarchy("E11")
    codes = {c.concept_code for c in result.concepts}
    assert codes == {"E11.2", "E11.21"}
    assert result.unmappable == []


def test_search_concepts_tool(mini_vocab: MiniVocab) -> None:
    result = server.search_concepts("diabetic nephropathy", k=5)
    codes = {c.concept_code for c in result.concepts}
    assert "E11.21" in codes


def test_search_concepts_tool_unmappable(mini_vocab: MiniVocab) -> None:
    result = server.search_concepts("xyzzy plugh quux")
    assert result.concepts == []
    assert len(result.unmappable) == 1
