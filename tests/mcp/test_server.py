"""Smoke tests: each MCP tool delegates to the engine correctly.

Not testing the transport itself (stdio/SSE plumbing is FastMCP's job) —
only that the thin wrappers in phenoforge.mcp.server call through to the
right engine function and shape the result as documented in AGENTS.md
(never a bare code list).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

import phenoforge.mcp.server as server
from phenoforge.engine.models import ProvenanceTier
from tests.scripts.conftest import MiniVocab


@pytest.fixture(autouse=True)
def _use_mini_vocab(mini_vocab: MiniVocab, tmp_path: Path) -> Iterator[None]:
    # Explicit nonexistent library_dir/index_path rather than relying on the
    # untouched defaults (data/phenotype_library, data/concept_index.lance)
    # not existing on disk — both are real, populated directories in any
    # checkout where the optional setup steps have been run.
    server.configure(
        mini_vocab.db_path,
        library_dir=tmp_path / "no_such_library",
        index_path=tmp_path / "no_such_index.lance",
    )
    yield
    server.configure(server.DEFAULT_VOCAB_DB_PATH)


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


def test_find_curated_definition_no_library_configured(mini_vocab: MiniVocab) -> None:
    result = server.find_curated_definition("type 2 diabetes")

    assert result.concepts == []
    assert len(result.unmappable) == 1
    assert "fetch_phenotype_library" in result.unmappable[0].reason


def test_find_curated_definition_tool(mini_vocab: MiniVocab, tmp_path: Path) -> None:
    library_dir = tmp_path / "phenotype_library"
    library_dir.mkdir()
    (library_dir / "manifest.json").write_text(
        json.dumps({"1": "Diabetic nephropathy demo cohort"})
    )
    (library_dir / "1.json").write_text(
        json.dumps(
            {
                "ConceptSets": [
                    {
                        "id": 0,
                        "name": "test",
                        "expression": {
                            "items": [
                                {
                                    "concept": {
                                        "CONCEPT_ID": mini_vocab.sn_normal_id,
                                        "CONCEPT_CODE": "90721000",
                                        "VOCABULARY_ID": "SNOMED",
                                        "CONCEPT_NAME": "irrelevant",
                                        "DOMAIN_ID": "Condition",
                                    },
                                    "isExcluded": False,
                                    "includeDescendants": False,
                                }
                            ]
                        },
                    }
                ]
            }
        )
    )
    server.configure(mini_vocab.db_path, library_dir=library_dir)

    result = server.find_curated_definition("diabetic nephropathy demo")

    assert len(result.concepts) == 1
    assert result.concepts[0].concept_code == "E11.21"
    assert result.concepts[0].tier is ProvenanceTier.CURATED


def test_find_curated_definition_no_match(mini_vocab: MiniVocab, tmp_path: Path) -> None:
    library_dir = tmp_path / "phenotype_library"
    library_dir.mkdir()
    (library_dir / "manifest.json").write_text(json.dumps({"1": "Type 2 diabetes mellitus"}))
    server.configure(mini_vocab.db_path, library_dir=library_dir)

    result = server.find_curated_definition("xyzzy plugh quux")

    assert result.concepts == []
    assert len(result.unmappable) == 1


def test_find_curated_definition_stale_manifest_entry(
    mini_vocab: MiniVocab, tmp_path: Path
) -> None:
    """manifest.json lists a cohort whose JSON file is missing (stale entry,
    partial fetch) — this must degrade to unmappable, not crash."""
    library_dir = tmp_path / "phenotype_library"
    library_dir.mkdir()
    (library_dir / "manifest.json").write_text(json.dumps({"1": "Type 2 diabetes mellitus"}))
    # deliberately no "1.json" written
    server.configure(mini_vocab.db_path, library_dir=library_dir)

    result = server.find_curated_definition("type 2 diabetes mellitus")

    assert result.concepts == []
    assert len(result.unmappable) == 1
    assert "fetch_phenotype_library" in result.unmappable[0].reason


def test_configure_explicit_none_resets_to_default(mini_vocab: MiniVocab, tmp_path: Path) -> None:
    from tests.engine.fake_embedder import fake_embed_fn

    server.configure(mini_vocab.db_path, dense_embed_fn=fake_embed_fn)
    assert server._dense_embed_fn is fake_embed_fn

    server.configure(mini_vocab.db_path, dense_embed_fn=None)
    assert server._dense_embed_fn is None


def test_configure_omitted_leaves_unchanged(mini_vocab: MiniVocab) -> None:
    from tests.engine.fake_embedder import fake_embed_fn

    server.configure(mini_vocab.db_path, dense_embed_fn=fake_embed_fn)
    server.configure(mini_vocab.db_path)  # dense_embed_fn omitted
    assert server._dense_embed_fn is fake_embed_fn


def test_search_concepts_fused_with_dense_index(mini_vocab: MiniVocab, tmp_path: Path) -> None:
    from scripts.build_index import build_dense_index
    from tests.engine.fake_embedder import fake_embed_fn

    con = duckdb.connect(str(mini_vocab.db_path), read_only=True)
    try:
        index_path = tmp_path / "concept_index.lance"
        build_dense_index(con, index_path, embed_fn=fake_embed_fn)
    finally:
        con.close()

    server.configure(mini_vocab.db_path, index_path=index_path, dense_embed_fn=fake_embed_fn)

    result = server.search_concepts("diabetic nephropathy", k=5)
    codes = {c.concept_code for c in result.concepts}
    assert "E11.21" in codes
    sources = {c.source.split(":", 1)[0] for c in result.concepts}
    assert sources & {"bm25", "dense"}
