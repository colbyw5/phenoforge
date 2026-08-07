"""Thin MCP server exposing the engine's lookup, expansion, and retrieval.

Tool registration only — no business logic (AGENTS.md). Every tool here
delegates directly to a ``phenoforge.engine`` function; if a change requires
new logic rather than a new call, it belongs in ``engine``, not here.

Transport is flag-controlled: stdio by default, matching what Claude Desktop
and other MCP clients expect today with no hosting step, but switching to
HTTP later only requires a different --transport value, not a rewrite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
from mcp.server.fastmcp import FastMCP

from phenoforge.engine.curated import find_matching_cohort, load_curated_concept_set
from phenoforge.engine.db import DEFAULT_VOCAB_DB_PATH, connect
from phenoforge.engine.dense import DenseRetriever, EmbedFn
from phenoforge.engine.expansion import expand_descendants
from phenoforge.engine.hybrid import reciprocal_rank_fusion
from phenoforge.engine.lookup import lookup_by_code
from phenoforge.engine.models import Concept, ConceptSet, UnmappableTerm
from phenoforge.engine.retrieval import BM25Retriever

mcp = FastMCP("phenoforge")

_DEFAULT_LIBRARY_DIR = Path("data/phenotype_library")
_DEFAULT_INDEX_PATH = Path("data/concept_index.lance")

_db_path: Path = DEFAULT_VOCAB_DB_PATH
_library_dir: Path = _DEFAULT_LIBRARY_DIR
_index_path: Path = _DEFAULT_INDEX_PATH
_dense_embed_fn: EmbedFn | None = None
_con: duckdb.DuckDBPyConnection | None = None
_retriever: BM25Retriever | None = None
_dense_retriever: DenseRetriever | None = None


def configure(
    db_path: Path,
    *,
    library_dir: Path | None = None,
    index_path: Path | None = None,
    dense_embed_fn: EmbedFn | None = None,
) -> None:
    """Point the server at a different vocabulary database, phenotype library, and/or dense index.

    Resets any already-open connection and cached retrievers so the next
    tool call rebuilds them against the new path(s). Intended for tests;
    the console entry point never needs to call this.

    :param db_path: Path to a DuckDB database built by
        ``scripts/load_vocab.py``.
    :param library_dir: Path to a directory of fetched cohort JSON files
        built by ``scripts/fetch_phenotype_library.py``. Left unchanged if
        not given.
    :param index_path: Path to a LanceDB index built by
        ``scripts/build_index.py``. Left unchanged if not given.
    :param dense_embed_fn: Embedding function for the dense retriever.
        Tests pass a fake embedder here so they never download BioLORD-2023;
        left unchanged if not given (production leaves this ``None``, which
        makes :class:`~phenoforge.engine.dense.DenseRetriever` default to
        the real model).
    """
    global _db_path, _library_dir, _index_path, _dense_embed_fn
    global _con, _retriever, _dense_retriever
    _db_path = db_path
    _con = None
    _retriever = None
    _dense_retriever = None
    if library_dir is not None:
        _library_dir = library_dir
    if index_path is not None:
        _index_path = index_path
    if dense_embed_fn is not None:
        _dense_embed_fn = dense_embed_fn


def _get_connection() -> duckdb.DuckDBPyConnection:
    """Return a lazily-opened, process-wide connection to ``vocab.duckdb``.

    :returns: An open, read-only DuckDB connection.
    :rtype: duckdb.DuckDBPyConnection
    """
    global _con
    if _con is None:
        _con = connect(_db_path)
    return _con


def _get_retriever() -> BM25Retriever:
    """Return a lazily-built, process-wide BM25 retriever.

    Built once per process rather than per call: constructing it tokenizes
    every ICD-10-CM concept name.

    :returns: A BM25 index over ICD-10-CM concept names.
    :rtype: BM25Retriever
    """
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever(_get_connection())
    return _retriever


def _get_dense_retriever() -> DenseRetriever | None:
    """Return a lazily-built, process-wide dense retriever, if an index exists.

    Returns ``None`` rather than raising when ``_index_path`` hasn't been
    built yet (``scripts/build_index.py`` hasn't run), so ``search_concepts``
    can degrade gracefully to BM25-only instead of requiring the index.

    :returns: A dense index over ICD-10-CM concept names, or ``None``.
    :rtype: DenseRetriever | None
    """
    global _dense_retriever
    if _dense_retriever is None and _index_path.exists():
        _dense_retriever = DenseRetriever(
            _get_connection(), embed_fn=_dense_embed_fn, index_path=_index_path
        )
    return _dense_retriever


@mcp.tool()
def lookup_concept(concept_code: str) -> Concept | None:
    """Look up a single ICD-10-CM concept by its exact billing code.

    Use this when the caller already has a specific code, e.g. ``"E11.21"``,
    and wants its name and OMOP metadata. Not a search tool — for free-text
    clinical terms use ``search_concepts`` instead.

    :param concept_code: An exact ICD-10-CM code, e.g. ``"E11.21"``.
    :returns: The matching concept, or ``None`` if no ICD-10-CM concept has
        that code.
    :rtype: Concept | None
    """
    return lookup_by_code(_get_connection(), concept_code)


@mcp.tool()
def expand_hierarchy(seed_code: str) -> ConceptSet:
    """Expand an ICD-10-CM code to every code beneath it in the billing hierarchy.

    For example, expanding ``"E11"`` (Type 2 diabetes mellitus) returns all
    of its more specific subtypes, such as ``"E11.21"`` (with diabetic
    nephropathy). Every returned concept is tagged ``generated`` provenance:
    it is a structural consequence of the vocabulary hierarchy, not a
    clinically validated inclusion, and should be treated as ungrounded
    until a human confirms it belongs in the target population.

    :param seed_code: An exact ICD-10-CM code to expand from, e.g. ``"E11"``.
    :returns: All descendant concepts, or an empty set if ``seed_code`` does
        not exist or has no descendants.
    :rtype: ConceptSet
    """
    return ConceptSet(concepts=expand_descendants(_get_connection(), seed_code))


@mcp.tool()
def search_concepts(query: str, k: int = 10) -> ConceptSet:
    """Search ICD-10-CM concept names for a free-text clinical term or phrase.

    Combines lexical (BM25) and semantic (embedding) matching, fused by
    reciprocal rank, so paraphrased or loosely-worded descriptions (e.g.
    "sugar disease" for diabetes) are found even without shared exact
    wording — always prefer this over guessing at exact terminology
    yourself. Every returned concept is tagged ``generated`` provenance and
    should be treated as ungrounded until a human confirms it.

    :param query: Free-text search string, e.g. ``"diabetic nephropathy"``.
    :param k: Maximum number of results to return.
    :returns: Fused, deduplicated results ordered by combined relevance. If
        nothing matches, ``concepts`` is empty and ``unmappable`` explains why.
    :rtype: ConceptSet
    """
    bm25_results, bm25_unmappable = _get_retriever().search(query, k=k)
    dense_retriever = _get_dense_retriever()
    if dense_retriever is not None:
        dense_results, _ = dense_retriever.search(query, k=k)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=k)
    else:
        fused = bm25_results

    if not fused:
        unmappable = bm25_unmappable or UnmappableTerm(
            term=query, reason="no match above threshold"
        )
        return ConceptSet(unmappable=[unmappable])
    return ConceptSet(concepts=fused)


@mcp.tool()
def find_curated_definition(query: str) -> ConceptSet:
    """Search the OHDSI Phenotype Library for a validated cohort definition.

    Use this FIRST for any population description that plausibly matches a
    peer-reviewed phenotype (e.g. "type 2 diabetes", "diabetic ketoacidosis")
    — a match here is ``curated`` provenance, safe to use as-is citing the
    cohort id. Only fall back to ``search_concepts``/``expand_hierarchy`` if
    nothing matches; those return ``generated`` provenance requiring human
    confirmation. This demo bundles a small, hand-picked set of
    diabetes/kidney-related cohorts, not the full library — a miss here
    does not mean no curated definition exists.

    :param query: Free-text population description.
    :returns: The best-matching cohort's resolved ICD-10-CM concepts tagged
        ``curated``, or an empty set with an explanatory ``unmappable``
        entry if nothing in the bundled set matches or the library has not
        been fetched yet.
    :rtype: ConceptSet
    """
    manifest_path = _library_dir / "manifest.json"
    if not manifest_path.exists():
        return ConceptSet(
            unmappable=[
                UnmappableTerm(
                    term=query,
                    reason=(
                        f"no phenotype library at {_library_dir} — run "
                        "scripts/fetch_phenotype_library.py first"
                    ),
                )
            ]
        )

    manifest = json.loads(manifest_path.read_text())
    cohort_id = find_matching_cohort(query, manifest)
    if cohort_id is None:
        return ConceptSet(
            unmappable=[UnmappableTerm(term=query, reason="no bundled cohort matches this query")]
        )
    return load_curated_concept_set(_get_connection(), cohort_id, _library_dir)


def main() -> None:
    """Entry point for the ``phenoforge-mcp`` console script.

    :raises FileNotFoundError: If ``data/vocab.duckdb`` has not been built
        yet via ``scripts/load_vocab.py``.
    """
    parser = argparse.ArgumentParser(description="phenoforge MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport. stdio for clients like Claude Desktop; HTTP is a flag, not a rewrite.",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
