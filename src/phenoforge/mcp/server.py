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
from pathlib import Path

import duckdb
from mcp.server.fastmcp import FastMCP

from phenoforge.engine.db import DEFAULT_VOCAB_DB_PATH, connect
from phenoforge.engine.expansion import expand_descendants
from phenoforge.engine.lookup import lookup_by_code
from phenoforge.engine.models import Concept, ConceptSet
from phenoforge.engine.retrieval import BM25Retriever

mcp = FastMCP("phenoforge")

_db_path: Path = DEFAULT_VOCAB_DB_PATH
_con: duckdb.DuckDBPyConnection | None = None
_retriever: BM25Retriever | None = None


def configure(db_path: Path) -> None:
    """Point the server at a different vocabulary database.

    Resets any already-open connection and cached retriever so the next
    tool call rebuilds them against the new path. Intended for tests; the
    console entry point never needs to call this.

    :param db_path: Path to a DuckDB database built by
        ``scripts/load_vocab.py``.
    """
    global _db_path, _con, _retriever
    _db_path = db_path
    _con = None
    _retriever = None


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

    Lexical (BM25) search only — it matches on shared words, not paraphrase
    or clinical synonymy, so exact terminology in the query works far better
    than a loose description. Every returned concept is tagged ``generated``
    provenance and should be treated as ungrounded until a human confirms it.

    :param query: Free-text search string, e.g. ``"diabetic nephropathy"``.
    :param k: Maximum number of results to return.
    :returns: Matching concepts ordered by relevance. If nothing scores
        above the match threshold, ``concepts`` is empty and ``unmappable``
        explains why.
    :rtype: ConceptSet
    """
    results, unmappable = _get_retriever().search(query, k=k)
    return ConceptSet(concepts=results, unmappable=[unmappable] if unmappable else [])


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
