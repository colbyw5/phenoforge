"""BM25 lexical retrieval over ICD-10-CM concept names.

One component of the hybrid retrieval design (DESIGN.md D8). Dense
(paraphrase) and graph-traversal components are not yet implemented; this
covers exact strings, code fragments, and rare literal terms today.
"""

from __future__ import annotations

import re

import duckdb
from rank_bm25 import BM25Okapi

from phenoforge.engine.models import ConceptWithProvenance, ProvenanceTier, UnmappableTerm

_ICD10CM = "ICD10CM"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """A BM25 index over ICD-10-CM ``concept_name`` text.

    Construction loads and tokenizes every ICD-10-CM concept name, so build
    one instance per process (e.g. once at MCP server startup) rather than
    per query.

    :ivar concept_ids: OMOP concept ids, parallel to the internal corpus.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        """Build the index from the vocabulary database.

        :param con: Open connection to ``vocab.duckdb``.
        """
        rows = con.execute(
            "SELECT concept_id, concept_code, concept_name, domain_id, vocabulary_id "
            "FROM concept WHERE vocabulary_id = ?",
            [_ICD10CM],
        ).fetchall()
        self._rows = rows
        corpus = [_tokenize(row[2]) for row in rows]
        self._bm25 = BM25Okapi(corpus)

    def search(
        self, query: str, k: int = 10, min_score: float = 0.0
    ) -> tuple[list[ConceptWithProvenance], UnmappableTerm | None]:
        """Search for concepts whose names best match a free-text query.

        :param query: Free-text search string.
        :param k: Maximum number of results to return.
        :param min_score: Minimum BM25 score for a result to be included.
        :returns: A tuple of (matching concepts tagged
            :attr:`~phenoforge.engine.models.ProvenanceTier.GENERATED`,
            ordered by descending score; an :class:`~phenoforge.engine.models.UnmappableTerm`
            if nothing scored above ``min_score``, else ``None``).
        :rtype: tuple[list[ConceptWithProvenance], UnmappableTerm | None]
        """
        tokens = _tokenize(query)
        if not tokens or not self._rows:
            return [], UnmappableTerm(term=query, reason="no BM25 match above threshold")

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:k]
        results = [
            ConceptWithProvenance(
                concept_id=self._rows[i][0],
                concept_code=self._rows[i][1],
                concept_name=self._rows[i][2],
                domain_id=self._rows[i][3],
                vocabulary_id=self._rows[i][4],
                tier=ProvenanceTier.GENERATED,
                source=f"bm25:{query}",
            )
            for i, score in ranked
            if score > min_score
        ]
        if not results:
            return [], UnmappableTerm(term=query, reason="no BM25 match above threshold")
        return results, None
