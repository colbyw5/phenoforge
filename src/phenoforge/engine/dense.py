"""Dense (semantic) retrieval over ICD-10-CM concept names via a LanceDB index.

Third of three retrieval components alongside BM25 (retrieval.py) and graph
traversal (expansion.py); covers paraphrase/synonymy that BM25 misses since
it matches on meaning rather than shared words.

The embedding function is injectable (``EmbedFn``) so tests never need to
download a real model — see ``tests/engine/fake_embedder.py``. Production
code gets a real embedder via :func:`default_embedder`, which lazily imports
``sentence_transformers`` so importing this module never requires it.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import lancedb

from phenoforge.engine.models import ConceptWithProvenance, ProvenanceTier, UnmappableTerm

_ICD10CM = "ICD10CM"
_TABLE_NAME = "concepts"
_BIOLORD_MODEL = "FremyCompany/BioLORD-2023"

EmbedFn = Callable[[list[str]], list[list[float]]]


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalize a vector so LanceDB's L2 distance maps cleanly to cosine similarity.

    Applied to every embedding this module produces or receives — never
    trusts an ``EmbedFn`` implementation to have already normalized, since a
    silently-unnormalized embedder (e.g. raw ``sentence-transformers``
    output without ``normalize_embeddings=True``) breaks every similarity
    threshold without raising: this shipped once and was only caught by a
    manual smoke test against the real model, because
    :mod:`tests.engine.fake_embedder` already normalized and so never
    exercised the bug.

    :param vector: Raw embedding vector.
    :returns: The same vector scaled to unit L2 norm (unchanged if already
        zero-length).
    :rtype: list[float]
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def default_embedder() -> EmbedFn:
    """Build an embedding function backed by real BioLORD-2023.

    Imports ``sentence_transformers`` and loads the model on call, not at
    module import time, so ``import phenoforge.engine.dense`` never requires
    the dependency to be installed or the model to be downloadable.

    :returns: A function embedding a batch of texts into L2-normalized
        BioLORD-2023 vectors (see :data:`EmbedFn`).
    :rtype: EmbedFn
    :raises ImportError: If ``sentence-transformers`` is not installed.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_BIOLORD_MODEL)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    return embed


def _embed_concept_rows(
    con: duckdb.DuckDBPyConnection, embed_fn: EmbedFn, batch_size: int = 64
) -> list[dict[str, Any]]:
    """Fetch every ICD-10-CM concept and embed its name in batches.

    Shared by :class:`DenseRetriever`'s in-memory fallback and
    ``scripts/build_index.py``, so the two never duplicate LanceDB-record
    construction logic.

    :param con: Open connection to ``vocab.duckdb``.
    :param embed_fn: Embedding function to apply to each batch of concept names.
    :param batch_size: Number of concept names embedded per call to ``embed_fn``.
    :returns: One record per concept, ready to write into a LanceDB table.
    :rtype: list[dict[str, Any]]
    """
    rows = con.execute(
        "SELECT concept_id, concept_code, concept_name, domain_id, vocabulary_id "
        "FROM concept WHERE vocabulary_id = ?",
        [_ICD10CM],
    ).fetchall()

    records: list[dict[str, Any]] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        vectors = embed_fn([row[2] for row in batch])
        for row, vector in zip(batch, vectors, strict=True):
            vector = _normalize(vector)
            records.append(
                {
                    "concept_id": row[0],
                    "concept_code": row[1],
                    "concept_name": row[2],
                    "domain_id": row[3],
                    "vocabulary_id": row[4],
                    "vector": vector,
                }
            )
    return records


class DenseRetriever:
    """A LanceDB-backed semantic index over ICD-10-CM ``concept_name`` text.

    Two modes: if ``index_path`` is given and already exists, opens the
    persisted table built by ``scripts/build_index.py`` (the production
    path — LanceDB's value is a persisted, memory-mapped index, unlike
    :class:`~phenoforge.engine.retrieval.BM25Retriever`, which is cheap
    enough to rebuild every process start). If not, builds an ephemeral
    in-memory table via :func:`_embed_concept_rows` (the test path, with an
    injected fake embedder — no model download or prebuilt artifact needed).
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        embed_fn: EmbedFn | None = None,
        index_path: Path | None = None,
    ) -> None:
        """Open or build the semantic index.

        :param con: Open connection to ``vocab.duckdb``.
        :param embed_fn: Embedding function. Defaults to
            :func:`default_embedder` (real BioLORD-2023) if not given —
            always needed, even when opening a prebuilt table, to embed
            query text at search time.
        :param index_path: Path to a persisted LanceDB directory built by
            ``scripts/build_index.py``. If missing or not given, an
            ephemeral in-memory index is built instead.
        """
        self._embed_fn = embed_fn or default_embedder()

        if index_path is not None and index_path.exists():
            db = lancedb.connect(str(index_path))
            self._table = db.open_table(_TABLE_NAME)
        else:
            db = lancedb.connect(tempfile.mkdtemp())
            records = _embed_concept_rows(con, self._embed_fn)
            self._table = db.create_table(_TABLE_NAME, data=records)

    def search(
        self, query: str, k: int = 10, min_score: float = 0.0
    ) -> tuple[list[ConceptWithProvenance], UnmappableTerm | None]:
        """Search for concepts whose names are semantically closest to a query.

        :param query: Free-text search string.
        :param k: Maximum number of results to return.
        :param min_score: Minimum cosine similarity (derived from L2 distance
            over the embedder's vectors) for a result to be included.
        :returns: A tuple of (matching concepts tagged
            :attr:`~phenoforge.engine.models.ProvenanceTier.GENERATED`,
            ordered by descending similarity; an
            :class:`~phenoforge.engine.models.UnmappableTerm` if nothing
            scored above ``min_score``, else ``None``).
        :rtype: tuple[list[ConceptWithProvenance], UnmappableTerm | None]
        """
        query_vector = _normalize(self._embed_fn([query])[0])
        rows = self._table.search(query_vector).limit(k).to_list()

        results = []
        for row in rows:
            similarity = 1.0 - (row["_distance"] / 2.0)
            if similarity <= min_score:
                continue
            results.append(
                ConceptWithProvenance(
                    concept_id=row["concept_id"],
                    concept_code=row["concept_code"],
                    concept_name=row["concept_name"],
                    domain_id=row["domain_id"],
                    vocabulary_id=row["vocabulary_id"],
                    tier=ProvenanceTier.GENERATED,
                    source=f"dense:{query}",
                )
            )

        if not results:
            return [], UnmappableTerm(term=query, reason="no dense match above threshold")
        return results, None
