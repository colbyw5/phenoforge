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
_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
_MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
_MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

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
        vectors: list[list[float]] = model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
        return vectors

    return embed


def _cls_pooled_embedder(model_name: str, max_length: int = 64) -> EmbedFn:
    """Build an ``EmbedFn`` using a raw ``transformers`` checkpoint's [CLS] embedding.

    Both SapBERT and MedCPT ship as plain ``transformers`` checkpoints (no
    ``sentence-transformers`` ``modules.json``), and each model card's own
    reference usage takes the last hidden state's [CLS] token (index 0) as
    the embedding, not mean pooling — ``sentence-transformers``'s
    auto-wrapping for a bare checkpoint defaults to mean pooling, which
    would silently not match either model's documented usage. Loading via
    plain ``transformers`` instead makes the pooling choice explicit and
    correct for both.

    :param model_name: HuggingFace repo id of the checkpoint.
    :param max_length: Token truncation length. ``64`` matches the shorter
        of the two models' own documented defaults (MedCPT's query
        encoder) — these inputs are concept names and short query terms,
        never full articles, so truncation in practice should rarely bind.
    :returns: A function embedding a batch of texts into (unnormalized)
        [CLS] vectors — callers normalize separately, same as every other
        embedder in this module.
    :rtype: EmbedFn
    :raises ImportError: If ``transformers``/``torch`` is not installed.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def embed(texts: list[str]) -> list[list[float]]:
        encoded = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            cls_vectors = model(**encoded).last_hidden_state[:, 0, :]
        result: list[list[float]] = cls_vectors.tolist()
        return result

    return embed


def sapbert_embedder() -> EmbedFn:
    """Build an embedding function backed by SapBERT-from-PubMedBERT-fulltext.

    Benchmarking alternative to :func:`default_embedder` (BioLORD-2023) —
    see ``scripts/compare_encoders.py``. Symmetric, like BioLORD: the same
    function embeds both concept names (indexing) and query text.

    :returns: A function embedding a batch of texts into [CLS] vectors (see
        :data:`EmbedFn`). Callers that feed this into :class:`DenseRetriever`
        get L2 normalization for free; raw output here is unnormalized.
    :rtype: EmbedFn
    :raises ImportError: If ``transformers``/``torch`` is not installed.
    """
    return _cls_pooled_embedder(_SAPBERT_MODEL, max_length=25)


def medcpt_embedders() -> tuple[EmbedFn, EmbedFn]:
    """Build MedCPT's asymmetric article/query embedding functions.

    Unlike BioLORD/SapBERT, MedCPT ships as two separate encoders — an
    Article Encoder (meant for the text being indexed) and a Query Encoder
    (meant for short search queries) — trained specifically for
    query-to-article retrieval, not as one symmetric encoder. Collapsing
    both into a single function (e.g. using the article encoder for
    queries too) would misrepresent how the model is meant to be used and
    bias the comparison against it; see ``scripts/compare_encoders.py``,
    which passes the article function for indexing and the query function
    for search via :class:`DenseRetriever`'s ``query_embed_fn``.

    :returns: ``(article_embed_fn, query_embed_fn)``.
    :rtype: tuple[EmbedFn, EmbedFn]
    :raises ImportError: If ``transformers``/``torch`` is not installed.
    """
    article_fn = _cls_pooled_embedder(_MEDCPT_ARTICLE_MODEL, max_length=512)
    query_fn = _cls_pooled_embedder(_MEDCPT_QUERY_MODEL, max_length=64)
    return article_fn, query_fn


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
        query_embed_fn: EmbedFn | None = None,
    ) -> None:
        """Open or build the semantic index.

        :param con: Open connection to ``vocab.duckdb``.
        :param embed_fn: Embedding function used to embed concept names at
            index-build time (and, absent ``query_embed_fn``, at query
            time too). Defaults to :func:`default_embedder` (real
            BioLORD-2023) if not given.
        :param index_path: Path to a persisted LanceDB directory built by
            ``scripts/build_index.py``. If missing, not given, or present
            but unreadable (interrupted or corrupt ``build_index.py`` run),
            an ephemeral in-memory index is built instead.
        :param query_embed_fn: Separate embedding function for query text
            at search time, for an asymmetric encoder pair like MedCPT's
            article/query encoders (see :func:`medcpt_embedders`).
            Defaults to ``embed_fn`` itself — correct for every symmetric
            embedder (BioLORD, SapBERT), where the same function embeds
            both concept names and queries.
        """
        self._embed_fn = embed_fn or default_embedder()
        self._query_embed_fn = query_embed_fn or self._embed_fn
        self._table: Any = None

        if index_path is not None and index_path.exists():
            try:
                db = lancedb.connect(str(index_path))
                self._table = db.open_table(_TABLE_NAME)
            except Exception:
                # LanceDB doesn't document a specific exception type for a
                # missing/corrupt table (observed: ValueError today), and
                # this boundary is exactly where an interrupted build_index.py
                # run would surface — fall back rather than propagate, same
                # as the "no index built yet" case below.
                self._table = None

        if self._table is None:
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
        query_vector = _normalize(self._query_embed_fn([query])[0])
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
