"""A deterministic, dependency-free stand-in for a real sentence embedder.

Used by dense-retrieval tests so they never download BioLORD-2023 or any
other model. Not a pytest fixture (plain function) so it's importable from
any test directory (tests/engine, tests/scripts, tests/mcp) without fixture
re-export plumbing.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIM = 512


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _embed_one(text: str) -> list[float]:
    """Feature-hash a single text into a small fixed-dim, L2-normalized vector.

    Each token is hashed into one of ``_DIM`` buckets and accumulated, so
    texts sharing tokens end up with nonzero cosine similarity — enough
    lexical signal for meaningful test assertions without any ML model.
    """
    vector = [0.0] * _DIM
    for token in _tokenize(text):
        bucket = int(hashlib.sha256(token.encode()).hexdigest(), 16) % _DIM
        vector[bucket] += 1.0

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Matches the ``EmbedFn`` contract in ``dense.py``.

    :param texts: Texts to embed.
    :returns: One L2-normalized, ``_DIM``-dimensional vector per input text.
    :rtype: list[list[float]]
    """
    return [_embed_one(text) for text in texts]
