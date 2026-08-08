"""Fixtures for agent tests: reuses the miniature vocabulary and demo phenotype library.

No agent test ever calls :func:`~phenoforge.agent.nodes.default_decomposer`
(real Anthropic call) or builds a real
:class:`~phenoforge.engine.dense.DenseRetriever` (real model download) —
``fake_decompose_fn`` and ``dense=None`` stand in for both everywhere.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from phenoforge.agent.nodes import DecomposeFn
from phenoforge.engine.db import connect
from tests.eval.conftest import library_dir  # noqa: F401  (re-exported fixture)
from tests.scripts.conftest import MiniVocab, mini_vocab  # noqa: F401  (re-exported fixture)

#: A seed term that resolves against ``library_dir``'s bundled cohort, and one
#: that deliberately doesn't — exercises both branches of both conditional
#: edges (curated hit/miss, confirm reached/skipped) from one fixture setup.
RESOLVING_TERM = "diabetic nephropathy"
UNRESOLVING_TERM = "made up unrelated condition xyz"


@pytest.fixture
def con(mini_vocab: MiniVocab) -> Iterator[duckdb.DuckDBPyConnection]:  # noqa: F811
    """Open a read-only engine connection to the miniature vocabulary.

    :param mini_vocab: The built fixture database.
    :yields: A DuckDB connection, closed on teardown.
    """
    connection = connect(mini_vocab.db_path)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def fake_decompose_fn() -> DecomposeFn:
    """A deterministic stand-in for :func:`~phenoforge.agent.nodes.default_decomposer`.

    :returns: A function ignoring its input and returning
        ``[RESOLVING_TERM, UNRESOLVING_TERM]``.
    :rtype: DecomposeFn
    """

    def decompose(population_description: str) -> list[str]:
        return [RESOLVING_TERM, UNRESOLVING_TERM]

    return decompose
