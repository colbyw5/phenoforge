"""Reuses the ~200-concept miniature vocabulary built for the loader tests."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from phenoforge.engine.db import connect
from tests.scripts.conftest import MiniVocab, mini_vocab  # noqa: F401  (re-exported fixture)


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
