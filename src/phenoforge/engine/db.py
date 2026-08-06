"""Read-only connection helper for ``vocab.duckdb``.

No I/O to MCP or LLMs lives in this package (AGENTS.md layering rule) — this
module only knows how to open the database built by ``scripts/load_vocab.py``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_VOCAB_DB_PATH = Path("data/vocab.duckdb")


def connect(db_path: Path = DEFAULT_VOCAB_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open a read-only connection to the vocabulary database.

    :param db_path: Path to a DuckDB database built by
        ``scripts/load_vocab.py``.
    :returns: An open, read-only DuckDB connection.
    :rtype: duckdb.DuckDBPyConnection
    :raises FileNotFoundError: If ``db_path`` does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Build it with: python scripts/load_vocab.py <athena_dir>"
        )
    return duckdb.connect(str(db_path), read_only=True)
