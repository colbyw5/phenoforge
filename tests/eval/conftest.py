"""Reuses the ~200-concept miniature vocabulary built for the loader tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

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


@pytest.fixture
def library_dir(mini_vocab: MiniVocab, tmp_path: Path) -> Path:  # noqa: F811
    """A fake fetched phenotype library: one cohort mapping to E11.21 via `sn_normal_id`.

    :param mini_vocab: The built fixture database.
    :param tmp_path: pytest-provided temporary directory.
    :returns: Directory containing ``manifest.json`` and ``1.json``.
    """
    directory = tmp_path / "phenotype_library"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"1": "Diabetic nephropathy demo cohort"}))
    (directory / "1.json").write_text(
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
    return directory
