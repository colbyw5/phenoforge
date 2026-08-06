"""Build ``data/vocab.duckdb`` from an unzipped OHDSI Athena download.

Loads ``CONCEPT`` and ``CONCEPT_RELATIONSHIP``, filtered to ICD-10-CM (the
shipped, user-facing vocabulary) and SNOMED (loaded for local use only, per
DESIGN.md D9 — never exposed through the engine or MCP surface, used solely
for the future OHDSI Phenotype Library resolution path, DESIGN.md D4).
Optionally loads an AHRQ CCSR category file (DESIGN.md D10).

``CONCEPT_ANCESTOR`` is deliberately not loaded: it has zero ICD-10-CM
coverage (DESIGN.md D1). Descendant expansion instead walks
``CONCEPT_RELATIONSHIP`` ``Is a`` / ``Subsumes`` edges via a recursive CTE at
query time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import typer
from pydantic import BaseModel, Field

app = typer.Typer(add_completion=False)

_ICD10CM = "ICD10CM"
_SNOMED = "SNOMED"
_HIERARCHY_RELATIONSHIPS = ("Is a", "Subsumes")
_MAPPING_RELATIONSHIPS = ("Mapped from", "Maps to")


class VocabPaths(BaseModel):
    """Filesystem locations of the source files a load reads from.

    :ivar athena_dir: Root of the unzipped Athena bulk download. Must contain
        ``CONCEPT.csv`` and ``CONCEPT_RELATIONSHIP.csv``.
    :ivar ccsr_csv: Optional path to an AHRQ CCSR category file
        (``DXCCSR_v*.csv``). Skipped if not provided.
    """

    athena_dir: Path
    ccsr_csv: Path | None = None

    @property
    def concept_tsv(self) -> Path:
        """Path to the Athena ``CONCEPT`` export.

        :returns: Path to ``CONCEPT.csv`` under ``athena_dir``.
        :rtype: Path
        """
        return self.athena_dir / "CONCEPT.csv"

    @property
    def concept_relationship_tsv(self) -> Path:
        """Path to the Athena ``CONCEPT_RELATIONSHIP`` export.

        :returns: Path to ``CONCEPT_RELATIONSHIP.csv`` under ``athena_dir``.
        :rtype: Path
        """
        return self.athena_dir / "CONCEPT_RELATIONSHIP.csv"


class LoadReport(BaseModel):
    """Summary of a completed load, returned for logging and testing.

    :ivar concept_counts: Row count loaded into ``concept``, keyed by
        ``vocabulary_id``.
    :ivar relationship_counts: Row count loaded into ``concept_relationship``,
        keyed by ``relationship_id``.
    :ivar ccsr_rows: Row count loaded into ``concept_ccsr``, or ``0`` if the
        CCSR file was not supplied.
    :ivar high_fanout_snomed_concepts: SNOMED ``concept_id`` values whose
        ``Mapped from`` fan-out into ICD-10-CM exceeds the configured
        threshold. Flagged, not dropped — resolution-time consumers decide
        what to do with them (DESIGN.md D4).
    """

    concept_counts: dict[str, int] = Field(default_factory=dict)
    relationship_counts: dict[str, int] = Field(default_factory=dict)
    ccsr_rows: int = 0
    high_fanout_snomed_concepts: list[int] = Field(default_factory=list)


def _load_concept(con: duckdb.DuckDBPyConnection, paths: VocabPaths) -> None:
    con.execute(
        """
        CREATE TABLE concept AS
        SELECT
            concept_id,
            concept_name,
            domain_id,
            vocabulary_id,
            concept_class_id,
            standard_concept,
            concept_code,
            valid_start_date,
            valid_end_date,
            invalid_reason
        FROM read_csv(
            ?,
            delim='\t',
            header=true,
            quote='',
            dateformat='%Y%m%d',
            columns={
                'concept_id': 'BIGINT',
                'concept_name': 'VARCHAR',
                'domain_id': 'VARCHAR',
                'vocabulary_id': 'VARCHAR',
                'concept_class_id': 'VARCHAR',
                'standard_concept': 'VARCHAR',
                'concept_code': 'VARCHAR',
                'valid_start_date': 'DATE',
                'valid_end_date': 'DATE',
                'invalid_reason': 'VARCHAR'
            }
        )
        WHERE vocabulary_id IN (?, ?)
          AND invalid_reason IS NULL
        """,
        [str(paths.concept_tsv), _ICD10CM, _SNOMED],
    )
    con.execute("ALTER TABLE concept ADD PRIMARY KEY (concept_id)")
    con.execute("CREATE UNIQUE INDEX idx_concept_code ON concept(vocabulary_id, concept_code)")


def _load_concept_relationship(con: duckdb.DuckDBPyConnection, paths: VocabPaths) -> None:
    relationship_ids = _HIERARCHY_RELATIONSHIPS + _MAPPING_RELATIONSHIPS
    placeholders = ", ".join("?" for _ in relationship_ids)
    con.execute(
        f"""
        CREATE TABLE concept_relationship AS
        SELECT
            r.concept_id_1,
            r.concept_id_2,
            r.relationship_id,
            r.valid_start_date,
            r.valid_end_date,
            r.invalid_reason
        FROM read_csv(
            ?,
            delim='\t',
            header=true,
            quote='',
            dateformat='%Y%m%d',
            columns={{
                'concept_id_1': 'BIGINT',
                'concept_id_2': 'BIGINT',
                'relationship_id': 'VARCHAR',
                'valid_start_date': 'DATE',
                'valid_end_date': 'DATE',
                'invalid_reason': 'VARCHAR'
            }}
        ) r
        JOIN concept c1 ON c1.concept_id = r.concept_id_1
        JOIN concept c2 ON c2.concept_id = r.concept_id_2
        WHERE r.relationship_id IN ({placeholders})
          AND r.invalid_reason IS NULL
        """,
        [str(paths.concept_relationship_tsv), *relationship_ids],
    )
    con.execute("CREATE INDEX idx_rel_from ON concept_relationship(concept_id_1, relationship_id)")
    con.execute("CREATE INDEX idx_rel_to ON concept_relationship(concept_id_2, relationship_id)")


def _read_ccsr_rows(ccsr_csv: Path) -> list[dict[str, str]]:
    """Parse an AHRQ CCSR file, working around its mixed quoting.

    Codes are single-quoted (``'A000'``), descriptions double-quoted.
    ``csv.reader`` with ``quotechar='"'`` handles the double-quoted fields
    correctly; the surviving single quotes on code-like fields are stripped
    per field. See DESIGN.md D10.

    :param ccsr_csv: Path to the CCSR category CSV.
    :returns: One dict per data row, keyed by header name.
    :rtype: list[dict[str, str]]
    """
    with ccsr_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, quotechar='"')
        header = [h.strip().strip("'").strip() for h in next(reader)]
        rows = []
        for raw_row in reader:
            if not raw_row:
                continue
            cleaned = [cell.strip().strip("'").strip() for cell in raw_row]
            rows.append(dict(zip(header, cleaned, strict=True)))
    return rows


def _load_ccsr(con: duckdb.DuckDBPyConnection, ccsr_csv: Path) -> int:
    rows = _read_ccsr_rows(ccsr_csv)

    code_col = next(c for c in rows[0] if "code" in c.lower() and "icd" in c.lower())
    category_desc_cols = [c for c in rows[0] if "ccsr" in c.lower() and "description" in c.lower()]
    category_cols = [
        c
        for c in rows[0]
        if "ccsr" in c.lower() and "category" in c.lower() and c not in category_desc_cols
    ]

    con.execute(
        "CREATE TABLE ccsr_category (category_id VARCHAR PRIMARY KEY, category_name VARCHAR)"
    )
    con.execute("CREATE TABLE concept_ccsr (concept_code VARCHAR, category_id VARCHAR)")

    categories: dict[str, str] = {}
    concept_ccsr_rows: list[tuple[str, str]] = []
    for row in rows:
        code = row[code_col]
        for cat_col, desc_col in zip(category_cols, category_desc_cols, strict=True):
            category_id = row.get(cat_col, "").strip()
            if not category_id:
                continue
            categories.setdefault(category_id, row.get(desc_col, "").strip())
            concept_ccsr_rows.append((code, category_id))

    con.executemany("INSERT INTO ccsr_category VALUES (?, ?)", list(categories.items()))
    con.executemany("INSERT INTO concept_ccsr VALUES (?, ?)", concept_ccsr_rows)
    con.execute("CREATE INDEX idx_ccsr_code ON concept_ccsr(concept_code)")
    con.execute("CREATE INDEX idx_ccsr_category ON concept_ccsr(category_id)")
    return len(rows)


def _find_high_fanout_snomed_concepts(
    con: duckdb.DuckDBPyConnection, fanout_threshold: int
) -> list[int]:
    """Flag SNOMED concepts whose ``Mapped from`` fan-out into ICD-10-CM is excessive.

    DESIGN.md D4: fan-out is 1..3029 across the full download; anything past
    the threshold is structural (a high-level grouper), not clinical
    specificity, and should be surfaced for review rather than expanded
    blindly at resolution time.

    :param con: Open connection with ``concept`` and ``concept_relationship`` loaded.
    :param fanout_threshold: Concepts mapping to more than this many
        ICD-10-CM codes are flagged.
    :returns: SNOMED ``concept_id`` values exceeding the threshold.
    :rtype: list[int]
    """
    # For relationship_id='Mapped from', concept_id_1 is the SNOMED concept
    # and concept_id_2 is the ICD-10-CM concept it was mapped from (verified
    # against the v2026 download — the docs' "SNOMED -> ICD10CM" framing
    # describes the mapping direction, not the column order).
    rows = con.execute(
        """
        SELECT r.concept_id_1 AS snomed_concept_id, COUNT(*) AS n
        FROM concept_relationship r
        JOIN concept sn ON sn.concept_id = r.concept_id_1 AND sn.vocabulary_id = ?
        JOIN concept icd ON icd.concept_id = r.concept_id_2 AND icd.vocabulary_id = ?
        WHERE r.relationship_id = 'Mapped from'
        GROUP BY r.concept_id_1
        HAVING COUNT(*) > ?
        ORDER BY n DESC
        """,
        [_SNOMED, _ICD10CM, fanout_threshold],
    ).fetchall()
    return [int(row[0]) for row in rows]


def build_vocab_db(
    paths: VocabPaths,
    db_path: Path,
    fanout_threshold: int = 100,
) -> LoadReport:
    """Build the vocabulary DuckDB database from Athena source files.

    :param paths: Locations of the Athena export and optional CCSR file.
    :param db_path: Output path for the DuckDB database. Overwritten if it
        already exists.
    :param fanout_threshold: SNOMED-to-ICD-10-CM ``Mapped from`` fan-out above
        this count is flagged in the returned report (DESIGN.md D4).
    :returns: Summary of what was loaded.
    :rtype: LoadReport
    :raises FileNotFoundError: If ``CONCEPT.csv`` or ``CONCEPT_RELATIONSHIP.csv``
        is missing under ``paths.athena_dir``.
    """
    if not paths.concept_tsv.exists():
        raise FileNotFoundError(paths.concept_tsv)
    if not paths.concept_relationship_tsv.exists():
        raise FileNotFoundError(paths.concept_relationship_tsv)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        _load_concept(con, paths)
        _load_concept_relationship(con, paths)

        concept_counts = dict(
            con.execute(
                "SELECT vocabulary_id, COUNT(*) FROM concept GROUP BY vocabulary_id"
            ).fetchall()
        )
        relationship_counts = dict(
            con.execute(
                "SELECT relationship_id, COUNT(*) FROM concept_relationship"
                " GROUP BY relationship_id"
            ).fetchall()
        )

        ccsr_rows = 0
        if paths.ccsr_csv is not None:
            ccsr_rows = _load_ccsr(con, paths.ccsr_csv)

        high_fanout = _find_high_fanout_snomed_concepts(con, fanout_threshold)

        return LoadReport(
            concept_counts=concept_counts,
            relationship_counts=relationship_counts,
            ccsr_rows=ccsr_rows,
            high_fanout_snomed_concepts=high_fanout,
        )
    finally:
        con.close()


@app.command()
def main(
    athena_dir: Path = typer.Argument(..., help="Path to an unzipped OHDSI Athena bulk download."),
    output: Path = typer.Option(
        Path("data/vocab.duckdb"), "--output", "-o", help="Output DuckDB path."
    ),
    ccsr_csv: Path | None = typer.Option(
        None,
        "--ccsr-csv",
        help="Path to an AHRQ CCSR category CSV (e.g. DXCCSR_v2026-1.csv).",
    ),
    fanout_threshold: int = typer.Option(
        100,
        "--fanout-threshold",
        help="Flag SNOMED concepts mapping to more ICD-10-CM codes than this.",
    ),
) -> None:
    """Build data/vocab.duckdb from an unzipped Athena download.

    :param athena_dir: Path to the unzipped Athena bulk download.
    :param output: Output path for the built DuckDB database.
    :param ccsr_csv: Optional path to an AHRQ CCSR category CSV.
    :param fanout_threshold: SNOMED to ICD-10-CM fan-out flag threshold.
    """
    paths = VocabPaths(athena_dir=athena_dir, ccsr_csv=ccsr_csv)
    report = build_vocab_db(paths, output, fanout_threshold=fanout_threshold)

    typer.echo(f"Built {output}")
    for vocab, count in report.concept_counts.items():
        typer.echo(f"  concept[{vocab}]: {count:,}")
    for rel, count in report.relationship_counts.items():
        typer.echo(f"  concept_relationship[{rel}]: {count:,}")
    if report.ccsr_rows:
        typer.echo(f"  concept_ccsr rows: {report.ccsr_rows:,}")
    if report.high_fanout_snomed_concepts:
        typer.echo(
            f"  WARNING: {len(report.high_fanout_snomed_concepts)} SNOMED concepts "
            f"exceed fan-out threshold ({fanout_threshold})"
        )


if __name__ == "__main__":
    app()
