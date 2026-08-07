"""Fetch a small demo set of OHDSI Phenotype Library cohort definitions.

The OHDSI PhenotypeLibrary GitHub repo has no confirmed LICENSE file (the R
package's ``DESCRIPTION`` claims Apache, but that's a manifest claim, not a
verified grant for the cohort JSON content itself). Following the same
instinct already applied to the Athena vocabulary download: fetch into a
gitignored local directory, never commit the content to this repo.

Cohort ids below are a hand-picked demo set (diabetes/kidney-relevant,
matching the README's own population-description example), not the full
library — see README for the full list and rationale.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer
from pydantic import BaseModel

app = typer.Typer(add_completion=False)

_RAW_BASE = "https://raw.githubusercontent.com/OHDSI/PhenotypeLibrary/main/inst/cohorts"

_COHORT_NAMES = {
    40: "Diabetes Mellitus Type 2 or history of diabetes",
    288: "Type 2 Diabetes Mellitus indexed on diagnosis/treatment/lab",
    503: "Type 2 diabetes mellitus",
    499: "Type 1 diabetes mellitus",
    611: "Diabetic ketoacidosis",
    619: "Gestational diabetes mellitus",
    647: "Retinopathy due to diabetes mellitus",
    687: "Chronic kidney disease",
}
_COHORT_IDS = list(_COHORT_NAMES)


class FetchReport(BaseModel):
    """Summary of a completed fetch.

    :ivar fetched: Cohort ids successfully fetched and written to disk.
    :ivar failed: Cohort ids that failed to fetch (e.g. HTTP error, moved id).
    """

    fetched: list[int]
    failed: list[int]


def fetch_phenotype_library(
    cohort_ids: list[int],
    output_dir: Path,
    client: httpx.Client | None = None,
) -> FetchReport:
    """Fetch cohort JSON files and a name manifest into ``output_dir``.

    :param cohort_ids: OHDSI Phenotype Library cohort ids to fetch.
    :param output_dir: Directory to write ``{id}.json`` files and
        ``manifest.json`` into. Created if missing.
    :param client: Injectable HTTP client, for tests to pass a mock
        transport and never touch the network. Defaults to a real
        ``httpx.Client``.
    :returns: Which cohort ids succeeded and which failed.
    :rtype: FetchReport
    """
    owns_client = client is None
    client = client or httpx.Client()
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[int] = []
    failed: list[int] = []
    manifest: dict[str, str] = {}
    try:
        for cohort_id in cohort_ids:
            response = client.get(f"{_RAW_BASE}/{cohort_id}.json")
            if response.status_code != 200:
                failed.append(cohort_id)
                continue
            (output_dir / f"{cohort_id}.json").write_text(response.text)
            manifest[str(cohort_id)] = _COHORT_NAMES.get(cohort_id, str(cohort_id))
            fetched.append(cohort_id)
    finally:
        if owns_client:
            client.close()

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return FetchReport(fetched=fetched, failed=failed)


@app.command()
def main(
    output: Path = typer.Option(
        Path("data/phenotype_library"),
        "--output",
        "-o",
        help="Output directory for cohort JSON files and manifest.json.",
    ),
) -> None:
    """Fetch the bundled demo set of OHDSI Phenotype Library cohorts.

    :param output: Output directory for cohort JSON files and manifest.json.
    """
    report = fetch_phenotype_library(_COHORT_IDS, output)
    typer.echo(f"Fetched {len(report.fetched)} cohorts to {output}")
    if report.failed:
        typer.echo(f"  WARNING: failed to fetch cohort ids: {report.failed}")


if __name__ == "__main__":
    app()
