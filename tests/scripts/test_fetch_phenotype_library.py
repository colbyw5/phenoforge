"""Tests for scripts/fetch_phenotype_library.py, no real network access."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from scripts.fetch_phenotype_library import fetch_phenotype_library


def _mock_client(responses: dict[int, httpx.Response]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        cohort_id = int(request.url.path.rsplit("/", 1)[-1].removesuffix(".json"))
        return responses.get(cohort_id, httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_writes_cohort_json_and_manifest(tmp_path: Path) -> None:
    body = json.dumps({"ConceptSets": []})
    client = _mock_client({40: httpx.Response(200, text=body), 503: httpx.Response(200, text=body)})

    report = fetch_phenotype_library([40, 503], tmp_path, client=client)

    assert report.fetched == [40, 503]
    assert report.failed == []
    assert (tmp_path / "40.json").read_text() == body
    assert (tmp_path / "503.json").read_text() == body
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["40"] == "Diabetes Mellitus Type 2 or history of diabetes"
    assert manifest["503"] == "Type 2 diabetes mellitus"


def test_fetch_reports_failures(tmp_path: Path) -> None:
    body = json.dumps({"ConceptSets": []})
    client = _mock_client({40: httpx.Response(200, text=body)})

    report = fetch_phenotype_library([40, 999], tmp_path, client=client)

    assert report.fetched == [40]
    assert report.failed == [999]
    assert not (tmp_path / "999.json").exists()


def test_fetch_creates_output_dir(tmp_path: Path) -> None:
    body = json.dumps({"ConceptSets": []})
    client = _mock_client({40: httpx.Response(200, text=body)})
    output_dir = tmp_path / "nested" / "phenotype_library"

    fetch_phenotype_library([40], output_dir, client=client)

    assert output_dir.exists()
    assert (output_dir / "40.json").exists()
