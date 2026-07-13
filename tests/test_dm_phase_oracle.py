from __future__ import annotations

import json
from pathlib import Path

import pytest

from dispersion.dm_phase_suite.oracle import build_oracle_report


@pytest.mark.slow
def test_published_oracle_bright_geometries(tmp_path: Path) -> None:
    path = tmp_path / "oracle.json"
    report = build_oracle_report(path)
    assert report["pass"]
    assert {row["instrument"] for row in report["rows"]} == {"chime", "dsa"}
    assert json.loads(path.read_text()) == report
