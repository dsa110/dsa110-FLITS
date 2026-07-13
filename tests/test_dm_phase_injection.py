from __future__ import annotations

from pathlib import Path

import pytest

from dispersion.dm_phase_suite.injection import run_quick_injections


@pytest.mark.slow
def test_quick_known_truth_gate(tmp_path: Path) -> None:
    report = run_quick_injections(tmp_path / "quick.json")
    assert report["pass"]
    assert len(report["rows"]) == 16
