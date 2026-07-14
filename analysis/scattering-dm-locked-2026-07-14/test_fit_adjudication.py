"""Integrity checks for the frozen DM-locked joint-fit roster."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PBF_ROSTER = HERE.parent / "scattering-refit-2026-06" / "citable_alpha_roster.json"


def _rows() -> list[dict[str, str]]:
    with (RESULTS / "fit_adjudication.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_adjudicated_fit_uses_the_recorded_fixed_dm_residuals():
    rows = _rows()
    assert len(rows) == 12
    for row in rows:
        if not row["fit_json"]:
            continue
        fit = json.loads((RESULTS / row["fit_json"]).read_text())
        fixed = fit["fixed_parameters"]
        adopted = float(row["adopted_dm"])
        assert fixed["delta_dm_C"] == pytest.approx(
            adopted - float(row["product_dm_C"]), abs=5e-7
        )
        assert fixed["delta_dm_D"] == pytest.approx(
            adopted - float(row["product_dm_D"]), abs=5e-7
        )


def test_pbf_roster_is_exactly_the_physically_accepted_subset():
    accepted = {row["burst"] for row in _rows() if row["adjudication"] == "accepted_physical"}
    roster = json.loads(PBF_ROSTER.read_text())
    promoted = {row["nickname"] for row in roster["tier_a_fully_adjudicated"]}
    assert promoted == accepted
    assert promoted == {"whitney", "oran", "isha", "phineas", "freya", "johndoeII", "mahi"}
