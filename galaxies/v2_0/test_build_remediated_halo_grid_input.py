"""Regression tests for the remediated halo-grid input and overlay."""

from __future__ import annotations

import pandas as pd

from galaxies.v2_0 import build_remediated_halo_grid_input as builder
from galaxies.v2_0.sightline_halo_grid import _contributor_mask


def _row(**updates):
    row = {
        "frb_name": "FRB 20220207C",
        "frb_z": 0.043,
        "frb_dec": 72.8823,
        "ra": 310.1,
        "dec": 72.8,
        "z": 0.02,
        "b_kpc": 25.0,
        "m_delta": 1.0e12,
        "r_delta_computed": 180.0,
        "is_foreground": True,
        "intersects_strict": True,
        "is_cluster": False,
    }
    row.update(updates)
    return row


def test_position_dedupe_drops_cross_listed_copy():
    frame = pd.DataFrame(
        [
            _row(),
            _row(ra=310.10001, dec=72.80001, z=0.02001),
            _row(ra=310.2, dec=72.9, z=0.03),
        ]
    )

    result = builder._position_dedupe(frame)

    assert len(result) == 2
    assert list(result.ra) == [310.1, 310.2]


def test_build_marks_existing_and_appends_missing_contributors(tmp_path, monkeypatch):
    source = tmp_path / "halos.csv"
    pd.DataFrame([_row()]).to_csv(source, index=False)
    contributors = [
        _row(ra=310.1001, dec=72.8001),
        _row(ra=310.3, dec=72.95, z=0.025, b_kpc=40.0, m_delta=2.0e11),
    ]
    monkeypatch.setattr(builder, "_budget_contributors", lambda: contributors)

    result = builder.build(str(source))

    assert len(result) == 2
    assert result["budget_contributor"].all()
    assert set(result.ra.round(3)) == {310.1, 310.3}


def test_contributor_mask_is_optional_and_boolean():
    assert not _contributor_mask(pd.DataFrame({"x": [1, 2]})).any()
    mask = _contributor_mask(
        pd.DataFrame({"budget_contributor": [True, "1", "false", 0]})
    )
    assert mask.tolist() == [True, True, False, False]
