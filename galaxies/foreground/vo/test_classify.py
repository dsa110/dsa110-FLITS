import numpy as np
import pandas as pd
import pytest

from .classify import add_intersection_flags, summarize_sightlines


@pytest.mark.unit
def test_intersection_flags_three_levels():
    # r_delta_computed = 100 kpc; b values chosen to fall into each bracket.
    df = pd.DataFrame(
        {
            "b_kpc": [50.0, 90.0, 110.0, 130.0, np.nan],
            "r_delta_computed": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    out = add_intersection_flags(df)  # defaults: loose=1.2, strict=0.8

    # Row 0: b/r=0.5 -> all three flags True
    assert out.loc[0, "intersects_strict"]
    assert out.loc[0, "intersects_nominal"]
    assert out.loc[0, "intersects_loose"]
    # Row 1: b/r=0.9 -> nominal + loose, not strict
    assert not out.loc[1, "intersects_strict"]
    assert out.loc[1, "intersects_nominal"]
    assert out.loc[1, "intersects_loose"]
    # Row 2: b/r=1.1 -> only loose
    assert not out.loc[2, "intersects_strict"]
    assert not out.loc[2, "intersects_nominal"]
    assert out.loc[2, "intersects_loose"]
    # Row 3: b/r=1.3 -> none
    assert not out.loc[3, "intersects_strict"]
    assert not out.loc[3, "intersects_nominal"]
    assert not out.loc[3, "intersects_loose"]
    # Row 4: NaN b -> all False, b_over_rvir NaN
    assert not out.loc[4, "intersects_strict"]
    assert np.isnan(out.loc[4, "b_over_rvir"])


@pytest.mark.unit
def test_summarize_sightlines_aggregates_per_frb():
    df = pd.DataFrame(
        {
            "frb_name": ["FRB_A", "FRB_A", "FRB_A", "FRB_B", "FRB_B"],
            "name": ["g1", "g2", "g3", "h1", "h2"],
            "id": ["1", "2", "3", "4", "5"],
            "b_kpc": [50.0, 110.0, 200.0, 70.0, 300.0],
            "r_delta_computed": [100.0, 100.0, 100.0, 100.0, 100.0],
            "z": [0.1, 0.1, 0.4, 0.05, 0.4],
            "frb_z": [0.5, 0.5, 0.5, 0.3, 0.3],
        }
    )
    out = add_intersection_flags(df)
    summary = summarize_sightlines(out)

    assert list(summary["frb_name"]) == ["FRB_A", "FRB_B"]
    a = summary[summary.frb_name == "FRB_A"].iloc[0]
    assert a["n_candidates"] == 3
    assert a["n_foreground"] == 3  # all candidates have z < frb_z (0.5)
    assert a["n_intersects_strict"] == 1  # only b=50 < 0.8*100
    assert a["n_intersects_nominal"] == 1  # only b=50 <= 100
    assert a["n_intersects_loose"] == 2  # b=50 and b=110 <= 1.2*100
    assert a["closest_name"] == "g1"
    assert abs(a["min_b_over_rvir"] - 0.5) < 1e-9

    b = summary[summary.frb_name == "FRB_B"].iloc[0]
    assert b["n_candidates"] == 2
    assert b["n_foreground"] == 1  # only h1 has z(0.05) < frb_z(0.3); h2 has z=0.4
    assert b["closest_name"] == "h1"


@pytest.mark.unit
def test_summarize_excludes_background_intersections_from_headline():
    """Regression: a background candidate (z >= frb_z) inside R_vir must not be counted.

    Constructs one BG candidate with b/r=0.5 (would intersect at all three brackets)
    and one FG candidate with b/r=2.0 (does not intersect). The headline counts
    should be zero, the closest-foreground fields should describe the FG row only.
    """
    df = pd.DataFrame(
        {
            "frb_name": ["FRB_X", "FRB_X"],
            "name": ["bg_close", "fg_far"],
            "id": ["B", "F"],
            "b_kpc": [50.0, 200.0],
            "r_delta_computed": [100.0, 100.0],
            "z": [0.6, 0.1],  # bg has z > frb_z; fg has z < frb_z
            "frb_z": [0.5, 0.5],
        }
    )
    out = add_intersection_flags(df)

    # Sanity: per-row flags do reflect geometry on both rows
    assert out.loc[0, "intersects_strict"] and out.loc[0, "intersects_nominal"]
    assert not out.loc[1, "intersects_nominal"]

    summary = summarize_sightlines(out)
    row = summary.iloc[0]
    assert row["n_candidates"] == 2
    assert row["n_foreground"] == 1
    # BG intersection must NOT show up in any of the headline counts
    assert row["n_intersects_strict"] == 0
    assert row["n_intersects_nominal"] == 0
    assert row["n_intersects_loose"] == 0
    # closest_* should describe the FG row, not the BG row
    assert row["closest_name"] == "fg_far"
    assert abs(row["min_b_over_rvir"] - 2.0) < 1e-9
