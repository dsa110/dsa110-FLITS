"""Correctness tests for the expanded foreground catalog primitives.

Expected values come from published equations or invariants, not from the
production builder.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from galaxies.foreground.expanded_catalog import (
    cluver14_log_mstar,
    dutton_maccio14_c200c,
    m200c_to_r200c,
    rho_critical_msun_kpc3,
    select_match,
    stern12_status,
)
from galaxies.foreground.build_expanded_catalog import build_frame
from galaxies.foreground.census_registry import load_intervening_census_registry
from galaxies.foreground.vo.halos import mstar_to_mhalo


def test_match_is_nearest_and_order_independent() -> None:
    rows = [
        {"id": "far", "ra_deg": 10.00050, "dec_deg": 20.0},
        {"id": "near", "ra_deg": 10.00005, "dec_deg": 20.0},
        {"id": "middle", "ra_deg": 10.00020, "dec_deg": 20.0},
    ]
    expected = select_match(rows, (10.0, 20.0), 3.0, 0.05)
    reversed_result = select_match(list(reversed(rows)), (10.0, 20.0), 3.0, 0.05)
    assert expected.status == "matched"
    assert expected.selected_id == "near"
    assert reversed_result.selected_id == expected.selected_id
    assert reversed_result.separation_arcsec == pytest.approx(expected.separation_arcsec)
    assert expected.candidate_count == 3
    assert expected.second_separation_arcsec > expected.separation_arcsec


def test_match_states_distinguish_empty_ambiguous_and_query_error() -> None:
    assert select_match([], (10.0, 20.0), 3.0, 0.05).status == "unmatched"
    assert select_match(None, (10.0, 20.0), 3.0, 0.05, query_error="timeout").status == "query_error"
    rows = [
        {"id": "a", "ra_deg": 10.000050, "dec_deg": 20.0},
        {"id": "b", "ra_deg": 10.000055, "dec_deg": 20.0},
    ]
    assert select_match(rows, (10.0, 20.0), 3.0, 0.05).status == "ambiguous"


def test_cluver_equation_two_and_uncertainty() -> None:
    result = cluver14_log_mstar(
        15.0,
        14.8,
        40.0,
        rest_frame=True,
        valid_photometry=True,
        w1_error=0.03,
        w2_error=0.04,
        distance_modulus_error=0.10,
    )
    expected = -0.4 * ((15.0 - 40.0) - 3.24) - 2.54 * 0.2 - 0.17
    expected_error = math.sqrt((2.94 * 0.03) ** 2 + (2.54 * 0.04) ** 2 + (0.4 * 0.10) ** 2)
    assert result.status == "pass"
    assert result.value == pytest.approx(expected, abs=1e-12)
    assert result.uncertainty == pytest.approx(expected_error, abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"rest_frame": False, "valid_photometry": True, "w1_error": 0.1, "w2_error": 0.1}, "not_rest_frame"),
        ({"rest_frame": True, "valid_photometry": False, "w1_error": 0.1, "w2_error": 0.1}, "invalid_photometry"),
        ({"rest_frame": True, "valid_photometry": True}, "missing_uncertainty"),
    ],
)
def test_cluver_inapplicable_values_are_null(kwargs: dict, status: str) -> None:
    result = cluver14_log_mstar(15.0, 14.8, 40.0, **kwargs)
    assert result.status == status
    assert result.value is None


def test_stern_depth_is_required() -> None:
    assert stern12_status(0.9, w2=15.2, color_valid=True) == "outside_validated_depth"
    assert stern12_status(0.7, w2=14.0, color_valid=True) == "not_selected_within_depth"
    assert stern12_status(0.9, w2=14.0, color_valid=True) == "selected_by_stern12"
    assert stern12_status(0.9, w2=14.0, color_valid=False) == "insufficient_color"


def test_r200c_enclosed_mass_identity() -> None:
    m200 = 10.0**11.5997073
    radius = m200c_to_r200c(m200, z=0.2)
    enclosed = 4.0 * np.pi / 3.0 * radius**3 * 200.0 * rho_critical_msun_kpc3(0.2)
    assert enclosed == pytest.approx(m200, rel=1e-12)


def test_moster_published_reference_case() -> None:
    assert np.log10(mstar_to_mhalo(1.0e10, 0.2)) == pytest.approx(11.5997072638, abs=1e-9)


def test_dutton_maccio_published_parameterization() -> None:
    mass, z, h = 1.0e12, 0.5, 0.671
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = -0.101 + 0.026 * z
    expected = 10.0 ** (a + b * np.log10(mass * h / 1.0e12))
    assert dutton_maccio14_c200c(mass, z) == pytest.approx(expected, rel=1e-12)


def test_built_catalog_schema_and_registry_authority() -> None:
    frame, manifest = build_frame()
    registry = load_intervening_census_registry().copy()
    registry["obj"] = registry["obj"].astype(str)
    assert len(frame) == 52 == manifest["row_count"]
    assert not frame[["nickname", "type", "object_id"]].astype(str).duplicated().any()
    for catalog in ("gsc242", "allwise", "catwise2020", "unwise"):
        required = {
            f"{catalog}_status",
            f"{catalog}_id",
            f"{catalog}_separation_arcsec",
            f"{catalog}_candidate_count",
            f"{catalog}_second_separation_arcsec",
            f"{catalog}_release",
            f"{catalog}_retrieved_at_utc",
            f"{catalog}_snapshot_sha256",
        }
        assert required <= set(frame.columns)
        assert set(frame[f"{catalog}_status"]) <= {"matched", "unmatched", "ambiguous", "query_error"}
    assert {
        "allwise_w1_err_mag", "allwise_w2_err_mag", "allwise_qph", "allwise_ccf",
        "allwise_ex", "catwise2020_w1_err_mag", "catwise2020_w2_err_mag",
        "catwise2020_pmQual", "catwise2020_abf", "unwise_w1_flux_err",
        "unwise_w2_flux_err", "unwise_q_w1", "unwise_q_w2", "unwise_ff_w1",
        "unwise_ff_w2", "w1_w2_color_err_mag",
    } <= set(frame.columns)
    assert not any("r_vir" in name.lower() or "rvir" in name.lower() for name in frame.columns)
    expected = registry[["final_verdict", "registry_tier", "budget_eligible"]].reset_index(drop=True)
    actual = frame[["final_verdict", "registry_tier", "budget_eligible"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_derived_values_are_null_unless_status_is_pass_like() -> None:
    frame, _ = build_frame()
    for value, status in (
        ("m200c_msun", "m200c_status"),
        ("r200c_kpc", "r200c_status"),
        ("c200c", "c200c_status"),
        ("scale_radius_kpc", "scale_radius_status"),
    ):
        finite = pd.to_numeric(frame[value], errors="coerce").notna()
        assert frame.loc[finite, status].str.startswith("pass").all()


def test_committed_snapshots_are_complete_and_query_clean() -> None:
    snapshot_dir = Path(__file__).resolve().parent / "data" / "catalog_crossmatch_snapshots"
    import json

    for path in snapshot_dir.glob("*.json"):
        payload = json.loads(path.read_text())
        assert len(payload["queries"]) == 52
        assert all(query["query_status"] == "ok" for query in payload["queries"])
        assert all(len(query["response_sha256"]) == 64 for query in payload["queries"])
