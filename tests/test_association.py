"""Tests for crossmatching/association.py — CHIME-DSA association significance (pillars 1-4)."""

import math
from pathlib import Path

import numpy as np
import pytest

from crossmatching.association import (
    OMEGA_WIN_BASELINE_DEG2,
    build_association_report,
    chance_mu,
    chance_probability,
    dm_agreement,
    expected_chance_associations,
    f_dm,
    omega_disk_deg2,
    position_agreement,
    residual_pedestal,
    timing_budget_ms,
)

BASE = dict(rate_per_day=1000.0, omega_win_deg2=OMEGA_WIN_BASELINE_DEG2, dt_s=1.0, ddm=5.0)
ROOT = Path(__file__).resolve().parents[1]


# --- Pillar 1: chance-coincidence probability ---------------------------------
def test_chance_mu_regression_dm500():
    # pinned from the validated experiment (.agents/experiment-chance-coincidence-falsealarm.md)
    assert chance_mu(500.0, **BASE) == pytest.approx(5.023345e-09, rel=1e-4)


def test_chance_mu_scales_linearly_in_small_window():
    base = chance_mu(500.0, **BASE)
    assert chance_mu(500.0, **{**BASE, "dt_s": 2.0}) == pytest.approx(2 * base, rel=1e-9)
    assert chance_mu(500.0, **{**BASE, "ddm": 10.0}) == pytest.approx(2 * base, rel=1e-9)


def test_chance_matches_monte_carlo_in_measurable_regime():
    # analytic must equal a direct background MC where the MC has enough hits (mu ~ 0.046)
    infl = dict(rate_per_day=1000.0, omega_win_deg2=200.0, dt_s=3600.0, ddm=50.0)
    p_an = chance_probability(500.0, **infl)
    rng = np.random.default_rng(7)
    lam = chance_mu(500.0, **infl) / f_dm(500.0, 50.0)  # mean events in pos+time box
    n = 2_000_000
    counts = rng.poisson(lam, size=n)
    total = int(counts.sum())
    dms = np.exp(rng.normal(math.log(500.0), 0.7, size=total))
    hit = np.zeros(n, bool)
    hit[np.repeat(np.arange(n), counts)[np.abs(dms - 500.0) <= 50.0]] = True
    p_mc = hit.mean()
    assert p_mc == pytest.approx(p_an, rel=0.05)


def test_expected_chance_associations_sums_mu():
    dms = [262.4, 500.0, 960.1]
    assert expected_chance_associations(dms, **BASE) == pytest.approx(
        sum(chance_mu(d, **BASE) for d in dms), rel=1e-12
    )


# --- Pillar 2: independent DM agreement ---------------------------------------
def test_dm_agreement_consistent():
    r = dm_agreement(dm_chime=500.0, dm_chime_err=2.0, dm_dsa=502.0, dm_dsa_err=1.0)
    assert r["delta"] == pytest.approx(2.0)
    assert r["sigma"] == pytest.approx(math.sqrt(5.0))
    assert r["n_sigma"] == pytest.approx(2.0 / math.sqrt(5.0))
    assert r["consistent"] is True


def test_dm_agreement_inconsistent_beyond_3sigma():
    r = dm_agreement(dm_chime=500.0, dm_chime_err=1.0, dm_dsa=510.0, dm_dsa_err=1.0)
    assert r["consistent"] is False


def test_dm_agreement_missing_chime_dm_returns_null_reason():
    r = dm_agreement(dm_chime=None, dm_chime_err=None, dm_dsa=502.0, dm_dsa_err=1.0)
    assert r["consistent"] is None and "no CHIME DM" in r["reason"]


# --- Pillar 3: timing budget + residual-pedestal significance ------------------
def test_timing_budget_quadrature():
    got = timing_budget_ms(
        dm_unc_ms=2.4, fwhm_ms=0.96, clock_ms=0.1, baseline_ms=0.05, intrachannel_ms=0.2
    )
    assert got == pytest.approx(math.sqrt(2.4**2 + 0.96**2 + 0.1**2 + 0.05**2 + 0.2**2))


def test_residual_pedestal_significance():
    # equal residuals of +2.4 with errors 2.4 -> weighted mean 2.4, error 2.4/sqrt(12)
    res = [2.4] * 12
    err = [2.4] * 12
    r = residual_pedestal(res, err)
    assert r["weighted_mean_ms"] == pytest.approx(2.4)
    assert r["error_ms"] == pytest.approx(2.4 / math.sqrt(12))
    assert r["n_sigma"] == pytest.approx(math.sqrt(12))


# --- Pillar 4: positional coincidence -----------------------------------------
def test_omega_disk_area():
    assert omega_disk_deg2(0.5) == pytest.approx(math.pi * 0.25)


def test_position_agreement_inside_outside_null():
    dsa = "20h40m47.886s +72d52m56.378s"
    near = position_agreement(dsa, 310.1995, 72.8823, radius_deg=0.1)
    assert near["consistent"] is True and near["separation_deg"] < 0.01
    far = position_agreement(dsa, 300.0, 60.0, radius_deg=0.1)
    assert far["consistent"] is False and far["separation_deg"] > 10.0
    null = position_agreement(dsa, None, None, radius_deg=0.1)
    assert null["consistent"] is None and "no CHIME position" in null["reason"]


# --- Phase 5: assembled report (golden untouched) -----------------------------
def test_report_has_chance_P_for_all_12_and_golden_untouched():
    golden = ROOT / "crossmatching/toa_crossmatch_results.json"
    golden_before = golden.read_text()
    report = build_association_report(ROOT / "crossmatching/notebook_reproduction_fixture.json")
    assert len(report["bursts"]) == 12
    assert all(b["chance_coincidence_P"] < 1e-3 for b in report["bursts"])
    assert report["expected_chance_associations"] < 1e-3
    # building the report must not touch the golden artifact
    assert golden.read_text() == golden_before


def test_report_activates_pillars_2_and_4_from_chime_inputs(tmp_path):
    import json

    # stub CHIME-side inputs for one burst (zach); the other 11 stay null+reason
    stub = [
        {
            "chime_id": "210456524",
            "name": "zach",
            "dm_chime": 264.67,
            "dm_chime_err": 1.85,
            "chime_ra_deg": 310.1807,
            "chime_dec_deg": 72.8976,
        }
    ]
    p = tmp_path / "chime_side_inputs.json"
    p.write_text(json.dumps(stub))
    report = build_association_report(
        ROOT / "crossmatching/notebook_reproduction_fixture.json", chime_inputs_path=p
    )
    by = {b["name"]: b for b in report["bursts"]}
    z = by["zach"]
    da = z["dm_agreement"]
    assert da["consistent"] is True  # activated, and 264.67 vs DSA agrees within 3 sigma
    assert da["delta"] == pytest.approx(abs(264.67 - z["dm"]), rel=1e-6)
    assert da["n_sigma"] == pytest.approx(da["delta"] / da["sigma"], rel=1e-6)
    assert z["position"]["consistent"] is not None and z["position"]["separation_deg"] < 0.1
    # a burst absent from the stub stays null+reason
    other = next(b for b in report["bursts"] if b["name"] != "zach")
    assert other["dm_agreement"]["consistent"] is None
    assert other["position"]["consistent"] is None
    assert report["inputs"]["chime_localization_radius_deg"] == 0.1


def test_report_with_real_chime_inputs_activates_pillars():
    # committed CHIME-side extraction (RFI-mask recipe, figure-reviewed): 2 real / 7 marginal /
    # 3 noise -> 9/12 DM active (isha/phineas/mahi noise -> null), all 12 positions consistent.
    chime = ROOT / "crossmatching/chime_side_inputs.json"
    if not chime.exists():
        import pytest

        pytest.skip("chime_side_inputs.json not present")
    report = build_association_report(
        ROOT / "crossmatching/notebook_reproduction_fixture.json", chime_inputs_path=chime
    )
    by = {b["name"]: b for b in report["bursts"]}
    noise = {"isha", "phineas", "mahi"}
    assert {n for n in by if by[n]["dm_confidence"] == "noise"} == noise
    for n in noise:  # noise -> nulled, not fabricated
        assert by[n]["dm_agreement"]["consistent"] is None
    dm_active = [b for b in report["bursts"] if b["dm_agreement"]["consistent"] is not None]
    assert len(dm_active) == 9
    assert all(b["dm_agreement"]["consistent"] is True for b in dm_active)  # all within 3 sigma
    assert all(b["position"]["consistent"] is True for b in report["bursts"])  # 12/12 positions
    assert {b["name"] for b in report["bursts"] if b["dm_confidence"] == "real"} == {
        "zach",
        "freya",
    }
