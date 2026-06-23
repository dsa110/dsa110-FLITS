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
    position_consistent,
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


def test_position_inside_and_outside_chime_disk():
    dsa = "20h40m47.886s +72d52m56.378s"
    assert position_consistent(dsa, "20h40m50s +72d53m00s", radius_deg=0.2) is True
    assert position_consistent(dsa, "20h00m00s +60d00m00s", radius_deg=0.2) is False


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
