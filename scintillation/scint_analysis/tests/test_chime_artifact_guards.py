"""Artifact-control guards for CHIME upchannelized scintillation products.

These unit tests pin the pure verdict functions in
``scint_analysis.chime_artifact_guards`` to the numeric picture established by
the ``experiment-freya-chime-instrumental-origin`` experiment (docs/rse/specs/,
arms A/B1/C). They are data-free (no burst spectra) and encode the
falsifiable outcomes:

  - harmonic mask removes the coarse-channel comb lag bins and is a no-op when
    disabled (the DSA path);
  - a CHIME config missing a required mitigation is demoted to diagnostic_only,
    while a non-CHIME telescope is never demoted;
  - the freya CHIME off-pulse null FAILS (off median 37.7 kHz brackets on-pulse
    35.19 kHz) while the DSA off-pulse is white -> null passes;
  - freya CHIME low-lag excision collapses the fit (no wing) while DSA is
    excision-robust;
  - a full-mitigation CHIME burst that still fails the null / stability is
    forced to diagnostic_only (the actual freya case).

Mutation check: forcing any verdict the wrong way (e.g. treating a bracketing
off-pulse median as a pass) flips the corresponding assertion, so the tests
have teeth.
"""

from __future__ import annotations

import sys
from pathlib import Path

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

import numpy as np

from scint_analysis import chime_artifact_guards as guards


# --- harmonic mask (rec #1 / the --band chime trap) -------------------------


def test_harmonic_mask_removes_comb_when_enabled():
    # lags 0..1 MHz at 0.01 spacing; comb at k*0.390625 (0.390625, 0.78125).
    lags = np.arange(-100, 101) * 0.01
    acf = np.ones_like(lags)
    err = np.ones_like(lags)
    L, A, E, rec = guards.apply_harmonic_mask_to_fit(
        lags, acf, err, {"enable": True, "spacing_mhz": 0.390625, "halfwidth_mhz": 0.05}
    )
    assert rec["enabled"] is True
    assert rec["n_bins_removed"] > 0
    assert rec["n_bins_removed"] + rec["n_bins_kept"] == lags.size
    assert L.size == rec["n_bins_kept"] == A.size == E.size
    # every removed lag was actually near a positive comb harmonic
    removed = np.setdiff1d(np.round(lags, 6), np.round(L, 6))
    for lag in removed:
        k = round(abs(lag) / 0.390625)
        assert k >= 1 and abs(abs(lag) - k * 0.390625) <= 0.05 + 1e-9


def test_harmonic_mask_noop_when_disabled():
    lags = np.arange(-50, 51) * 0.01
    acf = np.ones_like(lags)
    for cfg in (None, {}, {"enable": False}):
        L, A, E, rec = guards.apply_harmonic_mask_to_fit(lags, acf, None, cfg)
        assert rec["enabled"] is False
        assert rec["n_bins_removed"] == 0
        assert L.size == lags.size
        assert E is None


# --- fail-closed provenance gate (rec #2) -----------------------------------


def _chime_cfg(grid=True, bandpass=True, harmonic=True):
    analysis = {"fitting": {}}
    if grid:
        analysis["grid_regularization"] = {"enable": True}
    if bandpass:
        analysis["bandpass_normalization"] = {"enable": True}
    if harmonic:
        analysis["fitting"]["harmonic_mask"] = {"enable": True}
    return {"telescope": "chime", "analysis": analysis}


def test_provenance_full_stack_is_measurement():
    p = guards.chime_provenance_status(_chime_cfg())
    assert p["is_chime"] is True
    assert p["status"] == guards.MEASUREMENT
    assert p["missing"] == []


def test_provenance_missing_mitigation_is_diagnostic_only():
    # casey_chime.yaml shape: harmonic mask on, but no grid/bandpass blocks.
    p = guards.chime_provenance_status(_chime_cfg(grid=False, bandpass=False))
    assert p["status"] == guards.DIAGNOSTIC_ONLY
    assert set(p["missing"]) == {"grid_regularization", "bandpass_normalization"}


def test_provenance_non_chime_never_demoted():
    p = guards.chime_provenance_status({"telescope": "dsa", "analysis": {}})
    assert p["is_chime"] is False
    assert p["status"] == guards.MEASUREMENT
    assert p["missing"] == []


# --- off-pulse ACF null (arm A / rec #3) ------------------------------------


def test_off_pulse_null_fails_on_freya_numbers():
    # arm A: on-pulse 35.19 kHz, off-pulse slices bracket it (median ~37.7 kHz).
    on = 0.03519
    off = [0.0524, 0.0423, 0.0380, 0.0328, 0.0382, 0.0345, 0.0375, 0.0393]
    v = guards.off_pulse_null_verdict(on, off)
    assert v["null_pass"] is False
    assert v["ratio"] <= 2.0
    assert v["off_n_fits"] == len(off)


def test_off_pulse_null_passes_when_off_is_white():
    # DSA arm C: off-pulse effectively white -> few/no consistent fits.
    v = guards.off_pulse_null_verdict(0.448, [])
    assert v["null_pass"] is True


def test_off_pulse_null_passes_when_scales_differ():
    # off-pulse fits exist but at a very different scale from on-pulse.
    v = guards.off_pulse_null_verdict(0.448, [0.02, 0.025, 0.03, 0.022])
    assert v["null_pass"] is True
    assert v["ratio"] > 2.0


def test_off_pulse_null_inconclusive_without_on_width():
    v = guards.off_pulse_null_verdict(None, [0.03, 0.03, 0.03])
    assert v["null_pass"] is None


# --- low-lag excision stability (arm B1 / rec #4) ---------------------------


def test_low_lag_collapse_on_freya():
    # arm B1: 35.19 -> 31.2 (N=2) -> 23.4 (N=3) -> degenerate (N=6, fit fails).
    v = guards.low_lag_stability_verdict(0.03519, {2: 0.03123, 3: 0.02338, 6: None})
    assert v["stable"] is False
    assert 6 in v["failed_ks"]  # degenerate refit


def test_low_lag_stable_on_dsa():
    # arm C: gamma 713/710/629/615/554 kHz at N=2/3/4/6/8 -> wing survives.
    v = guards.low_lag_stability_verdict(
        0.713, {2: 0.713, 3: 0.710, 4: 0.629, 6: 0.615, 8: 0.554}
    )
    assert v["stable"] is True
    assert v["failed_ks"] == []
    assert v["min_ratio"] > 0.5


def test_low_lag_inconclusive_without_full_width():
    v = guards.low_lag_stability_verdict(None, {2: 0.03})
    assert v["stable"] is None


# --- harmonic-mask systematic (rec #5) --------------------------------------


def test_harmonic_systematic_reports_band_not_correction():
    # freya B1: 35.19 (unmasked) -> 42.21 (masked); ~20% systematic.
    v = guards.harmonic_mask_systematic(0.03519, 0.04221)
    assert v["dnu_unmasked_mhz"] == 0.03519
    assert v["dnu_masked_mhz"] == 0.04221
    assert abs(v["systematic_frac"] - (0.04221 - 0.03519) / 0.03519) < 1e-9


# --- combined finalize ------------------------------------------------------


def test_finalize_freya_full_stack_still_diagnostic():
    # The actual freya case: every mitigation enabled, but the null fails AND
    # the wing collapses -> must NOT be reported as a measurement.
    prov = guards.chime_provenance_status(_chime_cfg())
    null = guards.off_pulse_null_verdict(0.03519, [0.038, 0.033, 0.038, 0.035, 0.039])
    stab = guards.low_lag_stability_verdict(0.03519, {2: 0.031, 3: 0.023, 6: None})
    f = guards.finalize_measurement_status(prov, off_pulse_null=null, low_lag_stability=stab)
    assert f["status"] == guards.DIAGNOSTIC_ONLY
    assert f["downgraded"] is True
    assert set(f["failed_checks"]) == {"off_pulse_null", "low_lag_stability"}


def test_finalize_clean_chime_is_measurement():
    prov = guards.chime_provenance_status(_chime_cfg())
    null = guards.off_pulse_null_verdict(0.448, [])  # passes
    stab = guards.low_lag_stability_verdict(0.713, {2: 0.71, 3: 0.70})  # stable
    f = guards.finalize_measurement_status(prov, off_pulse_null=null, low_lag_stability=stab)
    assert f["status"] == guards.MEASUREMENT
    assert f["downgraded"] is False


def test_finalize_non_chime_ignores_physical_checks():
    prov = guards.chime_provenance_status({"telescope": "dsa", "analysis": {}})
    # even a (hypothetical) failing null must not demote a DSA result via this gate
    null = {"null_pass": False}
    f = guards.finalize_measurement_status(prov, off_pulse_null=null)
    assert f["status"] == guards.MEASUREMENT
    assert f["downgraded"] is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
