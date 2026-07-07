"""DM-power residual estimator tests.

These tests lock down the behavior needed for the CHIME/DSA co-detection
cross-check before the implementation exists: physical dispersion scale,
residual-vs-absolute DM semantics, no default circular wrapping, and honest
non-detections.
"""

from __future__ import annotations

import numpy as np
import pytest

from dispersion.chime_dm import K_DM
from dispersion.dm_power_analysis import (
    DEFAULT_DM_STEP,
    _dm_ref_source,
    _freq_grid_source,
    _orient_waterfall_to_ascending_frequency,
    fit_dm_power_result,
    mark_diagnostic_candidate_only,
    measure_dm_power,
    residual_delay_s,
    shift_waterfall_residual_dm,
)


def _inject_structured_waterfall(
    *,
    freqs: np.ndarray,
    dt_s: float,
    ntime: int,
    residual_dm: float,
    components: list[tuple[float, float, float]],
    noise_sigma: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Return a residual-dispersed `(nchan, ntime)` waterfall."""
    rng = np.random.default_rng(seed)
    t = np.arange(ntime) * dt_s
    nu_ref = float(freqs.max())
    wf = np.zeros((freqs.size, ntime), dtype=float)
    delay = K_DM * residual_dm * (1.0 / freqs**2 - 1.0 / nu_ref**2)
    for j, arr_delay in enumerate(delay):
        for t0, amp, width_s in components:
            wf[j] += amp * np.exp(-0.5 * ((t - t0 - arr_delay) / width_s) ** 2)
    wf += noise_sigma * rng.standard_normal(wf.shape)
    return wf


def test_residual_delay_uses_physical_scale():
    freqs = np.array([400.0, 800.0])
    delay = residual_delay_s(freqs, residual_dm=20.0, nu_ref_mhz=800.0)
    expected = K_DM * 20.0 * (1.0 / freqs**2 - 1.0 / 800.0**2)
    assert delay == pytest.approx(expected)
    assert delay[0] == pytest.approx(0.38895, rel=5e-3)


def test_measure_dm_power_reports_absolute_dm_from_residual_peak():
    freqs = np.linspace(400.0, 800.0, 48)
    dt_s = 5.0e-4
    dm_ref = 500.0
    residual_true = 8.0
    wf = _inject_structured_waterfall(
        freqs=freqs,
        dt_s=dt_s,
        ntime=1600,
        residual_dm=residual_true,
        components=[(0.72, 3.0, 1.2e-3), (0.735, 2.4, 1.2e-3)],
        seed=2,
    )
    result = measure_dm_power(
        wf,
        freqs,
        dt_s,
        dm_ref,
        np.arange(-12.0, 12.1, 1.0),
        n_boot=24,
        random_state=3,
    )

    assert result["constrains_dm"], result["reason"]
    assert result["residual_dm_best"] == pytest.approx(residual_true, abs=1.5)
    assert result["dm"] == pytest.approx(dm_ref + residual_true, abs=1.5)
    assert result["dm"] != pytest.approx(result["residual_dm_best"])
    assert result["dm_ref"] == pytest.approx(dm_ref)


def test_zero_fill_shift_does_not_wrap_edges_by_default():
    freqs = np.array([400.0, 800.0])
    dt_s = 1.0e-3
    wf = np.zeros((2, 32), dtype=float)
    wf[0, 30] = 1.0
    shifted = shift_waterfall_residual_dm(
        wf,
        freqs,
        dt_s,
        residual_dm=1.6,
        nu_ref_mhz=800.0,
        mode="zero_fill",
    )
    circular = shift_waterfall_residual_dm(
        wf,
        freqs,
        dt_s,
        residual_dm=1.6,
        nu_ref_mhz=800.0,
        mode="fourier_circular",
    )

    assert shifted[0, :5].sum() == pytest.approx(0.0)
    assert np.sum(np.abs(circular[0, :5])) > 0.1


def test_smooth_low_snr_result_is_unconstrained():
    rng = np.random.default_rng(4)
    freqs = np.linspace(400.0, 800.0, 48)
    dt_s = 1.0e-3
    t = np.arange(512) * dt_s
    smooth = np.exp(-0.5 * ((t - 0.25) / 0.04) ** 2)
    wf = np.tile(smooth, (freqs.size, 1))
    wf += 0.8 * rng.standard_normal(wf.shape)

    result = measure_dm_power(
        wf,
        freqs,
        dt_s,
        500.0,
        np.arange(-5.0, 5.1, 1.0),
        n_boot=16,
        random_state=5,
    )

    assert not result["constrains_dm"]
    assert result["dm"] is None
    assert result["dm_err"] is None
    assert result["residual_dm_best"] is None


def test_fit_uses_consistent_delay_bin_peaks_not_global_score_threshold():
    grid = np.arange(-4.0, 4.1, 1.0)
    true_residual = 1.0
    log_power = np.empty((grid.size, 5, 4), dtype=float)
    for b in range(log_power.shape[1]):
        for k in range(log_power.shape[2]):
            center = true_residual + 0.03 * (k - 1.5) + 0.01 * b
            log_power[:, b, k] = 10.0 * np.exp(-0.5 * ((grid - center) / 0.6) ** 2)
    curve = {
        "residual_dm_grid": grid,
        "log_power": log_power,
        "score": np.tile(np.linspace(1.0, 1.1, grid.size)[:, None], (1, log_power.shape[1])),
        "nu_ref_mhz": 800.0,
        "dt_s": 1.0e-3,
        "shift_mode": "zero_fill",
    }

    result = fit_dm_power_result(curve, dm_ref=500.0)

    assert result["peak_snr"] < 2.5
    assert result["constrains_dm"], result["reason"]
    assert result["residual_dm_best"] == pytest.approx(true_residual, abs=0.2)


def test_global_score_curve_max_has_bootstrap_and_grid_floor_error():
    grid = np.arange(-1.0, 1.01, 0.25)
    score = np.empty((grid.size, 4), dtype=float)
    centers = [-0.25, 0.0, 0.0, 0.25]
    for b, center in enumerate(centers):
        score[:, b] = np.exp(-0.5 * ((grid - center) / 0.18) ** 2)
    curve = {
        "residual_dm_grid": grid,
        "log_power": np.tile(score[:, :, None], (1, 1, 4)),
        "score": score,
        "nu_ref_mhz": 800.0,
        "dt_s": 1.0e-3,
        "shift_mode": "zero_fill",
    }

    result = fit_dm_power_result(curve, dm_ref=100.0)
    marked = mark_diagnostic_candidate_only(result)

    assert result["global_score_grid_max_residual_dm"] == pytest.approx(0.0)
    assert result["global_score_grid_max_err"] >= 0.125
    assert marked["dm_power_curve_max_dm"] == pytest.approx(100.0)
    assert marked["dm_power_curve_max_err"] == pytest.approx(result["global_score_grid_max_err"])
    assert marked["dm_power_candidate_err"] == pytest.approx(result["global_score_grid_max_err"])


def test_default_dm_trial_spacing_is_fine_enough_for_diagnostic_curves():
    assert DEFAULT_DM_STEP <= 0.05


def test_dm_power_manifest_rows_carry_explicit_provenance_sources():
    chime_row = {"telescope": "chime", "side_input": {"dm_dsa": 462.174}}
    dsa_row = {"telescope": "dsa", "fixture": {"dm": 462.174}}

    assert _dm_ref_source(chime_row) == "crossmatching/chime_side_inputs.json:dm_dsa"
    assert _dm_ref_source(dsa_row) == "crossmatching/notebook_reproduction_fixture.json:dm"
    assert "CHIME_DF_MHZ" in _freq_grid_source("chime", 32768)
    assert "DSA_FCH1_MHZ" in _freq_grid_source("dsa", 6144)


def test_manifest_cube_rows_are_flipped_to_ascending_frequency_order():
    raw_descending = np.array([[800.0, 801.0], [600.0, 601.0], [400.0, 401.0]])

    oriented = _orient_waterfall_to_ascending_frequency(raw_descending, "chime")

    assert oriented.tolist() == [[400.0, 401.0], [600.0, 601.0], [800.0, 801.0]]


def test_diagnostic_candidate_marking_preserves_peak_without_accepting_dm():
    result = {
        "dm": 602.29,
        "dm_err": 0.14,
        "residual_dm_best": -0.05,
        "dm_ref": 602.346,
        "global_score_residual_dm": 0.25,
        "global_score_grid_max_residual_dm": 0.0,
        "global_score_grid_max_err": 0.03,
        "constrains_dm": True,
        "reason": "ok",
    }

    marked = mark_diagnostic_candidate_only(result)

    assert not marked["constrains_dm"]
    assert marked["dm"] is None
    assert marked["dm_err"] is None
    assert marked["residual_dm_best"] is None
    assert marked["dm_power_curve_max_dm"] == pytest.approx(602.346)
    assert marked["dm_power_candidate_dm"] == pytest.approx(602.346)
    assert marked["dm_power_per_bin_candidate_dm"] == pytest.approx(602.29)
    assert marked["dm_power_curve_max_err"] == pytest.approx(0.03)
    assert marked["dm_power_candidate_err"] == pytest.approx(0.03)
    assert marked["dm_power_per_bin_candidate_err"] == pytest.approx(0.14)
    assert "diagnostic" in marked["reason"]


def test_shape_orientation_is_validated():
    wf = np.zeros((8, 64))
    freqs = np.linspace(400.0, 800.0, 7)
    with pytest.raises(ValueError, match="freq_mhz"):
        measure_dm_power(wf, freqs, 1.0e-3, 100.0, np.arange(-1.0, 1.1, 1.0))
