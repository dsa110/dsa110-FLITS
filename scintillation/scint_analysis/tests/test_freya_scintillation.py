from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

from scint_analysis import freya_scintillation  # noqa: E402
from scint_analysis.core import ACF, DynamicSpectrum  # noqa: E402
from scint_analysis.freya_scintillation import (  # noqa: E402
    estimate_structure_bandwidth,
    measure_scintillation_bandwidth,
    run_notebook_style_analysis,
    to_jsonable,
)


def _synthetic_scintillating_spectrum(nchan: int = 512) -> tuple[np.ndarray, float]:
    channel_width_mhz = 0.02
    freq = np.arange(nchan, dtype=float) * channel_width_mhz
    rng = np.random.default_rng(17)
    white = rng.normal(0.0, 1.0, nchan)
    kernel_lags = np.arange(-60, 61)
    kernel = np.exp(-0.5 * (kernel_lags / 9.0) ** 2)
    kernel /= kernel.sum()
    scint = np.convolve(white, kernel, mode="same")
    envelope = 1.0 + 0.08 * (freq - freq.mean()) / np.ptp(freq)
    spectrum = 100.0 * envelope * (1.0 + 0.2 * scint / np.nanstd(scint))
    return spectrum, channel_width_mhz


def _synthetic_dynamic_spectrum() -> DynamicSpectrum:
    spectrum, channel_width_mhz = _synthetic_scintillating_spectrum()
    nchan = spectrum.size
    nt = 96
    freqs = 1300.0 + np.arange(nchan, dtype=float) * channel_width_mhz
    times = np.arange(nt, dtype=float) * 0.001
    profile = np.exp(-0.5 * ((np.arange(nt) - 48.0) / 4.0) ** 2)
    noise = np.random.default_rng(23).normal(0.0, 0.4, (nchan, nt))
    power = noise + spectrum[:, None] * profile[None, :]
    return DynamicSpectrum(power, freqs, times)


def test_measure_scintillation_bandwidth_recovers_positive_lorentzian_width():
    spectrum, channel_width_mhz = _synthetic_scintillating_spectrum()

    result = measure_scintillation_bandwidth(
        spectrum,
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )

    assert result.success
    assert 0.05 < result.delta_nu_mhz < 1.0
    assert np.isfinite(result.delta_nu_err_mhz)
    assert result.channel_width_mhz == channel_width_mhz
    assert result.modulation_index > 0.0


def test_masked_channels_are_excluded_from_bandwidth_estimates():
    spectrum, channel_width_mhz = _synthetic_scintillating_spectrum()
    mask = np.zeros(spectrum.size, dtype=bool)
    mask[128] = True
    clean_masked = np.ma.masked_array(spectrum.copy(), mask=mask)
    polluted_masked = np.ma.masked_array(spectrum.copy(), mask=mask)
    polluted_masked.data[128] = 1.0e9

    clean_acf = measure_scintillation_bandwidth(
        clean_masked,
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )
    polluted_acf = measure_scintillation_bandwidth(
        polluted_masked,
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )
    clean_structure = estimate_structure_bandwidth(
        clean_masked,
        channel_width_mhz=channel_width_mhz,
    )
    polluted_structure = estimate_structure_bandwidth(
        polluted_masked,
        channel_width_mhz=channel_width_mhz,
    )

    assert polluted_acf.success == clean_acf.success
    assert polluted_acf.modulation_index == pytest.approx(clean_acf.modulation_index)
    assert polluted_acf.delta_nu_mhz == pytest.approx(clean_acf.delta_nu_mhz)
    assert polluted_structure.delta_nu_mhz == pytest.approx(clean_structure.delta_nu_mhz)
    assert polluted_structure.structure_function == pytest.approx(clean_structure.structure_function)


def test_structure_bandwidth_requires_valid_pairs_per_lag():
    spectrum = np.linspace(1.0, 2.0, 96)
    mask = np.ones(spectrum.size, dtype=bool)
    mask[::4] = False

    estimate = estimate_structure_bandwidth(
        np.ma.masked_array(spectrum, mask=mask),
        channel_width_mhz=0.02,
    )

    assert estimate.structure_function[0] == pytest.approx(0.0)
    assert estimate.structure_function[1] is None
    assert estimate.structure_function[2] is None
    assert estimate.structure_function[3] is None
    assert estimate.structure_function[4] is not None


def test_structure_bandwidth_serializes_unsupported_lags_as_strict_json_nulls():
    spectrum = np.linspace(1.0, 2.0, 96)
    mask = np.ones(spectrum.size, dtype=bool)
    mask[::4] = False
    estimate = estimate_structure_bandwidth(
        np.ma.masked_array(spectrum, mask=mask),
        channel_width_mhz=0.02,
    )

    payload = json.dumps(to_jsonable(estimate), allow_nan=False)
    decoded = json.loads(payload)

    assert decoded["structure_function"][1] is None
    assert decoded["structure_function"][2] is None
    assert decoded["structure_function"][3] is None


def test_structure_bandwidth_returns_channel_scaled_half_power_estimate():
    spectrum, channel_width_mhz = _synthetic_scintillating_spectrum()

    estimate = estimate_structure_bandwidth(spectrum, channel_width_mhz=channel_width_mhz)

    assert estimate.delta_nu_mhz > channel_width_mhz
    assert estimate.method == "half_power"
    assert estimate.lag_index > 0
    assert len(estimate.structure_function) == spectrum.size


def _lorentzian_acf_object(
    gamma_true_mhz: float,
    channel_width_mhz: float = 0.02,
    max_lag_mhz: float = 2.0,
    noise_rms: float = 0.005,
) -> ACF:
    nlag = int(max_lag_mhz / channel_width_mhz)
    lags = np.arange(-nlag, nlag + 1, dtype=float) * channel_width_mhz
    acf = 0.8 / (1.0 + (lags / gamma_true_mhz) ** 2) + 0.05
    acf = acf + np.random.default_rng(41).normal(0.0, noise_rms, lags.size)
    err = np.full(lags.size, 0.02)
    return ACF(acf, lags, err)


def test_lorentzian_fit_recovers_known_width_within_tolerance(monkeypatch):
    gamma_true_mhz = 0.4
    channel_width_mhz = 0.02
    acf_obj = _lorentzian_acf_object(gamma_true_mhz, channel_width_mhz)
    monkeypatch.setattr(freya_scintillation, "calculate_acf", lambda *a, **k: acf_obj)

    result = measure_scintillation_bandwidth(
        np.full(512, 100.0),
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.5,
    )

    assert result.success
    assert result.message == "ok"
    assert result.delta_nu_mhz == pytest.approx(gamma_true_mhz, rel=0.02)
    assert result.delta_nu_err_mhz is not None


def test_boundary_pinned_width_is_reported_as_failed_fit(monkeypatch):
    # True width far beyond the fit range: gamma rails at the fit_lag_mhz bound
    # and must not be emitted as a measured bandwidth (issue #118).
    channel_width_mhz = 0.02
    acf_obj = _lorentzian_acf_object(5.0, channel_width_mhz, max_lag_mhz=2.0)
    monkeypatch.setattr(freya_scintillation, "calculate_acf", lambda *a, **k: acf_obj)

    result = measure_scintillation_bandwidth(
        np.full(512, 100.0),
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )

    assert not result.success
    assert result.delta_nu_mhz is None
    assert result.delta_nu_err_mhz is None
    assert "upper bound" in result.message
    assert "lower limit" in result.message
    assert result.acf_model


def test_lower_bound_pinned_width_is_reported_as_failed_fit(monkeypatch):
    # Deterministic branch pin: a solver solution at the 0.25*channel_width
    # lower bound is unresolved by channelization, not a measurement.
    channel_width_mhz = 0.02
    acf_obj = _lorentzian_acf_object(0.4, channel_width_mhz)
    monkeypatch.setattr(freya_scintillation, "calculate_acf", lambda *a, **k: acf_obj)
    monkeypatch.setattr(
        freya_scintillation,
        "curve_fit",
        lambda *a, **k: (np.array([0.8, 0.25 * channel_width_mhz, 0.05]), np.eye(3) * 1e-6),
    )

    result = measure_scintillation_bandwidth(
        np.full(512, 100.0),
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.5,
    )

    assert not result.success
    assert result.delta_nu_mhz is None
    assert "lower bound" in result.message
    assert "channelization" in result.message


def test_nonfinite_covariance_is_reported_as_failed_fit(monkeypatch):
    channel_width_mhz = 0.02
    acf_obj = _lorentzian_acf_object(0.4, channel_width_mhz)
    monkeypatch.setattr(freya_scintillation, "calculate_acf", lambda *a, **k: acf_obj)
    monkeypatch.setattr(
        freya_scintillation,
        "curve_fit",
        lambda *a, **k: (np.array([0.8, 0.4, 0.05]), np.full((3, 3), np.nan)),
    )

    result = measure_scintillation_bandwidth(
        np.full(512, 100.0),
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.5,
    )

    assert not result.success
    assert result.delta_nu_mhz is None
    assert "non-finite covariance" in result.message


def _baseline_guard_cfg(noise_window: list[int]) -> dict:
    return {
        "input_data_path": "unused.npz",
        "pipeline_options": {"downsample": {"f_factor": 1, "t_factor": 1}},
        "analysis": {
            "rfi_masking": {
                "manual_burst_window": [40, 56],
                "manual_noise_window": noise_window,
            },
            "baseline_subtraction": {"enable": True, "poly_order": 1},
        },
    }


@pytest.mark.parametrize(
    ("noise_window", "expect_subtraction"),
    # 50/51 pin the exact pipeline.py:178 edge (`> off_pulse_lims[0] + 50`).
    [([0, 30], False), ([0, 50], False), ([0, 51], True), ([0, 90], True)],
)
def test_baseline_subtraction_requires_pipeline_scale_off_window(
    monkeypatch, noise_window, expect_subtraction
):
    # Guard must match pipeline.py: >50 off-pulse bins, else skip with a warning.
    calls: list[int] = []

    def spy_subtract(self, off_pulse_spectrum, poly_order=1):
        calls.append(poly_order)
        return self, None

    monkeypatch.setattr(
        DynamicSpectrum,
        "from_numpy_file",
        classmethod(lambda cls, path: _synthetic_dynamic_spectrum()),
    )
    monkeypatch.setattr(DynamicSpectrum, "mask_rfi", lambda self, cfg: self)
    monkeypatch.setattr(DynamicSpectrum, "subtract_poly_baseline", spy_subtract)

    masked, burst_lims, off_lims = freya_scintillation.prepare_spectrum_from_config(
        _baseline_guard_cfg(noise_window)
    )

    assert burst_lims == (40, 56)
    assert off_lims == (noise_window[0], noise_window[1])
    assert (len(calls) == 1) is expect_subtraction


def test_run_notebook_style_analysis_writes_json_and_figures(tmp_path):
    ds = _synthetic_dynamic_spectrum()

    result = run_notebook_style_analysis(
        ds,
        burst_id="freya-test",
        burst_lims=(43, 54),
        off_pulse_lims=(0, 30),
        output_dir=tmp_path,
        write_figures=True,
    )

    result_path = tmp_path / "freya-test_scintillation.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text())
    assert payload["burst_id"] == "freya-test"
    assert payload["acf"]["success"] is True
    assert payload["figures"]
    for fig in payload["figures"]:
        assert (tmp_path / fig["path"]).exists()
        assert fig["kind"] in {"dynamic_spectrum", "acf", "structure_function"}
    assert result.acf.success
