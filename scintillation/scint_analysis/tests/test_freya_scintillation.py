from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

from scint_analysis.core import DynamicSpectrum  # noqa: E402
from scint_analysis.freya_scintillation import (  # noqa: E402
    estimate_structure_bandwidth,
    measure_scintillation_bandwidth,
    run_notebook_style_analysis,
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
    assert np.isnan(estimate.structure_function[1])
    assert np.isnan(estimate.structure_function[2])
    assert np.isnan(estimate.structure_function[3])
    assert np.isfinite(estimate.structure_function[4])


def test_structure_bandwidth_returns_channel_scaled_half_power_estimate():
    spectrum, channel_width_mhz = _synthetic_scintillating_spectrum()

    estimate = estimate_structure_bandwidth(spectrum, channel_width_mhz=channel_width_mhz)

    assert estimate.delta_nu_mhz > channel_width_mhz
    assert estimate.method == "half_power"
    assert estimate.lag_index > 0
    assert len(estimate.structure_function) == spectrum.size


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
