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


def _synthetic_rippled_dynamic_spectrum() -> tuple[DynamicSpectrum, float, float]:
    """Ripple x scintillation dynamic spectrum for the flat-fielding test.

    Multiplicative PFB model: power = G(f) * (noise + burst * I(f)), where G is
    a strong static coarse-channel scallop, I(f) carries a known-width smooth
    scintillation pattern, and the burst is confined to a few time bins. The
    off-pulse mean of channel f is G(f)*<noise>, so dividing by it must cancel
    G(f) exactly while leaving I(f) intact.
    """
    channel_width_mhz = 0.02
    nchan = 1024
    nt = 128
    freq = np.arange(nchan, dtype=float) * channel_width_mhz  # ascending -> no flip
    rng = np.random.default_rng(7)

    # Narrow scintillation kernel (width ~5 ch) kept well below the ripple
    # period so the intrinsic ACF is negligible one ripple-period out; the only
    # thing living at that lag is the scallop.
    white = rng.normal(0.0, 1.0, nchan)
    kernel_lags = np.arange(-40, 41)
    kernel = np.exp(-0.5 * (kernel_lags / 5.0) ** 2)
    kernel /= kernel.sum()
    scint = np.convolve(white, kernel, mode="same")
    intrinsic = 1.0 + 0.4 * scint / np.nanstd(scint)  # I(f), known width

    ripple_period_chan = 40.0
    scallop = 1.0 + 0.6 * np.cos(2.0 * np.pi * np.arange(nchan) / ripple_period_chan)

    times = np.arange(nt, dtype=float) * 0.001
    burst = np.exp(-0.5 * ((np.arange(nt) - 96.0) / 3.0) ** 2)
    noise = rng.normal(0.0, 0.05, (nchan, nt)) + 1.0  # positive off-pulse floor
    sky = noise + 8.0 * burst[None, :] * intrinsic[:, None]
    power = scallop[:, None] * sky
    return DynamicSpectrum(power, freq, times), channel_width_mhz, ripple_period_chan


def _acf_at_lag(spectrum_1d: np.ma.MaskedArray, lag_bins: int) -> float:
    y = np.ma.masked_invalid(spectrum_1d)
    y = y - np.ma.mean(y)
    a0 = float(np.ma.sum(y * y))
    a_lag = float(np.ma.sum(y[:-lag_bins] * y[lag_bins:]))
    return a_lag / a0


def test_bandpass_normalization_removes_ripple_and_recovers_width():
    ds, channel_width_mhz, ripple_period_chan = _synthetic_rippled_dynamic_spectrum()
    off_lims = (0, 80)
    on_lims = (88, 105)

    raw_on = ds.get_spectrum(on_lims)
    normed = freya_scintillation.normalize_bandpass(ds, off_lims)
    normed_on = normed.get_spectrum(on_lims)

    # A periodic scallop anti-correlates strongly at half its period (cos(pi) =
    # -1); broadband scintillation does not. That half-period dip is therefore a
    # clean signature of the ripple alone, and flat-fielding must erase it.
    half_period = int(ripple_period_chan // 2)
    raw_half = _acf_at_lag(raw_on, half_period)
    normed_half = _acf_at_lag(normed_on, half_period)
    assert raw_half < -0.3
    assert abs(normed_half) < 0.1

    # The known-width central scintillation component survives the flat-field.
    result = measure_scintillation_bandwidth(
        normed_on,
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )
    assert result.success
    assert channel_width_mhz < result.delta_nu_mhz < 1.0


def test_bandpass_normalization_masks_gain_starved_channels():
    ds, _channel_width_mhz, _ripple = _synthetic_rippled_dynamic_spectrum()
    ds.power.data[10, :] = 0.0  # zero off-pulse mean -> below floor
    ds.power.data[20, :] = -3.0  # negative off-pulse mean
    off_lims = (0, 80)

    normed = freya_scintillation.normalize_bandpass(ds, off_lims)
    mask = np.ma.getmaskarray(normed.power)

    assert mask[10].all()
    assert mask[20].all()
    # A healthy channel is normalised to ~1 in its off-pulse window, not masked.
    assert not mask[15].all()
    healthy = np.ma.mean(normed.power[15, off_lims[0] : off_lims[1]])
    assert np.isfinite(healthy)
    assert healthy == pytest.approx(1.0, abs=0.05)


def test_bandpass_normalization_requires_sufficient_off_pulse_bins():
    ds, _channel_width_mhz, _ripple = _synthetic_rippled_dynamic_spectrum()
    with pytest.raises(ValueError, match="off-pulse time bins"):
        freya_scintillation.normalize_bandpass(ds, (0, 40))  # 40 < 50-bin floor


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
    # No scan requested: the fields exist and are null (JSON contract stable).
    assert payload["fit_window_scan"] is None
    assert payload["fit_window_systematic_mhz"] is None


def _synthetic_gapped_dynamic_spectrum() -> tuple[DynamicSpectrum, DynamicSpectrum, float, np.ndarray]:
    """A scintillating dynamic spectrum plus a gapped copy with 25% of its
    channels deleted in blocks (mimicking missing coarse channels), so the
    gapped mean spacing overstates the native step by ~1.33x."""
    spectrum, channel_width_mhz = _synthetic_scintillating_spectrum()
    nchan = spectrum.size
    nt = 96
    freqs = 1300.0 + np.arange(nchan, dtype=float) * channel_width_mhz
    times = np.arange(nt, dtype=float) * 0.001
    profile = np.exp(-0.5 * ((np.arange(nt) - 48.0) / 4.0) ** 2)
    noise = np.random.default_rng(23).normal(0.0, 0.4, (nchan, nt))
    power = noise + spectrum[:, None] * profile[None, :]
    full = DynamicSpectrum(power.copy(), freqs.copy(), times.copy())

    keep = np.ones(nchan, dtype=bool)
    keep[80:120] = False
    keep[230:280] = False
    keep[390:428] = False
    gapped = DynamicSpectrum(power[keep], freqs[keep], times.copy())
    return full, gapped, channel_width_mhz, keep


def test_regularize_frequency_grid_noop_on_uniform_grid():
    ds = _synthetic_dynamic_spectrum()
    assert freya_scintillation.regularize_frequency_grid(ds) is ds


def test_regularize_frequency_grid_restores_axis_and_unbiases_width():
    full, gapped, native, keep = _synthetic_gapped_dynamic_spectrum()
    stretch = gapped.channel_width_mhz / native
    assert stretch > 1.25  # the gapped mean spacing is materially wrong

    reg = freya_scintillation.regularize_frequency_grid(gapped)
    assert reg.num_channels == full.num_channels
    assert reg.channel_width_mhz == pytest.approx(native, rel=1e-9)
    assert np.allclose(reg.frequencies, full.frequencies)
    # Filler rows are fully masked; surviving rows carry the original data.
    mask = np.ma.getmaskarray(reg.power)
    assert mask[~keep].all()
    assert not mask[keep].any()
    assert np.allclose(reg.power.data[keep], gapped.power.data)

    on_lims = (43, 54)
    w_full = measure_scintillation_bandwidth(
        full.get_spectrum(on_lims),
        channel_width_mhz=full.channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )
    w_gap = measure_scintillation_bandwidth(
        gapped.get_spectrum(on_lims),
        channel_width_mhz=gapped.channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )
    w_reg = measure_scintillation_bandwidth(
        reg.get_spectrum(on_lims),
        channel_width_mhz=reg.channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )
    assert w_full.success and w_gap.success and w_reg.success
    # The deterministic claim: on IDENTICAL data the gapped path overstates
    # the regularized width by the axis-stretch factor (mean/native spacing).
    assert w_gap.delta_nu_mhz / w_reg.delta_nu_mhz == pytest.approx(stretch, rel=0.15)
    # Sanity anchor only: deleting 25% of a finite-scintle realization moves
    # the fitted width (sample variance), so the ungapped-truth comparison is
    # deliberately loose.
    assert w_reg.delta_nu_mhz == pytest.approx(w_full.delta_nu_mhz, rel=0.5)


def test_regularize_frequency_grid_snaps_interblock_offsets():
    _full, gapped, native, _keep = _synthetic_gapped_dynamic_spectrum()
    freqs = gapped.frequencies.copy()
    # Emulate upchannelized inter-block registration drift: shift everything
    # above the first gap by 0.3 of a fine channel.
    freqs[80:] += 0.3 * native
    drifted = DynamicSpectrum(gapped.power.data.copy(), freqs, gapped.times.copy())

    reg = freya_scintillation.regularize_frequency_grid(drifted)
    diffs = np.diff(reg.frequencies)
    assert np.allclose(diffs, diffs[0])
    assert reg.channel_width_mhz == pytest.approx(native, rel=1e-6)
    # The uniform axis must come from NaN-filled re-embedding, not from
    # compacting the gapped axis: full grid length restored, exactly the
    # missing rows masked.
    assert reg.num_channels == 512
    filler_rows = int(np.ma.getmaskarray(reg.power).all(axis=1).sum())
    assert filler_rows == 512 - drifted.num_channels


def test_regularize_frequency_grid_rejects_colliding_channels():
    _full, gapped, native, _keep = _synthetic_gapped_dynamic_spectrum()
    freqs = gapped.frequencies.copy()
    # Push one channel onto its neighbour's grid position: a compressed axis
    # must be rejected, never silently merged.
    freqs[40] = freqs[41] - 0.1 * native
    bad = DynamicSpectrum(gapped.power.data.copy(), freqs, gapped.times.copy())
    with pytest.raises(ValueError, match="same grid position"):
        freya_scintillation.regularize_frequency_grid(bad)


@pytest.mark.parametrize(
    ("enable", "f_factor"), [(False, 1), (True, 1), (True, 2)]
)
def test_grid_regularization_config_wiring_and_warning(monkeypatch, caplog, enable, f_factor):
    _full, gapped, native, _keep = _synthetic_gapped_dynamic_spectrum()
    monkeypatch.setattr(
        DynamicSpectrum,
        "from_numpy_file",
        classmethod(lambda cls, path: gapped),
    )
    monkeypatch.setattr(DynamicSpectrum, "mask_rfi", lambda self, cfg: self)
    cfg = {
        "input_data_path": "unused.npz",
        "pipeline_options": {"downsample": {"f_factor": f_factor, "t_factor": 1}},
        "analysis": {
            "rfi_masking": {
                "manual_burst_window": [43, 54],
                "manual_noise_window": [0, 30],
            },
            "grid_regularization": {"enable": enable},
        },
    }

    with caplog.at_level("WARNING", logger=freya_scintillation.log.name):
        masked, _burst, _off = freya_scintillation.prepare_spectrum_from_config(cfg)

    if enable:
        # Regularization runs BEFORE downsampling: the full 512-channel grid is
        # what gets block-averaged (512//f), never the compact gapped axis
        # (384//f). At f_factor=2 that ordering is the only way to get 256.
        assert masked.num_channels == 512 // f_factor
        assert masked.channel_width_mhz == pytest.approx(f_factor * native, rel=1e-9)
        assert not any("grid_regularization" in rec.message for rec in caplog.records)
    else:
        assert masked.channel_width_mhz > 1.25 * native
        assert any(
            "Enable analysis.grid_regularization" in rec.getMessage()
            for rec in caplog.records
        )


def test_modulation_index_acf_reports_physical_depth():
    # On-spectrum = burst * intrinsic pattern + additive noise floor. The
    # std/mean diagnostic is diluted by the floor in its denominator; the
    # ACF-amplitude estimate divides by (mean_on - off_mean)^2 and recovers
    # the physical modulation of the burst term.
    spectrum, channel_width_mhz = _synthetic_scintillating_spectrum()
    floor = float(np.mean(spectrum))  # comparable to the burst term
    with_floor = spectrum + floor
    m_true = float(np.std(spectrum) / np.mean(spectrum))

    result = measure_scintillation_bandwidth(
        with_floor,
        channel_width_mhz=channel_width_mhz,
        off_burst_spectrum_mean=floor,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )

    assert result.success
    assert result.modulation_index_acf is not None
    # The diluted diagnostic underestimates; the ACF estimate matches truth.
    assert result.modulation_index < 0.75 * m_true
    assert result.modulation_index_acf == pytest.approx(m_true, rel=0.25)


def test_fit_window_scan_records_windows_and_systematic(tmp_path):
    ds = _synthetic_dynamic_spectrum()

    result = run_notebook_style_analysis(
        ds,
        burst_id="freya-scan",
        burst_lims=(43, 54),
        off_pulse_lims=(0, 30),
        output_dir=tmp_path,
        fit_lag_scan_mhz=[1.0, 0.5, 0.01],  # 0.01 < one channel: invalid window
        write_figures=False,
    )

    payload = json.loads((tmp_path / "freya-scan_scintillation.json").read_text())
    scan = payload["fit_window_scan"]
    assert [entry["fit_lag_mhz"] for entry in scan] == [1.0, 0.5, 0.01]
    assert scan[0]["success"] is True and scan[1]["success"] is True
    assert scan[2]["success"] is False
    assert "invalid fit window" in scan[2]["message"]
    widths = [entry["delta_nu_mhz"] for entry in scan[:2]]
    assert payload["fit_window_systematic_mhz"] == pytest.approx(
        max(widths) - min(widths)
    )
    assert result.fit_window_systematic_mhz is not None


def test_fit_window_scan_survives_non_finite_entries(tmp_path):
    # A NaN from YAML must become a visible failed record, not crash the
    # strict allow_nan=False JSON writer.
    ds = _synthetic_dynamic_spectrum()

    run_notebook_style_analysis(
        ds,
        burst_id="freya-nan-scan",
        burst_lims=(43, 54),
        off_pulse_lims=(0, 30),
        output_dir=tmp_path,
        fit_lag_scan_mhz=[1.0, float("nan")],
        write_figures=False,
    )

    payload = json.loads((tmp_path / "freya-nan-scan_scintillation.json").read_text())
    scan = payload["fit_window_scan"]
    assert scan[0]["success"] is True
    assert scan[1] == {
        "fit_lag_mhz": None,
        "success": False,
        "delta_nu_mhz": None,
        "delta_nu_err_mhz": None,
        "message": "invalid fit window: non-finite value nan",
    }


def test_modulation_index_acf_requires_off_pulse_level():
    # Without an off-pulse mean the ACF denominator is mean_on^2 and the
    # fitted amplitude is floor-diluted, so the physical field must be None.
    spectrum, channel_width_mhz = _synthetic_scintillating_spectrum()

    result = measure_scintillation_bandwidth(
        spectrum,
        channel_width_mhz=channel_width_mhz,
        max_lag_mhz=2.0,
        fit_lag_mhz=1.0,
    )

    assert result.success
    assert result.modulation_index_acf is None


def _pipeline_cfg(tmp_path, enable_grid: bool) -> dict:
    return {
        "burst_id": "freya-pipe",
        "input_data_path": "unused.npz",
        "pipeline_options": {
            "downsample": {"f_factor": 1, "t_factor": 1},
            "cache_directory": str(tmp_path / "cache"),
        },
        "analysis": {
            "rfi_masking": {
                "manual_burst_window": [43, 54],
                "manual_noise_window": [0, 30],
            },
            "grid_regularization": {"enable": enable_grid},
        },
    }


@pytest.mark.parametrize("enable", [False, True])
def test_pipeline_prepare_data_applies_grid_regularization(monkeypatch, tmp_path, enable):
    # The subband pipeline shares the CLI's gating: an enabled config must not
    # be silently bypassed by ScintillationAnalysis.prepare_data.
    from scint_analysis.pipeline import ScintillationAnalysis

    _full, gapped, native, _keep = _synthetic_gapped_dynamic_spectrum()
    monkeypatch.setattr(
        DynamicSpectrum,
        "from_numpy_file",
        classmethod(lambda cls, path: gapped),
    )
    monkeypatch.setattr(DynamicSpectrum, "mask_rfi", lambda self, cfg: self)

    sa = ScintillationAnalysis(_pipeline_cfg(tmp_path, enable))
    sa.prepare_data()

    if enable:
        assert sa.masked_spectrum.num_channels == 512
        assert sa.masked_spectrum.channel_width_mhz == pytest.approx(native, rel=1e-9)
    else:
        assert sa.masked_spectrum.num_channels == gapped.num_channels
        assert sa.masked_spectrum.channel_width_mhz > 1.25 * native


def test_pipeline_applies_bandpass_normalization():
    from scint_analysis.pipeline import ScintillationAnalysis

    ds, _channel_width_mhz, ripple_period_chan = _synthetic_rippled_dynamic_spectrum()
    off_lims = (0, 80)
    on_lims = (88, 105)
    cfg = {
        "burst_id": "freya-pipe-bp",
        "input_data_path": "unused.npz",
        "analysis": {"bandpass_normalization": {"enable": True}},
    }
    sa = ScintillationAnalysis(cfg)
    sa.masked_spectrum = ds

    sa._apply_bandpass_normalization(off_lims)

    # Same ripple signature as the direct normalize_bandpass test: the
    # half-period anti-correlation of the scallop must be gone.
    half_period = int(ripple_period_chan // 2)
    assert _acf_at_lag(ds.get_spectrum(on_lims), half_period) < -0.3
    assert abs(_acf_at_lag(sa.masked_spectrum.get_spectrum(on_lims), half_period)) < 0.1

    # Flag off: untouched (identity), so existing configs are unaffected.
    sa_off = ScintillationAnalysis({"burst_id": "x", "input_data_path": "y", "analysis": {}})
    sa_off.masked_spectrum = ds
    sa_off._apply_bandpass_normalization(off_lims)
    assert sa_off.masked_spectrum is ds


def test_pipeline_bandpass_normalization_fails_closed_on_short_off_window():
    from scint_analysis.pipeline import ScintillationAnalysis

    ds, _channel_width_mhz, _ripple = _synthetic_rippled_dynamic_spectrum()
    sa = ScintillationAnalysis(
        {
            "burst_id": "freya-pipe-short",
            "input_data_path": "unused.npz",
            "analysis": {"bandpass_normalization": {"enable": True}},
        }
    )
    sa.masked_spectrum = ds
    with pytest.raises(ValueError, match="off-pulse time bins"):
        sa._apply_bandpass_normalization((0, 40))


def test_pipeline_cache_key_tracks_preprocessing_config(tmp_path):
    # Toggling a preprocessing-relevant flag must change the cache identity;
    # an irrelevant option must not.
    from scint_analysis.pipeline import ScintillationAnalysis

    base = _pipeline_cfg(tmp_path, enable_grid=False)
    on = _pipeline_cfg(tmp_path, enable_grid=True)
    irrelevant = _pipeline_cfg(tmp_path, enable_grid=False)
    irrelevant["pipeline_options"]["log_level"] = "DEBUG"

    path_base = ScintillationAnalysis(base)._get_cache_path("processed_spectrum")
    path_on = ScintillationAnalysis(on)._get_cache_path("processed_spectrum")
    path_irrelevant = ScintillationAnalysis(irrelevant)._get_cache_path("processed_spectrum")

    assert path_base != path_on
    assert path_base == path_irrelevant


def test_pipeline_cached_spectrum_not_reused_after_flag_change(monkeypatch, tmp_path):
    # The r2 P1 scenario: run once with regularization off and caching on,
    # then enable the flag -- the stale unregularized pickle must NOT be
    # silently reloaded.
    from scint_analysis.pipeline import ScintillationAnalysis

    _full, gapped, native, _keep = _synthetic_gapped_dynamic_spectrum()
    monkeypatch.setattr(
        DynamicSpectrum,
        "from_numpy_file",
        classmethod(lambda cls, path: gapped),
    )
    monkeypatch.setattr(DynamicSpectrum, "mask_rfi", lambda self, cfg: self)

    cfg_off = _pipeline_cfg(tmp_path, enable_grid=False)
    cfg_off["pipeline_options"]["save_intermediate_steps"] = True
    sa_off = ScintillationAnalysis(cfg_off)
    sa_off.prepare_data()
    assert sa_off.masked_spectrum.num_channels == gapped.num_channels

    cfg_on = _pipeline_cfg(tmp_path, enable_grid=True)
    cfg_on["pipeline_options"]["save_intermediate_steps"] = True
    sa_on = ScintillationAnalysis(cfg_on)
    sa_on.prepare_data()
    assert sa_on.masked_spectrum.num_channels == 512
    assert sa_on.masked_spectrum.channel_width_mhz == pytest.approx(native, rel=1e-9)
