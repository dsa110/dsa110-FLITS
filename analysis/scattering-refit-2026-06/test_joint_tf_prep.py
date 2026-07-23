"""Unit tests for joint_tf_prep: the robust window + S/N-driven resolution rules.

Synthetic dynamic spectra only (no data files), so these run fast and pin the two
behaviors the manuscript review demanded: (1) the window is stable against isolated
off-pulse spikes and captures a scattering tail, and (2) the resolution rule
returns the finest binning that clears the S/N floor and coarsens for faint bursts.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scattering"))
sys.path.insert(0, HERE)

import joint_tf_prep as J  # noqa: E402


def _scattered_profile(n=4000, peak=1200, amp=40.0, tau=60.0, rng=None):
    """Gaussian core + exponential scattering tail on a noisy baseline."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(n)
    core = amp * np.exp(-0.5 * ((t - peak) / 8.0) ** 2)
    tail = amp * 0.6 * np.exp(-(t - peak) / tau) * (t >= peak)
    return core + tail + rng.normal(0, 1.0, n)


def test_window_captures_tail_not_spike():
    prof = _scattered_profile(tau=80.0)  # burst core amp ~40 at index 1200
    # inject a lone 6-sigma off-pulse excursion far from the burst -- above the
    # legacy 3-sigma on-pulse threshold (which took global min/max and ran away)
    # but well below the burst, so peak-anchoring must ignore it.
    prof[50] += 6.0
    lo, hi = J.robust_onpulse_bounds(prof, dt_ms=0.03)
    assert lo > 200, f"window opened onto the off-pulse spike (lo={lo})"
    # the tail (peak 1200, decays over ~80 samples) must be inside
    assert hi > 1200 + 80, f"window clipped the scattering tail (hi={hi})"
    # and it must not run away to the whole array
    assert hi - lo < 0.6 * prof.size


def test_window_stable_to_sharp_burst():
    # a clean sharp burst must not collapse to a handful of samples
    rng = np.random.default_rng(1)
    prof = 30.0 * np.exp(-0.5 * ((np.arange(2000) - 1000) / 3.0) ** 2)
    prof += rng.normal(0, 1.0, 2000)
    lo, hi = J.robust_onpulse_bounds(prof, dt_ms=0.03)
    assert hi - lo >= 20, f"sharp-burst window collapsed ({hi - lo} samples)"


def test_native_chime_noise_does_not_chain_tail_to_cap():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        t = np.arange(16000)
        prof = 35.0 * np.exp(-0.5 * ((t - 6000) / 5.0) ** 2)
        prof += rng.normal(0, 1.0, t.size)
        _lo, hi = J.robust_onpulse_bounds(prof, dt_ms=0.00256)
        cap = int(round(J.WIN_TRAIL_CAP_MS / 0.00256))
        assert hi < 6000 + cap // 4, f"noise chained toward the tail cap (seed={seed}, hi={hi})"


def test_window_does_not_follow_a_low_significance_leading_shelf():
    rng = np.random.default_rng(5)
    t = np.arange(4000)
    prof = rng.normal(0, 1.0, t.size)
    prof += 40.0 * np.exp(-0.5 * ((t - 2000) / 8.0) ** 2)
    prof += 24.0 * np.exp(-(t - 2000) / 80.0) * (t >= 2000)
    clean_bounds = J.robust_onpulse_bounds(prof, dt_ms=0.03)
    prof[1800:1950] += 2.0
    lo, hi = J.robust_onpulse_bounds(prof, dt_ms=0.03)
    assert (lo, hi) == clean_bounds, "leading shelf changed the high-threshold core window"
    assert hi > 2080, f"trailing edge clipped the scattering tail (hi={hi})"


def test_resolution_finest_that_clears_floor():
    # bright, temporally-resolved burst -> should keep fine time bins (small t)
    rng = np.random.default_rng(2)
    nf, nt = 256, 4000
    data = rng.normal(0, 1.0, (nf, nt))
    prof = _scattered_profile(n=nt, amp=60.0, tau=50.0, rng=rng)
    data += (prof / prof.max() * 30.0)[None, :]  # bright signal on every channel
    win = J.robust_onpulse_bounds(np.nansum(data, 0), dt_ms=0.03)
    f, t = J.choose_resolution(data, win, nf, snr_target=10.0)
    assert t >= 1 and (t & (t - 1)) == 0, "t_factor must be a power of two"
    # bright burst -> the window stays under the tractability cap
    assert (win[1] - win[0]) // t <= J.MAX_TIME_BINS


def test_resolution_coarsens_for_faint():
    # Same fixed window + geometry for both, so ONLY brightness drives the time
    # choice (window width otherwise couples into the tractability cap). Few
    # channels so the band-integrated profile S/N tracks per-channel brightness.
    rng = np.random.default_rng(3)
    nf, nt = 16, 4000
    t = np.arange(nt)  # clean (noiseless) burst SHAPE: Gaussian core + scattering tail
    shape = np.exp(-0.5 * ((t - 1200) / 8.0) ** 2) + 0.6 * np.exp(-(t - 1200) / 50.0) * (t >= 1200)
    sig = shape[None, :]
    bright = rng.normal(0, 1.0, (nf, nt)) + sig * 60.0
    faint = rng.normal(0, 1.0, (nf, nt)) + sig * 1.0  # clearly needs coarsening
    win = (1000, 1600)  # identical window for both -> isolates the S/N floor
    _, t_bright = J.choose_resolution(bright, win, nf, snr_target=15.0)
    _, t_faint = J.choose_resolution(faint, win, nf, snr_target=15.0)
    assert t_faint > t_bright, (
        f"faint burst should bin coarser in time (bright t={t_bright}, faint t={t_faint})"
    )


def test_resolution_retries_coarser_time_when_channels_fail(monkeypatch):
    data = np.zeros((128, 4096))
    win = (1000, 1200)

    def qualification_by_time(data_ds, _win_ds, _target):
        # At native time resolution the integrated profile passes but every
        # channel fails. One factor coarser makes both statistics pass.
        if data_ds.shape[1] == 4096:
            return 20.0, 2.0, False
        return 20.0, 12.0, True

    monkeypatch.setattr(J, "resolution_snr_status", qualification_by_time)
    f, t = J.choose_resolution(data, win, data.shape[0], snr_target=10.0)
    assert (f, t) == (2, 2)


def test_pair_reselects_resolution_against_final_common_windows(monkeypatch):
    native = np.zeros((16, 64))
    probe_c = J._Probe(native, 1.0, object(), 20, (10, 20), 1, 1, 5.0)
    probe_d = J._Probe(native, 1.0, object(), 30, (25, 35), 1, 1, 5.0)
    probes = iter((probe_c, probe_d))
    monkeypatch.setattr(J, "_probe_band", lambda *_args, **_kwargs: next(probes))
    monkeypatch.setattr(J, "_common_peak_relative_window", lambda _probes: [(5, 25), (20, 40)])

    seen_windows = []

    def choose(_native, window, _nchan, **_kwargs):
        seen_windows.append(window)
        return (2, 4) if window == (5, 25) else (4, 8)

    monkeypatch.setattr(J, "choose_resolution", choose)
    monkeypatch.setattr(J, "_build_model", lambda probe, window: ((probe.f_factor, probe.t_factor), window))

    result_c, result_d = J.prepare_pair("c.yaml", "d.yaml", "burst", "out", snr_target=5.0)
    assert seen_windows == [(5, 25), (20, 40)]
    assert result_c == ((2, 4), (5, 25))
    assert result_d == ((4, 8), (20, 40))


def test_degenerate_final_window_cannot_become_qualified_after_full_trace_fallback():
    rng = np.random.default_rng(44)
    native = rng.normal(0.0, 1.0, (16, 512))
    native[:, 250] += 400.0
    final_window = (250, 251)
    f_factor, t_factor = J.choose_resolution(
        native, final_window, native.shape[0], snr_target=10.0
    )
    probe = J._Probe(
        native=native,
        dt_native=1.0,
        tel=SimpleNamespace(f_min_GHz=1.0, f_max_GHz=2.0, df_MHz_raw=1.0),
        peak=250,
        win=final_window,
        f_factor=f_factor,
        t_factor=t_factor,
        snr_target=10.0,
    )
    _, meta = J._build_model(probe, final_window)
    assert meta.peak_profile_snr > 10.0
    assert meta.median_channel_snr > 10.0
    assert meta.snr_qualified is False


def test_unreachable_snr_floor_is_explicitly_unqualified():
    rng = np.random.default_rng(31)
    data = rng.normal(0, 1.0, (16, 512))
    profile_snr, channel_snr, qualified = J.resolution_snr_status(
        data, (200, 260), snr_target=1000.0
    )
    assert np.isfinite(profile_snr)
    assert np.isfinite(channel_snr)
    assert qualified is False


def test_time_bin_cap_respected():
    # a very wide window at native resolution must be capped under MAX_TIME_BINS
    rng = np.random.default_rng(4)
    nf, nt = 64, 20000
    data = rng.normal(0, 1.0, (nf, nt))
    data[:, 9000:11000] += 50.0  # broad bright plateau
    win = J.robust_onpulse_bounds(np.nansum(data, 0), dt_ms=0.0026)
    f, t = J.choose_resolution(data, win, nf, snr_target=10.0)
    assert (win[1] - win[0]) // t <= J.MAX_TIME_BINS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
