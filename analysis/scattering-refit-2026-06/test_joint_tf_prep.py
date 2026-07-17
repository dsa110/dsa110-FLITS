"""Unit tests for joint_tf_prep: the robust window + S/N-driven resolution rules.

Synthetic dynamic spectra only (no data files), so these run fast and pin the two
behaviors the manuscript review demanded: (1) the window is stable against isolated
off-pulse spikes and captures a scattering tail, and (2) the resolution rule
returns the finest binning that clears the S/N floor and coarsens for faint bursts.
"""
import os
import sys

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
