"""Tests for the Route-B common-mode-immune Δν_d statistics.

Two invariants matter for correctness:

* **Bandpass cancellation** — the on/off ratio ``R_p = ⟨I⟩_on/⟨I⟩_off − 1``
  divides out any time-stable multiplicative gain ``g(ν)``, so S1 and S2 must
  be invariant to machine precision when the whole frame is multiplied by an
  arbitrary positive ``g(ν)``.  This is the algebraic property that lets Route
  B evade the common-mode instrumental response every earlier route retained.
* **Blinding guard** — no statistic may read the on-pulse window (samples
  250–350) unless the caller explicitly asserts ``allow_unblind``.

See docs/rse/specs/experiment-chime-scint-routeb-voltage.md (Faber2026).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_test_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

from scint_analysis import routeb_voltage as rb  # noqa: E402


def _synthetic_frames(seed: int = 0, n_channels: int = 2048, n_times: int = 200):
    rng = np.random.default_rng(seed)
    block_ids = np.arange(n_channels) // rb_block()
    dyn = [rng.normal(5.0, 1.0, size=(n_channels, n_times)) for _ in range(2)]
    return dyn, block_ids


def rb_block() -> int:
    return 64


@pytest.mark.parametrize("statistic", ["s1", "s2"])
def test_bandpass_cancellation_machine_precision(statistic):
    """Multiplying every channel by an arbitrary positive g(ν) leaves the
    cross-ACF unchanged to machine precision (the ratio cancels g exactly)."""
    dyn, block_ids = _synthetic_frames(seed=1)
    n_channels = dyn[0].shape[0]
    on = np.arange(0, 100)
    off = np.arange(100, 200)
    rng = np.random.default_rng(7)
    gain = np.exp(rng.normal(0.0, 0.8, size=n_channels)) + 0.05  # arbitrary, positive

    func = {"s1": rb.s1_ratio_cross_acf, "s2": rb.s2_time_split_cross_acf}[statistic]
    base = func(dyn, on, off, block_ids, channel_width_mhz=0.0061036)
    scaled = func(
        [d * gain[:, None] for d in dyn], on, off, block_ids, channel_width_mhz=0.0061036
    )
    delta = np.nanmax(np.abs(base.cross.acf - scaled.cross.acf))
    assert delta < 1e-12, f"{statistic} not invariant to g(ν): max|dACF|={delta:e}"


def test_bandpass_cancellation_scales_with_frequency():
    """A strongly frequency-structured gain (orders-of-magnitude variation)
    still cancels — the invariance is not a small-perturbation accident."""
    dyn, block_ids = _synthetic_frames(seed=2)
    n_channels = dyn[0].shape[0]
    on, off = np.arange(0, 100), np.arange(100, 200)
    freq = np.linspace(0, 1, n_channels)
    gain = 10.0 ** (2.0 * np.sin(20 * freq))  # spans ~1e-2 .. 1e2
    base = rb.s1_ratio_cross_acf(dyn, on, off, block_ids, channel_width_mhz=0.0061036)
    scaled = rb.s1_ratio_cross_acf(
        [d * gain[:, None] for d in dyn], on, off, block_ids, channel_width_mhz=0.0061036
    )
    assert np.nanmax(np.abs(base.cross.acf - scaled.cross.acf)) < 1e-12


def test_blinding_guard_raises_on_onpulse_window():
    for samples in (rb.samples_from_window((250, 350)), np.arange(240, 260), np.array([349])):
        with pytest.raises(rb.BlindingError):
            rb.assert_offpulse_samples(samples)


def test_blinding_guard_allows_offpulse_window():
    # Fully off-pulse windows pass; the on-pulse boundary is half-open [250,350).
    rb.assert_offpulse_samples(np.arange(0, 200))
    rb.assert_offpulse_samples(np.arange(350, 437))  # 350 is the first allowed sample
    rb.assert_offpulse_samples(np.arange(240, 250))  # 249 is the last allowed sample


def test_blinding_guard_bypassed_with_flag():
    # allow_unblind is the only way past the guard; it returns the indices.
    out = rb.assert_offpulse_samples(np.arange(250, 350), allow_unblind=True)
    assert out.min() == 250 and out.max() == 349


@pytest.mark.parametrize("func", [rb.s1_ratio_cross_acf, rb.s2_time_split_cross_acf])
def test_statistics_refuse_onpulse_without_flag(func):
    dyn, block_ids = _synthetic_frames(seed=3, n_times=400)
    on = np.arange(250, 300)  # inside the blinded window
    off = np.arange(0, 100)
    with pytest.raises(rb.BlindingError):
        func(dyn, on, off, block_ids, channel_width_mhz=0.0061036)
    # with the flag it proceeds (returns a result, possibly with a null fit)
    result = func(dyn, on, off, block_ids, channel_width_mhz=0.0061036, allow_unblind=True)
    assert result.cross.acf.size > 0


def test_lorentzian_field_hwhm_convention():
    """width_channels is the HWHM: the field ACF halves at that lag, and a
    high-SNR injection recovers the injected width through the fit."""
    rng = np.random.default_rng(11)
    n = 8192
    width = 30.0
    field = rb.lorentzian_gain_field(rng, n_channels=n, width_channels=width)
    assert abs(field.mean()) < 0.05 and abs(field.std() - 1.0) < 0.05
    # empirical circular ACF at the HWHM lag should be ~0.5
    fft = np.fft.fft(field)
    acf = np.real(np.fft.ifft(np.abs(fft) ** 2)) / n
    acf /= acf[0]
    assert 0.35 < acf[int(round(width))] < 0.65

    # end-to-end: strong injection -> recovered width within 15%
    dyn = [rng.normal(5.0, 0.02, size=(n, 200)) for _ in range(2)]
    block_ids = np.arange(n) // 64
    gain = 1.0 + 0.3 * field
    result = rb.s1_ratio_cross_acf(
        dyn, np.arange(0, 100), np.arange(100, 200), block_ids,
        channel_width_mhz=0.0061036, on_gain=gain, fit_max_mhz=0.5,
    )
    assert result.fit is not None
    recovered = result.fit["dnu_mhz"] / 0.0061036  # channels
    assert abs(recovered / width - 1.0) < 0.15


def test_voltage_intensity_is_squared_magnitude():
    v = np.array([[3 + 4j, 0], [1, 2j]])
    np.testing.assert_allclose(rb.voltage_intensity(v), np.array([[25.0, 0.0], [1.0, 4.0]]))


def test_s3_matches_s1_on_voltage_intensity():
    """S3 on complex voltages equals S1 on |V|² (same ratio construction)."""
    rng = np.random.default_rng(5)
    n = 2048
    volts = [rng.normal(size=(n, 200)) + 1j * rng.normal(size=(n, 200)) for _ in range(2)]
    intensity = [rb.voltage_intensity(v) for v in volts]
    block_ids = np.arange(n) // 64
    on, off = np.arange(0, 100), np.arange(100, 200)
    s3 = rb.s3_voltage_cross_acf(volts, on, off, block_ids, channel_width_mhz=0.0061036)
    s1 = rb.s1_ratio_cross_acf(intensity, on, off, block_ids, channel_width_mhz=0.0061036)
    np.testing.assert_allclose(s3.cross.acf, s1.cross.acf, rtol=0, atol=1e-15)
