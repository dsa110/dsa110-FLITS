"""Tests for the P3′ delay-domain matched Δν_d estimator.

The correctness invariants mirror Route B's, plus the matched-scan algebra:

* **Bandpass cancellation (T1)** — the split-ratio fields divide out any
  time-stable multiplicative ``g(ν)``, so the matched amplitudes must be
  invariant to machine precision under an arbitrary positive ``g(ν)`` —
  including a 35 kHz Lorentzian-shaped realization of the measured common
  mode.
* **Blinding guard (T6)** — field construction refuses the on-pulse window
  (samples 250–350) without ``allow_unblind``.
* **Matched-scan algebra** — the estimator recovers an injected amplitude on
  synthetic data (mini-T2), respects the frozen ``k < KMIN`` exclusion, and
  the template bank is seed-deterministic.

See docs/rse/specs/experiment-chime-scint-p3-optimal-estimator.md (§P3′
amendment) in Faber2026.
"""

from __future__ import annotations

import numpy as np
import pytest

from scint_analysis import optimal_dnu as od
from scint_analysis import routeb_voltage as rb


def _synthetic_frames(seed: int = 0, n_channels: int = 2048, n_times: int = 437):
    rng = np.random.default_rng(seed)
    return [
        1.0 + 0.05 * rng.standard_normal((n_channels, n_times)) for _ in range(2)
    ], rng


def _off_windows():
    on = np.arange(10, 110)
    off = np.arange(120, 240)
    return on, off


def test_t1_bandpass_invariance_machine_precision():
    dynamic, rng = _synthetic_frames()
    on, off = _off_windows()
    n = dynamic[0].shape[0]
    # arbitrary smooth gain plus a 35 kHz-scale Lorentzian-ACF realization of
    # the measured common mode (width 35.4/6.1036 ~ 5.8 fine channels)
    gain = 1.5 + 0.5 * np.sin(np.linspace(0, 9, n)) ** 2
    gain *= 1.0 + 0.586 * rb.lorentzian_gain_field(rng, n_channels=n, width_channels=5.8)
    gain = np.abs(gain) + 0.1
    base = od.split_ratio_fields(dynamic, on, off)
    scaled = od.split_ratio_fields([d * gain[:, None] for d in dynamic], on, off)
    for a, b in zip(base, scaled):
        assert np.nanmax(np.abs(a - b)) < 1e-12


def test_t6_blinding_guard():
    dynamic, _ = _synthetic_frames(n_times=437)
    with pytest.raises(rb.BlindingError):
        od.split_ratio_fields(dynamic, np.arange(240, 340), np.arange(0, 200))
    fields = od.split_ratio_fields(
        dynamic, np.arange(240, 340), np.arange(0, 200), allow_unblind=True
    )
    assert np.isfinite(fields[0]).all()


def test_delay_transform_removes_dc_and_matches_length():
    x = np.full(256, 3.7)
    spectrum = od.delay_transform(x)
    assert spectrum.size == 128
    assert np.allclose(np.abs(spectrum), 0.0)


def test_template_seed_determinism():
    mask = np.ones(512, dtype=bool)
    mask[100:110] = False
    kwargs = dict(
        n_channels=512, channel_width_khz=6.1, good_mask=mask, n_realizations=8
    )
    t1 = od.lorentzian_template(100.0, 3, **kwargs)
    t2 = od.lorentzian_template(100.0, 3, **kwargs)
    t3 = od.lorentzian_template(100.0, 4, **kwargs)
    assert np.array_equal(t1, t2)
    assert not np.array_equal(t1, t3)


def _mini_scan(n_channels: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    mask = np.ones(n_channels, dtype=bool)
    dnu = np.array([50.0, 150.0])
    templates = np.vstack(
        [
            od.lorentzian_template(
                d, gi, n_channels=n_channels, channel_width_khz=6.1,
                good_mask=mask, n_realizations=32,
            )
            for gi, d in enumerate(dnu)
        ]
    )
    return rng, mask, dnu, templates


def test_kmin_exclusion_changes_denominator():
    _, _, dnu, templates = _mini_scan(1024)
    variance = np.ones(templates.shape[1])
    wide = od.MatchedScan(dnu, templates, variance, kmin=1)
    cut = od.MatchedScan(dnu, templates, variance, kmin=11)
    # removing bins can only lose information: sigma grows
    assert (cut.sigma_analytic > wide.sigma_analytic).all()


def test_mini_t2_amplitude_recovery_on_synthetic():
    n = 2048
    rng, mask, dnu, templates = _mini_scan(n)
    width_channels = 150.0 / 6.1
    a_true = 4e-3
    nulls, signals = [], []
    for j in range(200):
        noise1 = 0.02 * rng.standard_normal(n)
        noise2 = 0.02 * rng.standard_normal(n)
        delta = rb.lorentzian_gain_field(rng, n_channels=n, width_channels=width_channels)
        s = np.sqrt(a_true) * delta
        nulls.append(od.cross_power(noise1, noise2))
        signals.append(od.cross_power(noise1 + s, noise2 + s))
    nulls, signals = np.asarray(nulls), np.asarray(signals)
    scan = od.MatchedScan(dnu, templates, od.smooth_variance(nulls.var(axis=0, ddof=1)))
    scan.calibrate(nulls[:100])
    grid = scan.nearest_grid_index(150.0)
    recovered = np.array([scan.amplitudes(p)[grid] for p in signals])
    stderr = recovered.std(ddof=1) / np.sqrt(recovered.size)
    assert abs(recovered.mean() - a_true) < 4 * stderr
    # T3-style sigma calibration on the null half not used for calibrate()
    a_eval = np.array([scan.amplitudes(p)[grid] for p in nulls[100:]])
    ratio = a_eval.std(ddof=1) / scan.null_sigma[grid]
    assert 0.7 < ratio < 1.3


def test_zscan_requires_calibration():
    _, _, dnu, templates = _mini_scan(1024)
    scan = od.MatchedScan(dnu, templates, np.ones(templates.shape[1]))
    with pytest.raises(RuntimeError):
        scan.zscan(np.zeros(templates.shape[1]))
