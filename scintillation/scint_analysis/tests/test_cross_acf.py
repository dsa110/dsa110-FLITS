"""Tests for independent-stream frequency cross-ACFs."""

from __future__ import annotations

import numpy as np

from scintillation.scint_analysis.cross_acf import (
    blockwise_cross_acf,
    blockwise_cross_acf_pairs,
    fit_cross_lorentzian,
)


def _stationary_lorentzian(rng: np.random.Generator, n: int, width_bins: float) -> np.ndarray:
    distances = np.minimum(np.arange(n), n - np.arange(n))
    covariance = 1.0 / (1.0 + (distances / width_bins) ** 2)
    power = np.maximum(np.real(np.fft.fft(covariance)), 0.0)
    sample = np.real(np.fft.ifft(np.fft.fft(rng.normal(size=n)) * np.sqrt(power)))
    return (sample - sample.mean()) / sample.std()


def test_independent_noise_cross_acf_is_consistent_with_zero():
    rng = np.random.default_rng(20260714)
    nblocks = 256
    block_size = 64
    n = nblocks * block_size
    result = blockwise_cross_acf(
        rng.normal(size=n),
        rng.normal(size=n),
        np.repeat(np.arange(nblocks), block_size),
        normalization_left=1.0,
        normalization_right=1.0,
        max_lag_bins=24,
    )

    assert np.nanmax(np.abs(result.acf / result.error)) < 3.0


def test_common_lorentzian_survives_independent_receiver_noise():
    rng = np.random.default_rng(74291)
    nblocks = 512
    block_size = 64
    n = nblocks * block_size
    truth_width_bins = 4.0
    common = 0.8 * _stationary_lorentzian(rng, n, truth_width_bins)
    left = common + rng.normal(scale=0.7, size=n)
    right = common + rng.normal(scale=0.7, size=n)
    channel_width = 0.006103515625
    result = blockwise_cross_acf(
        left,
        right,
        np.repeat(np.arange(nblocks), block_size),
        normalization_left=1.0,
        normalization_right=1.0,
        max_lag_bins=40,
    )
    fit = fit_cross_lorentzian(result, channel_width_mhz=channel_width)

    assert fit is not None
    assert np.isclose(fit["dnu_mhz"], truth_width_bins * channel_width, rtol=0.15)
    assert np.isclose(fit["m"], 0.8, rtol=0.15)


def test_cross_acf_rejects_nonmatching_inputs():
    with np.testing.assert_raises_regex(ValueError, "matching"):
        blockwise_cross_acf(
            np.ones(10),
            np.ones(9),
            np.arange(10),
            normalization_left=1.0,
            normalization_right=1.0,
            max_lag_bins=3,
        )


def test_kernel_correlated_receiver_noise_biases_auto_but_not_cross():
    # The upchannelization kernel correlates neighboring fine channels within
    # each stream.  The bias this puts into an autocorrelation must vanish in
    # the cross of two independent streams carrying the same kernel.
    rng = np.random.default_rng(20260715)
    nblocks = 384
    block_size = 64
    n = nblocks * block_size
    blocks = np.repeat(np.arange(nblocks), block_size)
    kernel_width_bins = 2.0
    left = _stationary_lorentzian(rng, n, kernel_width_bins)
    right = _stationary_lorentzian(rng, n, kernel_width_bins)

    cross = blockwise_cross_acf(
        left,
        right,
        blocks,
        normalization_left=1.0,
        normalization_right=1.0,
        max_lag_bins=24,
    )
    auto = blockwise_cross_acf(
        left,
        left,
        blocks,
        normalization_left=1.0,
        normalization_right=1.0,
        max_lag_bins=24,
    )

    assert np.nanmax(np.abs(cross.acf / cross.error)) < 3.0
    assert auto.acf[0] / auto.error[0] > 10.0


def test_time_disjoint_pairs_remove_equal_time_common_noise():
    # Polarized source self-noise is correlated between the polarizations at
    # equal times, so it contaminates the plain X x Y cross-ACF.  The
    # symmetrized time-disjoint pairing shares no time samples between its
    # factors and must recover the common signal cleanly.
    rng = np.random.default_rng(20260716)
    nblocks = 384
    block_size = 64
    n = nblocks * block_size
    blocks = np.repeat(np.arange(nblocks), block_size)
    truth_width_bins = 6.0
    truth_m = 0.7
    channel_width = 0.006103515625

    common_signal = truth_m * _stationary_lorentzian(rng, n, truth_width_bins)
    shared_even = 0.9 * _stationary_lorentzian(rng, n, 2.0)
    shared_odd = 0.9 * _stationary_lorentzian(rng, n, 2.0)
    x_even = common_signal + shared_even + 0.3 * rng.normal(size=n)
    y_even = common_signal + shared_even + 0.3 * rng.normal(size=n)
    x_odd = common_signal + shared_odd + 0.3 * rng.normal(size=n)
    y_odd = common_signal + shared_odd + 0.3 * rng.normal(size=n)

    contaminated = blockwise_cross_acf(
        0.5 * (x_even + x_odd),
        0.5 * (y_even + y_odd),
        blocks,
        normalization_left=1.0,
        normalization_right=1.0,
        max_lag_bins=40,
    )
    disjoint = blockwise_cross_acf_pairs(
        [(x_even, y_odd, 1.0, 1.0), (x_odd, y_even, 1.0, 1.0)],
        blocks,
        max_lag_bins=40,
    )

    # The equal-time estimator keeps the shared-noise ACF at low lags; the
    # disjoint estimator must not.
    assert contaminated.acf[0] - disjoint.acf[0] > 0.2

    fit = fit_cross_lorentzian(disjoint, channel_width_mhz=channel_width)
    assert fit is not None
    assert np.isclose(fit["dnu_mhz"], truth_width_bins * channel_width, rtol=0.15)
    assert np.isclose(fit["m"], truth_m, rtol=0.15)
