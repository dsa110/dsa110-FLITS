"""Tests for independent-stream frequency cross-ACFs."""

from __future__ import annotations

import numpy as np

from scintillation.scint_analysis.cross_acf import (
    all_pairs_cross_acf,
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


def _c1_low_modulation_inputs():
    """Seeded C1 synthetic: 2 pols x 12 times, m=0.15, width 6 native bins."""
    rng = np.random.default_rng(20260717)
    nblocks = 256
    block_size = 64
    n = nblocks * block_size
    n_times = 12
    truth_width_bins = 6.0
    truth_m = 0.15

    common = truth_m * _stationary_lorentzian(rng, n, truth_width_bins)
    pols = []
    for _ in range(2):
        times = []
        for _ in range(n_times):
            times.append(1.0 + common + 0.7 * rng.normal(size=n))
        pols.append(np.column_stack(times))
    blocks = np.repeat(np.arange(nblocks), block_size)
    return pols, blocks, block_size, truth_width_bins, truth_m


def test_all_pairs_cross_acf_recovers_low_modulation():
    """C1 uses all distinct-time pairs, not four collapsed spectra."""
    channel_width = 0.006103515625
    pols, blocks, block_size, truth_width_bins, truth_m = _c1_low_modulation_inputs()

    result = all_pairs_cross_acf(pols, blocks, max_lag_bins=40)
    fit = fit_cross_lorentzian(
        result,
        channel_width_mhz=channel_width,
        first_lag_bin=2,
        block_length=block_size,
    )

    assert fit is not None
    assert np.isclose(fit["dnu_mhz"], truth_width_bins * channel_width, rtol=0.25)
    assert np.isclose(fit["m"], truth_m, rtol=0.25)


def test_all_pairs_cross_acf_matches_pinned_reference():
    # Characterization pin for the vectorized rewrite: every CrossACF field
    # must match the per-pair implementation's output on the seeded C1
    # synthetic (fixture generated at the pre-vectorization commit; only
    # float summation order may differ, hence the tight tolerances).
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "all_pairs_reference_20260717.npz"
    pols, blocks, _, _, _ = _c1_low_modulation_inputs()

    result = all_pairs_cross_acf(pols, blocks, max_lag_bins=40)
    ref = np.load(fixture)
    np.testing.assert_array_equal(result.lag_bins, ref["lag_bins"])
    np.testing.assert_allclose(result.acf, ref["acf"], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(result.error, ref["error"], rtol=1e-10, atol=1e-13)
    np.testing.assert_array_equal(result.n_blocks, ref["n_blocks"])
    np.testing.assert_allclose(result.covariance, ref["covariance"], rtol=1e-9, atol=1e-16)
    np.testing.assert_allclose(result.block_acfs, ref["block_acfs"], rtol=1e-10, atol=1e-13)


def test_all_pairs_cross_acf_excludes_same_time_by_default():
    """Same-time products should not enter the default pair list."""
    rng = np.random.default_rng(20260718)
    nblocks = 64
    block_size = 64
    n = nblocks * block_size
    n_times = 3
    common = 0.5 * _stationary_lorentzian(rng, n, 4.0)
    pols = []
    for _ in range(2):
        times = []
        for _ in range(n_times):
            shared_noise = 0.9 * _stationary_lorentzian(rng, n, 2.0)
            times.append(1.0 + common + shared_noise + 0.3 * rng.normal(size=n))
        pols.append(np.column_stack(times))
    blocks = np.repeat(np.arange(nblocks), block_size)

    result = all_pairs_cross_acf(pols, blocks, max_lag_bins=20)
    # With only three time samples and same-time excluded, 2*3*2 = 12 pairs
    # remain; the shared equal-time noise must not bias the first lag.
    assert result.acf[0] < 0.3


def test_all_pairs_cross_acf_rejects_mismatched_inputs():
    blocks = np.repeat(np.arange(8), 8)
    with np.testing.assert_raises_regex(ValueError, "match"):
        all_pairs_cross_acf(
            [np.ones((64, 2)), np.ones((63, 2))],
            blocks,
            max_lag_bins=3,
        )
