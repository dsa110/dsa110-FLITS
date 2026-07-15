"""Independent-stream frequency cross-ACFs for CHIME scintillation tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CrossACF:
    """Block-averaged symmetric cross-covariance and its block standard error."""

    lag_bins: np.ndarray
    acf: np.ndarray
    error: np.ndarray
    n_blocks: np.ndarray
    covariance: np.ndarray
    block_acfs: np.ndarray | None = None


def _demean_by_block(
    values: np.ndarray, block_codes: np.ndarray, n_unique_blocks: int
) -> np.ndarray:
    finite = np.isfinite(values)
    counts = np.bincount(block_codes[finite], minlength=n_unique_blocks)
    sums = np.bincount(block_codes[finite], weights=values[finite], minlength=n_unique_blocks)
    means = np.divide(
        sums,
        counts,
        out=np.full(n_unique_blocks, np.nan, dtype=float),
        where=counts > 0,
    )
    return values - means[block_codes]


def _lag_block_stats(
    blocks: np.ndarray,
    block_codes: np.ndarray,
    n_unique_blocks: int,
    per_lag,
    *,
    max_lag_bins: int,
    min_blocks: int,
) -> CrossACF:
    """Block statistics over lags; ``per_lag(lag) -> (products, valid)``."""
    values = []
    errors = []
    counts = []
    block_acfs = np.full((n_unique_blocks, max_lag_bins), np.nan, dtype=float)
    for lag in range(1, max_lag_bins + 1):
        products, valid = per_lag(lag)
        codes = block_codes[:-lag][valid]
        block_counts = np.bincount(codes, minlength=n_unique_blocks)
        block_sums = np.bincount(codes, weights=products[valid], minlength=n_unique_blocks)
        supported = block_counts >= 2
        block_means = block_sums[supported] / block_counts[supported]
        block_acfs[supported, lag - 1] = block_means
        counts.append(int(block_means.size))
        if block_means.size < min_blocks:
            values.append(np.nan)
            errors.append(np.nan)
        else:
            values.append(float(np.mean(block_means)))
            errors.append(float(np.std(block_means, ddof=1) / np.sqrt(block_means.size)))
    complete_blocks = np.all(np.isfinite(block_acfs), axis=1)
    if np.count_nonzero(complete_blocks) >= min_blocks:
        covariance = np.cov(block_acfs[complete_blocks], rowvar=False, ddof=1) / np.count_nonzero(
            complete_blocks
        )
    else:
        covariance = np.diag(np.square(errors))
    return CrossACF(
        lag_bins=np.arange(1, max_lag_bins + 1, dtype=int),
        acf=np.asarray(values),
        error=np.asarray(errors),
        n_blocks=np.asarray(counts, dtype=int),
        covariance=np.asarray(covariance),
        block_acfs=block_acfs,
    )


def blockwise_cross_acf_pairs(
    pairs: Sequence[tuple[np.ndarray, np.ndarray, float, float]],
    block_ids: np.ndarray,
    *,
    max_lag_bins: int,
    min_blocks: int = 8,
) -> CrossACF:
    """Estimate a symmetric cross-ACF averaged over stream pairs.

    Each pair is ``(left, right, normalization_left, normalization_right)``.
    Symmetrized products from every pair are averaged at fixed channel
    position and lag before block statistics, so passing the time-disjoint
    pairs ``(X_even, Y_odd)`` and ``(X_odd, Y_even)`` removes any term
    correlated at equal times -- polarized source self-noise, common
    burst-time RFI -- from the expectation while the common spectral signal
    is kept.  Products are formed only within a parent coarse channel; ACF
    values are first averaged within each coarse block, then averaged across
    blocks, and the reported uncertainty is the standard error across blocks.
    """
    blocks = np.asarray(block_ids)
    if max_lag_bins < 1 or min_blocks < 2:
        raise ValueError("max_lag_bins must be positive and min_blocks must be at least 2")
    if not pairs:
        raise ValueError("at least one stream pair is required")
    _, block_codes = np.unique(blocks, return_inverse=True)
    n_unique_blocks = int(block_codes.max(initial=-1) + 1)

    # The estimator never crosses a parent coarse-channel boundary, so remove
    # the corresponding block offset rather than a single full-band mean.  The
    # narrowest targets are far below a 64-channel block, and real-background
    # injections quantify the high-pass transfer, which grows with the ratio
    # of scintle width to block size (relevant for the widest injected cells).
    centered = []
    for left, right, normalization_left, normalization_right in pairs:
        x = np.asarray(left, dtype=float)
        y = np.asarray(right, dtype=float)
        if x.ndim != 1 or y.shape != x.shape or blocks.shape != x.shape:
            raise ValueError("left, right, and block_ids must be matching 1-D arrays")
        denominator = float(normalization_left) * float(normalization_right)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("normalizations must have a positive finite product")
        if np.isfinite(x).sum() < 2 or np.isfinite(y).sum() < 2:
            raise ValueError("each input must contain at least two finite samples")
        centered.append(
            (
                _demean_by_block(x, block_codes, n_unique_blocks),
                _demean_by_block(y, block_codes, n_unique_blocks),
                denominator,
            )
        )

    def per_lag(lag: int) -> tuple[np.ndarray, np.ndarray]:
        valid = blocks[:-lag] == blocks[lag:]
        products = np.zeros(blocks.size - lag, dtype=float)
        for x, y, denominator in centered:
            valid &= (
                np.isfinite(x[:-lag])
                & np.isfinite(x[lag:])
                & np.isfinite(y[:-lag])
                & np.isfinite(y[lag:])
            )
            with np.errstate(invalid="ignore"):
                products = products + 0.5 * (x[:-lag] * y[lag:] + y[:-lag] * x[lag:]) / denominator
        return products / len(centered), valid

    return _lag_block_stats(
        blocks,
        block_codes,
        n_unique_blocks,
        per_lag,
        max_lag_bins=max_lag_bins,
        min_blocks=min_blocks,
    )


def blockwise_cross_acf(
    left: np.ndarray,
    right: np.ndarray,
    block_ids: np.ndarray,
    *,
    normalization_left: float,
    normalization_right: float,
    max_lag_bins: int,
    min_blocks: int = 8,
) -> CrossACF:
    """Estimate a symmetric cross-ACF from two independent noise streams.

    Single-pair form of :func:`blockwise_cross_acf_pairs`.  Receiver noise
    unique to either input has zero expectation, while frequency structure
    shared by both inputs -- including anything correlated at equal times --
    remains.
    """
    return blockwise_cross_acf_pairs(
        [(left, right, normalization_left, normalization_right)],
        block_ids,
        max_lag_bins=max_lag_bins,
        min_blocks=min_blocks,
    )


def _grouped_jackknife_errors(
    cross_acf: CrossACF,
    indices: np.ndarray,
    model,
    best: np.ndarray,
    factor: np.ndarray,
    lower: bool,
    *,
    bounds,
    n_groups: int = 16,
) -> np.ndarray | None:
    """Delete-one-group jackknife parameter errors from per-block ACFs.

    Groups are contiguous runs of parent coarse blocks, so each deletion
    removes a band segment much wider than any fitted scintle; the spread of
    the refitted parameters measures realization variance that the
    independent-block formal errors miss.  Refits reuse the full-fit
    whitening and start from the full-fit solution.  Returns None (caller
    keeps formal errors) when too few complete blocks support the jackknife.
    """
    if cross_acf.block_acfs is None:
        return None
    block_acfs = np.asarray(cross_acf.block_acfs, dtype=float)[:, indices]
    complete = np.all(np.isfinite(block_acfs), axis=1)
    block_acfs = block_acfs[complete]
    if block_acfs.shape[0] < 2 * n_groups:
        return None
    boundaries = np.linspace(0, block_acfs.shape[0], n_groups + 1).astype(int)
    estimates = []
    for group in range(n_groups):
        keep = np.ones(block_acfs.shape[0], dtype=bool)
        keep[boundaries[group] : boundaries[group + 1]] = False
        acf_group = block_acfs[keep].mean(axis=0)

        def residual(parameters: np.ndarray, acf_group: np.ndarray = acf_group) -> np.ndarray:
            return solve_triangular(
                factor,
                acf_group - model(parameters),
                lower=lower,
                check_finite=False,
            )

        refit = least_squares(residual, x0=best, bounds=bounds, max_nfev=2000)
        if not (refit.success and np.all(np.isfinite(refit.x))):
            return None
        estimates.append(refit.x)
    values = np.asarray(estimates)
    mean = values.mean(axis=0)
    return np.sqrt((len(values) - 1) / len(values) * np.sum((values - mean) ** 2, axis=0))


def _block_demeaned_model_acf(
    acf_full: np.ndarray, block_length: int, lag_bins: np.ndarray
) -> np.ndarray:
    """Expected within-block ACF of a block-demeaned stationary field.

    Removing each block's mean turns a true ACF C into
    C'(d) = C(d) - avg_i[mu_i + mu_(i+d)] + mu_bar with
    mu_i = (1/L) sum_k C(|i-k|): the high-pass that biases wide scintles
    low if the raw model is fitted instead.  Exact for complete blocks;
    masked channels make it approximate, which the injection gate measures.
    """
    length = int(block_length)
    lags = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    gram = acf_full[lags]
    mu = gram.mean(axis=1)
    mu_bar = float(mu.mean())
    transformed = np.empty(lag_bins.size)
    for index, lag in enumerate(lag_bins):
        positions = np.arange(length - int(lag))
        transformed[index] = (
            acf_full[int(lag)] - float(np.mean(mu[positions] + mu[positions + int(lag)])) + mu_bar
        )
    return transformed


def fit_cross_lorentzian(
    cross_acf: CrossACF,
    *,
    channel_width_mhz: float,
    first_lag_bin: int = 2,
    fit_max_mhz: float = 0.25,
    block_length: int | None = None,
) -> dict | None:
    """Fit a positive Lorentzian common-signal component to a cross-ACF.

    When ``block_length`` is given, the model is the block-demeaned
    expectation of the Lorentzian (matching the estimator's per-block mean
    removal) instead of the raw Lorentzian.
    """
    if channel_width_mhz <= 0 or fit_max_mhz <= 0:
        raise ValueError("channel_width_mhz and fit_max_mhz must be positive")
    lags = np.asarray(cross_acf.lag_bins, dtype=float) * float(channel_width_mhz)
    acf = np.asarray(cross_acf.acf, dtype=float)
    error = np.asarray(cross_acf.error, dtype=float)
    keep = (
        np.isfinite(lags)
        & np.isfinite(acf)
        & np.isfinite(error)
        & (error > 0)
        & (cross_acf.lag_bins >= int(first_lag_bin))
        & (lags <= float(fit_max_mhz))
    )
    indices = np.flatnonzero(keep)
    if indices.size < 12:
        return None
    x = lags[indices]
    z = acf[indices]
    covariance = np.asarray(cross_acf.covariance, dtype=float)[np.ix_(indices, indices)]
    scale = float(np.nanmedian(np.diag(covariance)))
    floor = max(scale * 1e-6, 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    factor, lower = cho_factor(covariance, lower=True, check_finite=False)

    fit_lag_bins = np.asarray(cross_acf.lag_bins, dtype=int)[indices]

    def model(parameters: np.ndarray) -> np.ndarray:
        width, amplitude, constant = parameters
        if block_length is None:
            return amplitude / (1.0 + (x / width) ** 2) + constant
        acf_full = amplitude / (
            1.0 + (np.arange(int(block_length)) * float(channel_width_mhz) / width) ** 2
        )
        return _block_demeaned_model_acf(acf_full, block_length, fit_lag_bins) + constant

    def residual(parameters: np.ndarray) -> np.ndarray:
        return solve_triangular(
            factor,
            z - model(parameters),
            lower=lower,
            check_finite=False,
        )

    starts = float(channel_width_mhz) * np.asarray([3.0, 4.0, 5.0, 6.0, 10.0, 20.0])
    candidates = []
    amplitude_start = max(float(np.nanmax(z) - np.nanmedian(z[-5:])), 1e-4)
    for width_start in starts:
        fit = least_squares(
            residual,
            x0=(min(width_start, 0.8 * fit_max_mhz), amplitude_start, 0.0),
            bounds=(
                (0.5 * channel_width_mhz, 0.0, -5.0),
                (fit_max_mhz, 9.0, 5.0),
            ),
            max_nfev=5000,
        )
        if fit.success and np.all(np.isfinite(fit.x)):
            candidates.append(fit)
    if not candidates:
        return None
    fit = min(candidates, key=lambda candidate: float(np.sum(candidate.fun**2)))
    dof = indices.size - fit.x.size
    if dof <= 0:
        return None
    redchi = float(np.sum(fit.fun**2) / dof)
    try:
        covariance = np.linalg.inv(fit.jac.T @ fit.jac)
        # Residual excess over the whitened expectation (redchi > 1) marks
        # per-realization scatter the block covariance does not carry.
        # Scaling up (never down) is the conservative direction: measured
        # over-coverage on synthetic wide/high-m cells (0.86-0.89 vs the
        # [0.53, 0.83] gate) fails closed, while omitting the factor
        # under-covers (0.33-0.53) and would overstate real confidence.
        # Open adjudication: see the B3 checkpoint report.
        parameter_error = np.sqrt(np.diag(covariance)) * np.sqrt(max(1.0, redchi))
    except np.linalg.LinAlgError:
        parameter_error = np.full(3, np.nan)
    # Formal GLS errors treat blocks as independent, but a single
    # scintillation realization correlates neighboring blocks, so they
    # under-cover for scintles approaching the 64-channel block scale.  A
    # deterministic delete-one-group jackknife over contiguous block groups
    # captures that realization variance; take the larger of the two error
    # estimates per parameter and let the injection coverage gate validate.
    jackknife = _grouped_jackknife_errors(
        cross_acf,
        indices,
        model,
        fit.x,
        factor,
        lower,
        bounds=((0.5 * channel_width_mhz, 0.0, -5.0), (fit_max_mhz, 9.0, 5.0)),
    )
    if jackknife is not None:
        parameter_error = np.where(
            np.isfinite(jackknife), np.maximum(parameter_error, jackknife), parameter_error
        )
    amplitude = float(fit.x[1])
    modulation = float(np.sqrt(amplitude))
    modulation_error = (
        float(0.5 * parameter_error[1] / modulation)
        if modulation > 0 and np.isfinite(parameter_error[1])
        else np.nan
    )
    return {
        "dnu_mhz": float(fit.x[0]),
        "dnu_err_mhz": float(parameter_error[0]),
        "m": modulation,
        "m_err": modulation_error,
        "amplitude": amplitude,
        "constant": float(fit.x[2]),
        "redchi": redchi,
        "n_fit_points": int(indices.size),
        "fit_lags_mhz": x.tolist(),
        "observed_acf": z.tolist(),
        "model_acf": model(fit.x).tolist(),
    }


def all_pairs_cross_acf(
    polarized_dynamic: Sequence[np.ndarray],
    block_ids: np.ndarray,
    *,
    max_lag_bins: int,
    min_blocks: int = 8,
    exclude_same_time: bool = True,
    normalizations: Sequence[np.ndarray] | None = None,
) -> CrossACF:
    """Build an all-pairs cross-ACF from independent polarization/time folds.

    Each input array is a 2-D polarization waterfall ``(n_channels, n_times)``
    whose time samples are preserved as individual folds.  Every admissible
    cross-product between distinct folds is formed, excluding auto-products and
    (by default) equal-time products that retain polarized source self-noise.
    The returned ``CrossACF`` is the block-averaged ACF across the folded pair
    ensemble, suitable for ``fit_cross_lorentzian``.

    ``normalizations`` is an optional per-fold mean intensity (e.g. the burst
    envelope) used as the denominator for each pair.  When omitted, the mean of
    each supplied spectrum is used.  Providing a fixed normalization lets
    off-pulse nulls and on-pulse pairs share the same amplitude scale.
    """
    if not polarized_dynamic:
        raise ValueError("at least one polarization waterfall is required")
    n_times = int(polarized_dynamic[0].shape[1])
    for idx, arr in enumerate(polarized_dynamic):
        if arr.ndim != 2 or arr.shape[1] != n_times:
            raise ValueError(
                f"polarized_dynamic[{idx}] must have shape (n_channels, {n_times})"
            )
        if arr.shape[0] != block_ids.size:
            raise ValueError("polarized_dynamic channel count must match block_ids")
    if normalizations is not None:
        if len(normalizations) != len(polarized_dynamic):
            raise ValueError("normalizations must match polarized_dynamic")
        for idx, arr in enumerate(normalizations):
            if arr.shape != (n_times,):
                raise ValueError(
                    f"normalizations[{idx}] must have shape ({n_times},)"
                )

    blocks = np.asarray(block_ids)
    if max_lag_bins < 1 or min_blocks < 2:
        raise ValueError("max_lag_bins must be positive and min_blocks must be at least 2")
    _, block_codes = np.unique(blocks, return_inverse=True)
    n_unique_blocks = int(block_codes.max(initial=-1) + 1)

    # Keep folds by the same admissibility rules the per-pair enumeration
    # used: at least two finite samples and a positive finite normalization.
    kept_scaled = []
    kept_times = []
    for pol in range(len(polarized_dynamic)):
        for time in range(n_times):
            spec = np.asarray(polarized_dynamic[pol][:, time], dtype=float)
            if np.isfinite(spec).sum() < 2:
                continue
            if normalizations is not None:
                norm = float(normalizations[pol][time])
            else:
                norm = float(np.nanmean(spec))
            if not np.isfinite(norm) or norm <= 0:
                continue
            kept_scaled.append(_demean_by_block(spec, block_codes, n_unique_blocks) / norm)
            kept_times.append(time)

    n_kept = len(kept_scaled)
    time_counts = np.bincount(np.asarray(kept_times, dtype=int), minlength=n_times)
    n_pairs = n_kept * (n_kept - 1) // 2
    if exclude_same_time:
        n_pairs -= int(np.sum(time_counts * (time_counts - 1) // 2))
    if n_pairs <= 0:
        raise ValueError("no admissible cross-product pairs found")

    # Sum over all admissible symmetrized pairs in closed form: with scaled
    # folds v_i = demeaned_i / norm_i, T = sum_i v_i and W_t = sum of folds at
    # time t,  sum_{a<b admissible} (v_a v_b)_sym = 0.5 (T*T - sum_t W_t*W_t)
    # when same-time pairs are excluded (per-fold auto terms cancel between
    # the two subtractions), and 0.5 (T*T - sum_i v_i*v_i) otherwise.  This is
    # O(n_folds) per lag instead of O(n_pairs) and identical in exact
    # arithmetic; the pinned characterization test bounds the float
    # summation-order difference.
    fold_matrix = np.vstack(kept_scaled)
    finite = np.isfinite(fold_matrix)
    total = fold_matrix.sum(axis=0)
    if exclude_same_time:
        group = np.zeros((n_times, blocks.size), dtype=float)
        for row, time in enumerate(kept_times):
            group[time] += fold_matrix[row]
    else:
        group = fold_matrix

    def per_lag(lag: int) -> tuple[np.ndarray, np.ndarray]:
        valid = blocks[:-lag] == blocks[lag:]
        valid &= np.all(finite[:, :-lag] & finite[:, lag:], axis=0)
        with np.errstate(invalid="ignore"):
            pair_sum = 0.5 * (
                total[:-lag] * total[lag:]
                - np.einsum("ij,ij->j", group[:, :-lag], group[:, lag:])
            )
        return pair_sum / n_pairs, valid

    return _lag_block_stats(
        blocks,
        block_codes,
        n_unique_blocks,
        per_lag,
        max_lag_bins=max_lag_bins,
        min_blocks=min_blocks,
    )
