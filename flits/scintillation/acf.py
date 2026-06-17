"""Autocorrelation-function estimation for scintillation analysis.

Migrated from the legacy `scint_pipeline_funcs` module: 1D ACF of a single
series and a 2D ACF that averages per-slice ACFs along one axis of a dynamic
spectrum.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, correlation_lags

__all__ = ["calculate_acf_1d", "calculate_acf_2d"]


def calculate_acf_1d(
    series: NDArray[np.floating],
    norm: bool = True,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    """Estimate the 1D autocorrelation function of a series.

    NaNs are dropped before the mean is removed and the ACF is computed.

    Parameters
    ----------
    series : ndarray
        1D input series (e.g. a single frequency channel or time sample).
    norm : bool, optional
        Normalize the ACF to unity at zero lag. Default True.

    Returns
    -------
    lags : ndarray
        Integer lags (mode ``"same"``), centred on zero.
    acf : ndarray
        Autocorrelation values. All-NaN of length ``len(series)`` if fewer
        than two finite samples are available.

    Notes
    -----
    Normalization divides by the central (zero-lag) value of the *computed* ACF
    rather than by an index into the original (NaN-containing) series; this
    fixes a latent off-by-NaN bug in the legacy code while leaving the
    finite-input result unchanged.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.size
    valid = series[~np.isnan(series)]
    if valid.size < 2:
        lags = np.arange(-n // 2 + 1, n // 2 + 1)
        return lags, np.full(n, np.nan)

    detrended = valid - valid.mean()
    acf = correlate(detrended, detrended, mode="same")
    lags = correlation_lags(detrended.size, detrended.size, mode="same")

    if norm:
        acf = acf / acf[detrended.size // 2]

    return lags, acf


def calculate_acf_2d(
    dyn_spec: NDArray[np.floating],
    axis: int = 1,
    norm: bool = True,
) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
    """Average 1D ACFs over the slices of a dynamic spectrum.

    For ``axis == 1`` (frequency) the ACF is computed for each time slice and
    the per-slice ACFs are averaged; for ``axis == 0`` (time) the roles swap.

    Parameters
    ----------
    dyn_spec : ndarray
        2D dynamic spectrum with shape ``(ntime, nfreq)``.
    axis : int, optional
        Axis along which to autocorrelate: 0 for time, 1 for frequency.
        Default 1.
    norm : bool, optional
        Normalize each per-slice ACF before averaging. Default True.

    Returns
    -------
    lags : ndarray
        Centred integer lags along ``axis``.
    avg_acf : ndarray
        NaN-mean of the per-slice ACFs.

    Raises
    ------
    ValueError
        If ``dyn_spec`` is not 2D.
    """
    dyn_spec = np.asarray(dyn_spec, dtype=np.float64)
    if dyn_spec.ndim != 2:
        raise ValueError("Input dynamic spectrum must be 2D.")

    n_slices = dyn_spec.shape[1 - axis]
    acf_len = dyn_spec.shape[axis]
    target_lags = np.arange(-acf_len // 2 + 1, acf_len // 2 + 1)
    all_acfs: list[NDArray[np.floating]] = []

    for i in range(n_slices):
        series = dyn_spec[i, :] if axis == 1 else dyn_spec[:, i]
        lags, acf = calculate_acf_1d(series, norm=norm)

        if len(acf) == len(target_lags):
            all_acfs.append(acf)
        else:
            # Centre-align a short ACF (e.g. when NaNs reduced its length) onto
            # the common lag grid, padding the rest with NaN.
            aligned = np.full(len(target_lags), np.nan)
            start_target = max(0, len(target_lags) // 2 - len(lags) // 2)
            start_acf = max(0, len(lags) // 2 - len(target_lags) // 2)
            overlap = min(len(lags) - start_acf, len(target_lags) - start_target)
            aligned[start_target:start_target + overlap] = (
                acf[start_acf:start_acf + overlap]
            )
            all_acfs.append(aligned)

    if not all_acfs:
        return target_lags, np.full(acf_len, np.nan)

    avg_acf = np.nanmean(np.array(all_acfs), axis=0)
    return target_lags, avg_acf
