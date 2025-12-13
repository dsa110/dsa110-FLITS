"""Scattering and broadening utilities for FRB modeling.

This module provides:
- Kernel construction and convolution for scatter broadening (Gaussian ⊗ exponential).
- Physical priors on scattering parameters.
- Per-frequency scattering timescale (τ(ν)) with power-law frequency scaling.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve


def scatter_broaden(
    signal: NDArray[np.floating],
    t: NDArray[np.floating],
    tau_ms: float | NDArray[np.floating],
    *,
    causal: bool = True,
) -> NDArray[np.floating]:
    """Apply scattering (exponential) broadening via convolution.

    Convolves the input signal with a causal exponential kernel:
        kernel(t) = (1/τ) * exp(-t/τ) * H(t)
    where H(t) is the Heaviside step function.

    Parameters
    ----------
    signal : ndarray
        Input signal (1D or 2D with shape (nfreq, ntime)). If 2D, broadening
        is applied independently to each row (frequency).
    t : ndarray
        Time axis in milliseconds (1D, length ntime).
    tau_ms : float or ndarray
        Scattering timescale(s) in milliseconds.
        - If scalar: applied uniformly.
        - If ndarray of length nfreq: per-frequency timescale (requires signal.ndim==2).
    causal : bool
        If True (default), kernel is causal (t >= 0). If False, symmetric.

    Returns
    -------
    ndarray
        Broadened signal, same shape as input.

    Notes
    -----
    The kernel is normalized by integrating to preserve total flux. Convolution
    is scaled by dt to maintain physical units.
    """
    if len(t) < 2:
        raise ValueError("Time axis must have at least 2 samples.")

    dt = float(t[1] - t[0])
    signal = np.asarray(signal, dtype=np.float64)
    tau_ms = np.atleast_1d(np.asarray(tau_ms, dtype=np.float64))

    # Validate dimensions
    if signal.ndim == 1:
        if tau_ms.size > 1:
            raise ValueError(
                "Per-frequency tau requires 2D signal; got 1D with tau.size={}.".format(
                    tau_ms.size
                )
            )
        tau_ms = tau_ms[0]
        is_2d = False
    elif signal.ndim == 2:
        nfreq, ntime = signal.shape
        if tau_ms.size == 1:
            tau_ms = np.full(nfreq, tau_ms[0])
        elif tau_ms.size != nfreq:
            raise ValueError(
                "tau_ms size {} must match signal.shape[0]={}".format(
                    tau_ms.size, nfreq
                )
            )
        is_2d = True
    else:
        raise ValueError("signal must be 1D or 2D, got shape {}.".format(signal.shape))

    # Build kernel(s)
    if causal:
        t_kernel = np.maximum(t - t.min(), 0.0)
    else:
        t_center = (t.min() + t.max()) / 2.0
        t_kernel = np.abs(t - t_center)

    # Apply convolution
    if is_2d:
        result = np.zeros_like(signal)
        for i, tau in enumerate(tau_ms):
            if tau <= 0.0:
                result[i, :] = signal[i, :]
            else:
                kernel = np.exp(-t_kernel / tau)
                kernel /= kernel.sum() if kernel.sum() > 0 else 1.0
                result[i, :] = fftconvolve(
                    signal[i, :], kernel, mode="same"
                ) * dt
    else:
        if tau_ms <= 0.0:
            result = signal.copy()
        else:
            kernel = np.exp(-t_kernel / tau_ms)
            kernel /= kernel.sum() if kernel.sum() > 0 else 1.0
            result = fftconvolve(signal, kernel, mode="same") * dt

    return result


def tau_per_freq(
    tau_ref_ms: float,
    freqs_mhz: NDArray[np.floating],
    alpha: float,
    ref_freq_mhz: float = 1000.0,
) -> NDArray[np.floating]:
    """Compute per-frequency scattering timescale via power-law scaling.

    Parameters
    ----------
    tau_ref_ms : float
        Reference scattering timescale (at ref_freq_mhz) in milliseconds.
    freqs_mhz : ndarray
        Frequencies in MHz.
    alpha : float
        Power-law exponent: τ(ν) = τ_ref * (ν_ref / ν)^α.
        Typical: α ≈ 4.0 (thin screen) to 4.4 (Kolmogorov).
    ref_freq_mhz : float
        Reference frequency in MHz (default 1000 = 1 GHz).

    Returns
    -------
    ndarray
        Per-frequency timescales in milliseconds.
    """
    freqs_mhz = np.asarray(freqs_mhz, dtype=np.float64)
    return tau_ref_ms * (ref_freq_mhz / freqs_mhz) ** alpha


def log_normal_prior(x: float, mu: float, sigma: float) -> float:
    """Log-probability for log-normal distribution (for τ_ms or similar).

    Parameters
    ----------
    x : float
        Value (must be > 0).
    mu : float
        Mean of log(x).
    sigma : float
        Std dev of log(x).

    Returns
    -------
    float
        Log-probability (unnormalized).
    """
    if x <= 0.0:
        return -np.inf
    return -0.5 * ((np.log(x) - mu) / sigma) ** 2 - np.log(sigma * x)


def gaussian_prior(x: float, mu: float, sigma: float) -> float:
    """Log-probability for Gaussian (normal) distribution.

    Parameters
    ----------
    x : float
        Value.
    mu : float
        Mean.
    sigma : float
        Std dev.

    Returns
    -------
    float
        Log-probability (unnormalized).
    """
    if sigma <= 0.0:
        return 0.0
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))


__all__ = [
    "scatter_broaden",
    "tau_per_freq",
    "log_normal_prior",
    "gaussian_prior",
]
