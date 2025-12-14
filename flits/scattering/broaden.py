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
        kernel(t') = (1/τ) × exp(-t'/τ) for t' ≥ 0
    This is the standard pulse-broadening function for thin-screen scattering.

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
        If True (default), kernel is causal (t ≥ 0). If False, symmetric (not physical).

    Returns
    -------
    ndarray
        Broadened signal, same shape as input.

    Notes
    -----
    The kernel is normalized to unit integral to preserve total flux.
    """
    if len(t) < 2:
        raise ValueError("Time axis must have at least 2 samples.")

    dt = float(t[1] - t[0])
    ntime = len(t)
    signal = np.asarray(signal, dtype=np.float64)
    tau_arr = np.atleast_1d(np.asarray(tau_ms, dtype=np.float64))

    # Validate dimensions and extract scalar tau if 1D
    tau_scalar: float = 0.0  # default, will be set if 1D
    if signal.ndim == 1:
        if tau_arr.size > 1:
            raise ValueError(
                f"Per-frequency tau requires 2D signal; got 1D with tau.size={tau_arr.size}."
            )
        tau_scalar = float(tau_arr[0])
        is_2d = False
    elif signal.ndim == 2:
        nfreq, _ = signal.shape
        if tau_arr.size == 1:
            tau_arr = np.full(nfreq, tau_arr[0])
        elif tau_arr.size != nfreq:
            raise ValueError(
                f"tau_ms size {tau_arr.size} must match signal.shape[0]={nfreq}"
            )
        is_2d = True
    else:
        raise ValueError(f"signal must be 1D or 2D, got shape {signal.shape}.")

    def _build_kernel(tau: float) -> NDArray[np.floating]:
        """Build a properly-sized causal exponential kernel."""
        # Kernel length: ~10τ is sufficient to capture 99.995% of the exponential
        # But cap at signal length to avoid huge kernels
        kernel_len = min(int(np.ceil(10 * tau / dt)), ntime)
        kernel_len = max(kernel_len, 3)  # at least 3 samples

        t_kernel = np.arange(kernel_len) * dt  # starts at 0
        if causal:
            kernel = np.exp(-t_kernel / tau)
        else:
            # Symmetric (not physically meaningful, but supported)
            t_sym = np.abs(t_kernel - t_kernel[kernel_len // 2])
            kernel = np.exp(-t_sym / tau)

        # Normalize to unit integral (sum × dt = 1)
        kernel /= kernel.sum() * dt
        return kernel

    # Apply convolution
    if is_2d:
        result = np.zeros_like(signal)
        for i, tau in enumerate(tau_arr):
            if tau <= 0.0:
                result[i, :] = signal[i, :]
            else:
                kernel = _build_kernel(tau)
                # mode="full" then trim to preserve causality and timing
                conv = fftconvolve(signal[i, :], kernel, mode="full")
                # For causal kernel, output is shifted; take first ntime samples
                result[i, :] = conv[:ntime] * dt
    else:
        if tau_scalar <= 0.0:
            result = signal.copy()
        else:
            kernel = _build_kernel(tau_scalar)
            conv = fftconvolve(signal, kernel, mode="full")
            result = conv[:ntime] * dt

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
