"""
burstfit_robust.py
==================

Diagnostic helpers to check the robustness of a scattering fit.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

# ## REFACTOR ##: Corrected the relative import to work with the pipeline structure.
from .burstfit import (
    FRBModel,
    FRBFitter,
    FRBParams,
    build_priors,
    DM_SMEAR_MS,
)
from flits.common.constants import DM_DELAY_MS

__all__ = [
    "subband_consistency",
    "leave_one_out_influence",
    "plot_influence",
    "dm_optimization_check",
    "fit_subband_profiles",
    "plot_subband_profiles",
]

# -----------------------------------------------------------------------------
# Sub-band consistency test
# -----------------------------------------------------------------------------


def subband_consistency(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm_init: float,
    df_MHz: float,
    init: FRBParams,
    *,
    model_key: str = "M3",
    n_sub: int = 4,
    n_steps: int = 500,
    pool=None,
    walker_width_frac: float = 0.02,
) -> Tuple[str | None, List[Tuple[float, float]], List[NDArray]]:
    """
    Re-fit the *same* model in `n_sub` frequency slices and return
    (parameter_name, [(mean, std), ...], [chain, ...]).
    """
    if n_sub < 2:
        raise ValueError("Need at least two sub-bands")

    if model_key in ("M2", "M3"):
        par_name = "tau_1ghz"
    elif model_key == "M1":
        par_name = "zeta"
    else:
        warnings.warn(f"Model {model_key} has no broadening param for subband check.")
        return None, [(np.nan, np.nan)] * n_sub, []

    edges = np.linspace(0, freq.size, n_sub + 1, dtype=int)
    results: List[Tuple[float, float]] = []
    # ## FIX ##: Initialized all_chains list before the loop.
    all_chains: List[NDArray] = []

    for i in range(n_sub):
        sl = slice(edges[i], edges[i + 1])
        sub_freq = freq[sl]
        sub_data = data[sl]

        # Skip sub-bands with too few channels
        if len(sub_freq) < 3:
            warnings.warn(
                f"Sub-band {i} has too few channels ({len(sub_freq)}), skipping"
            )
            results.append((np.nan, np.nan))
            all_chains.append(np.array([]))
            continue

        # Effective channel width for this sub-band can be different if non-uniform
        sub_df_MHz = np.mean(np.diff(sub_freq)) * 1000 if len(sub_freq) > 1 else df_MHz

        try:
            model = FRBModel(
                time=time,
                freq=sub_freq,
                data=sub_data,
                dm_init=dm_init,
                df_MHz=sub_df_MHz,
            )
            priors = build_priors(init, scale=3.0)

            # The FRBFitter needs priors for ALL possible params, it will slice internally
            full_priors, use_logw = build_priors(init, scale=3.0, log_weight_pos=True)

            fitter = FRBFitter(
                model,
                full_priors,
                n_steps=n_steps,
                pool=pool,
                log_weight_pos=use_logw,
                walker_width_frac=max(walker_width_frac, 0.1),
            )  # Ensure wider spread

            sampler = fitter.sample(init, model_key)

            burn = n_steps // 4
            chain = sampler.get_chain(discard=burn, thin=4, flat=True)
            all_chains.append(chain)

            order = FRBFitter._ORDER[model_key]
            par_idx = order.index(par_name)

            if chain.shape[0] > 0:
                par_vals = chain[:, par_idx]
                results.append((float(np.mean(par_vals)), float(np.std(par_vals))))
            else:
                results.append((np.nan, np.nan))

        except ValueError as e:
            # Handle "large condition number" and other initialization errors
            warnings.warn(f"Sub-band {i} MCMC failed: {e}")
            results.append((np.nan, np.nan))
            all_chains.append(np.array([]))
        except Exception as e:
            warnings.warn(f"Sub-band {i} unexpected error: {e}")
            results.append((np.nan, np.nan))
            all_chains.append(np.array([]))

    return par_name, results, all_chains


# -----------------------------------------------------------------------------
# Leave-one-out influence diagnostic
# -----------------------------------------------------------------------------


def leave_one_out_influence(
    data: NDArray[np.floating],
    model_dyn: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    ## FIX ##: Corrected function signature. It only needs data and model_dyn.
    The `freq` argument was unused and has been removed.

    Calculates Δχ² influence per channel for a given model.
    """
    resid_sq = (data - model_dyn) ** 2
    chi2_per_channel = np.sum(resid_sq, axis=1)
    total_chi2 = np.sum(chi2_per_channel)
    # Influence = (total_chi2) - (total_chi2 - chi2_per_channel)
    # This simplifies to just chi2_per_channel. A more standard definition is used below.
    # Influence = chi_sq_full - chi_sq_without_channel_i
    return chi2_per_channel


def plot_influence(ax, delta_chi2: NDArray[np.floating], freq: NDArray[np.floating]):
    sigma = np.nanstd(delta_chi2)
    # Use frequency channel index for x-axis if freq is non-uniform
    x_axis = freq
    width = np.mean(np.diff(freq)) if len(freq) > 1 else 1.0

    ax.bar(x_axis, delta_chi2, width=width, align="center", color="gray")
    ax.axhline(3 * sigma, ls="--", color="m", lw=1.5)
    ax.axhline(-3 * sigma, ls="--", color="m", lw=1.5)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Influence (Δχ²)")
    ax.set_title("Leave-One-Out Channel Influence")
    ax.margins(x=0.02)


# -----------------------------------------------------------------------------
# DM Optimization Check
# -----------------------------------------------------------------------------


def dm_optimization_check(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm_init: float,
    dm_range: float = 5.0,
    n_trials: int = 41,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Check if the assumed DM is optimal by computing S/N vs trial DM offset."""
    dm_offsets = np.linspace(-dm_range, dm_range, n_trials)
    snrs = np.zeros(n_trials)
    dt = time[1] - time[0]
    ref_freq = freq.max()

    for i, dm_offset in enumerate(dm_offsets):
        delays_ms = DM_DELAY_MS * dm_offset * (freq**-2 - ref_freq**-2)
        shifts = np.round(delays_ms / dt).astype(int)

        # Create a dedispersed time series
        profile = np.zeros_like(time)
        for j, shift in enumerate(shifts):
            profile += np.roll(data[j], -shift)

        noise_rms = np.std(profile[: time.size // 4])
        signal_peak = np.max(profile)
        snrs[i] = signal_peak / noise_rms if noise_rms > 0 else 0

    return dm_offsets, snrs


# ---------------------------------------------------------------------
# 1-D sub-band profile consistency
# ---------------------------------------------------------------------


def _pulse_model_1d(t, amp, mu, tau, sigma):
    """Convolution of a Gaussian with a causal exponential tail."""
    gauss = amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    # Ensure kernel is defined on the same time grid length
    t_kernel = t - t.min()
    tail = np.exp(-t_kernel / np.clip(tau, 1e-6, None))
    tail[t_kernel < 0] = 0.0  # ensure causality

    # Convolve and scale by time resolution
    dt = t[1] - t[0]
    convolved = fftconvolve(gauss, tail, mode="same")
    return convolved * dt / np.sum(tail)  # normalize convolution kernel


def fit_subband_profiles(
    dataset: Any, best_params: FRBParams, dm_init: float, n_sub: int = 4
):
    """Quick 1-D thin-screen fits in `n_sub` equal-width frequency slices."""
    freq = dataset.freq
    data = dataset.data
    time = dataset.time
    n_ch = freq.size
    step = n_ch // n_sub if n_sub > 0 else n_ch

    # Intrinsic (non-smearing) pulse width from 2D fit
    sigma_intr = getattr(best_params, "zeta", 0.0)

    centres, tau_hat, tau_err = [], [], []

    for k in range(n_sub):
        start = k * step
        stop = (k + 1) * step if k < n_sub - 1 else n_ch
        if start >= stop:
            continue

        sl = slice(start, stop)
        profile = np.nansum(data[sl, :], axis=0)
        c_freq = freq[sl].mean()
        df_mhz_sub = (
            (freq[sl][-1] - freq[sl][0]) * 1000 if len(freq[sl]) > 1 else dataset.df_MHz
        )

        # Total Gaussian width for this sub-band
        sigma_dm = DM_SMEAR_MS * dm_init * df_mhz_sub * (c_freq**-3.0)
        sigma_k = np.hypot(sigma_intr, sigma_dm)

        centres.append(c_freq)

        # Initial guess for curve_fit
        amp0 = np.max(profile)
        mu0 = time[np.argmax(profile)]
        tau0_global = getattr(best_params, "tau_1ghz", 1.0)
        tau0 = tau0_global * (c_freq / 1.0) ** -4.0
        p0 = (amp0, mu0, tau0)

        def model_to_fit(t, amp, mu, tau):
            return _pulse_model_1d(t, amp, mu, tau, sigma=sigma_k)

        try:
            bounds = ([0, time.min(), 0], [np.inf, time.max(), np.inf])
            popt, pcov = curve_fit(
                model_to_fit, time, profile, p0=p0, bounds=bounds, maxfev=5000
            )
            tau_hat.append(popt[2])
            tau_err.append(np.sqrt(np.diag(pcov))[2])
        except (RuntimeError, ValueError):
            tau_hat.append(np.nan)
            tau_err.append(np.nan)

    return np.asarray(centres), np.asarray(tau_hat), np.asarray(tau_err)


def plot_subband_profiles(ax, centres, tau_hat, tau_err, best_params, fontsize=10):
    """Scatter plot + ν⁻⁴ law for the 1-D τ values."""
    ok = np.isfinite(tau_hat) & np.isfinite(tau_err) & (tau_err > 0)
    if not ok.any():
        ax.text(
            0.5,
            0.5,
            "1D Profile Fits Failed",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    ax.errorbar(
        centres[ok],
        tau_hat[ok],
        yerr=tau_err[ok],
        fmt="o",
        ms=6,
        c="k",
        capsize=3,
        label="1-D Slice Fit",
    )

    if hasattr(best_params, "tau_1ghz"):
        tau1ghz_2d = getattr(best_params, "tau_1ghz", 0.0)
        if tau1ghz_2d > 0:
            nu_grid = np.linspace(centres.min() * 0.95, centres.max() * 1.05, 100)
            ax.plot(
                nu_grid,
                tau1ghz_2d * (nu_grid / 1.0) ** -4.0,
                lw=2,
                color="m",
                label=r"Global 2D Fit (τ∝ν$^{-4}$)",
            )

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("τ [ms]")
    ax.set_title("1-D Sub-band Profile Fit")
    ax.legend(loc="best", fontsize=fontsize)
    ax.set_yscale("log")
    ax.margins(x=0.05)
