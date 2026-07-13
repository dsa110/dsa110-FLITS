"""
burstfit_subband.py
One–dimensional pulse‐profile fits in several frequency sub-bands.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

__all__ = ["_pulse_model", "fit_subband_profiles", "plot_subband_profiles"]


# ------------------------------------------------------------------
# analytic 1-D thin–screen model:  G(t; σ) * E(t; τ)
# ------------------------------------------------------------------
def _pulse_model(t, amp, mu, tau, sigma):
    """
    Convolution of a Gaussian (amp, mu, sigma) with an exponential tail (tau).
    Returns the model evaluated on the supplied time grid.
    """
    gauss = amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    tail  = np.exp(-(t - t.min()) / tau)           # causal exponential
    tail[t < t.min()] = 0.0
    dt = t[1] - t[0]
    return fftconvolve(gauss, tail, mode="same") * dt


# ------------------------------------------------------------------
def fit_subband_profiles(dataset, best_params, n_sub=4, p0=None):
    """
    Fit a 1-D thin-screen profile to 'n_sub' equal-width frequency slices.

    Parameters
    ----------
    dataset      : BurstDataset   (already down-sampled)
    best_params  : FRBParams      (full-band best fit)
    n_sub        : int            (number of slices, default 4)
    p0           : tuple|None     (amp, mu, tau) starting guess

    Returns
    -------
    centres  : 1-D array, sub-band centre frequencies [GHz]
    tau_hat  : 1-D array, fitted tau_1GHz per sub-band   [ms]
    tau_err  : 1-D array, 1σ uncertainty from curve_fit
    """
    freq     = dataset.freq              # shape (n_chan,)
    data     = dataset.data              # shape (n_chan, n_time)
    time     = dataset.time              # shape (n_time,)
    n_chan   = freq.size
    step     = n_chan // n_sub
    sigma_fix = np.sqrt(best_params.zeta ** 2 + best_params.dm_smw ** 2)

    centres, tau_hat, tau_err = [], [], []

    for k in range(n_sub):
        sl       = slice(k * step, (k + 1) * step)
        profile  = np.nansum(data[sl, :], axis=0)   # collapse to 1-D
        c_freq   = np.mean(freq[sl])
        centres.append(c_freq)

        # initial guess: amplitude=max, mu=argmax, tau = best global
        if p0 is None:
            amp0 = profile.max()
            mu0  = time[np.argmax(profile)]
            tau0 = best_params.tau_1GHz * (c_freq / 1.0) ** -4
            p0   = (amp0, mu0, tau0)

        def _model(t, amp, mu, tau):
            return _pulse_model(t, amp, mu, tau, sigma_fix)

        popt, pcov = curve_fit(_model, time, profile, p0=p0, maxfev=5000)
        amp, mu, tau = popt
        tau_sigma    = np.sqrt(pcov[2, 2])

        tau_hat.append(tau)
        tau_err.append(tau_sigma)

    return np.asarray(centres), np.asarray(tau_hat), np.asarray(tau_err)


# ------------------------------------------------------------------
def plot_subband_profiles(ax, centres, tau_hat, tau_err, best_params):
    """
    Plot τ per sub-band and the global ν⁻⁴ law on an existing axes.

    Parameters
    ----------
    ax          : matplotlib.axes.Axes
    centres     : array, sub-band centre frequencies [GHz]
    tau_hat     : array, fitted tau values            [ms]
    tau_err     : array, ±1σ uncertainties
    best_params : FRBParams (for global τ₁GHz)
    """
    # scatter with error bars
    #ax.errorbar(centres, tau_hat, yerr=tau_err, fmt="o", ms=6, c="k",
    #            capsize=3, label="1-D fit per sub-band")

    # global thin-screen curve
    nu_grid = np.linspace(centres.min()*0.95, centres.max()*1.05, 200)
    tau_nu  = best_params.tau_1GHz * (nu_grid / 1.0) ** -4
    ax.plot(nu_grid, tau_nu, lw=1.5, color="m", label="global τ(ν) ∝ ν⁻⁴")
    ax.set_ylim(-3*np.abs(tau_err), 3*np.abs(tau_err))
    ax.set_xlabel(r"Frequency $\nu$ [GHz]")
    ax.set_ylabel(r"$\tau$ [ms]")
    ax.set_title("1-D Sub-Band Consistency")
    ax.legend(loc="best")
    ax.set_yscale("log")
