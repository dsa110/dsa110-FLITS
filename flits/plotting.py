"""Plotting helpers for FRB simulations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401

    _SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    _SCIENCEPLOTS_AVAILABLE = False

from scattering.scat_analysis.burstfit import FRBModel, FRBParams

# Default style for all FLITS plots
DEFAULT_STYLE = ["science", "notebook"]


def use_flits_style(style: list[str] | None = None) -> None:
    """Apply the FLITS plotting style.

    Parameters
    ----------
    style : list of str, optional
        List of matplotlib/scienceplots styles to use.
        Defaults to ["science", "notebook"].

    Notes
    -----
    Requires the SciencePlots package: pip install SciencePlots
    If not installed, falls back to matplotlib defaults with a warning.
    """
    if style is None:
        style = DEFAULT_STYLE

    if not _SCIENCEPLOTS_AVAILABLE:
        import warnings

        warnings.warn(
            "SciencePlots not installed. Install with: pip install SciencePlots\n"
            "Falling back to matplotlib defaults.",
            UserWarning,
            stacklevel=2,
        )
        return

    plt.style.use(style)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['axes.unicode_minus'] = False


# Automatically apply the style when this module is imported
use_flits_style()


def plot_time_series(t: np.ndarray, data: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot a simple time series."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, data)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Intensity [arb]")
    return ax


def plot_model(
    t: np.ndarray, freqs: np.ndarray, params: FRBParams, ax: plt.Axes | None = None
) -> plt.Axes:
    """Plot the average model time series over all frequencies.

    ``params`` is a core :class:`FRBParams`; ``freqs`` in MHz.
    """
    freqs_ghz = freqs / 1000.0
    df_MHz = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    model = FRBModel(time=t, freq=freqs_ghz, dm_init=params.delta_dm, df_MHz=df_MHz)
    spec = model(params, "M3")
    avg = spec.mean(axis=0)
    return plot_time_series(t, avg, ax=ax)


__all__ = ["plot_time_series", "plot_model", "use_flits_style", "DEFAULT_STYLE"]
