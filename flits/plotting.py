"""Plotting helpers for FRB simulations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401

    _SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    _SCIENCEPLOTS_AVAILABLE = False

from .models import FRBModel
from .params import FRBParams

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


# Automatically apply the style when this module is imported
use_flits_style()


def plot_time_series(
    t: np.ndarray, data: np.ndarray, ax: plt.Axes | None = None
) -> plt.Axes:
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
    """Plot the average model time series over all frequencies."""
    model = FRBModel(params)
    spec = model.simulate(t, freqs)
    avg = spec.mean(axis=0)
    return plot_time_series(t, avg, ax=ax)


__all__ = ["plot_time_series", "plot_model", "use_flits_style", "DEFAULT_STYLE"]
