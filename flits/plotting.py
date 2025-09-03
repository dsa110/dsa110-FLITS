"""Plotting helpers for FRB simulations."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .models import FRBModel
from .params import FRBParams


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
    """Plot the average model time series over all frequencies."""
    model = FRBModel(params)
    spec = model.simulate(t, freqs)
    avg = spec.mean(axis=0)
    return plot_time_series(t, avg, ax=ax)


__all__ = ["plot_time_series", "plot_model"]
