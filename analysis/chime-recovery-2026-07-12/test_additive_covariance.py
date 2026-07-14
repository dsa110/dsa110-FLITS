"""Regression tests for the bounded Freya A1 additive likelihood."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).with_name("validate_freya_additive_covariance.py")


def _module():
    spec = importlib.util.spec_from_file_location("freya_a1_validation", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fit_retains_short_lags_and_recovers_clean_lorentzian():
    module = _module()
    channel_width = 0.006103608758678547
    truth_width = 8.0 * channel_width
    truth_m = 0.7
    lags = channel_width * np.arange(1, 81, dtype=float)
    observed = truth_m**2 / (1.0 + (lags / truth_width) ** 2)
    error = np.full_like(lags, 1e-4)
    kernel = np.zeros_like(lags)
    kernel_covariance = np.zeros((lags.size, lags.size))

    fit = module._fit_additive_likelihood(
        lags,
        observed,
        error,
        kernel,
        kernel_covariance,
        channel_width_mhz=channel_width,
    )

    assert fit is not None
    assert np.isclose(fit["fit_lags_mhz"][0], 2.0 * channel_width)
    assert np.isclose(fit["dnu_mhz"], truth_width, rtol=0.01)
    assert np.isclose(fit["m"], truth_m, rtol=0.01)
