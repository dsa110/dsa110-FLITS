"""Data-free regression gates for the objective-window campaign."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))
sys.path.insert(0, str(_test_dir.parent.parent))

from scint_analysis import analysis
from scint_analysis import window_refit
from scint_analysis.core import DynamicSpectrum


def test_fixed_subband_slices_are_reused_exactly():
    rng = np.random.default_rng(4)
    power = rng.normal(0.0, 0.05, (256, 80))
    power[:, 30:40] += 1.0 + 0.15 * np.sin(np.arange(256)[:, None] / 3.0)
    spectrum = DynamicSpectrum(power, np.linspace(400.0, 800.0, 256), np.arange(80))
    slices = [[0, 53], [53, 119], [119, 181], [181, 256]]
    config = {
        "analysis": {
            "acf": {
                "num_subbands": 4,
                "use_snr_subbanding": True,
                "subband_channel_slices": slices,
                "max_lag_mhz": 30.0,
            },
            "noise": {"disable_template": True},
            "self_noise": {"disable": True},
            "rfi_masking": {"off_burst_buffer": 2},
        }
    }

    result = analysis.calculate_acfs_for_subbands(spectrum, config, (30, 40))

    assert result["subband_channel_slices"] == [tuple(item) for item in slices]


def test_physical_alpha_bounds_are_open():
    assert window_refit.alpha_is_physical({"alpha": 4.0})
    assert not window_refit.alpha_is_physical({"alpha": 1.5})
    assert not window_refit.alpha_is_physical({"alpha": 6.0})
    assert not window_refit.alpha_is_physical({"alpha": -2.0})


def test_two_lorentzian_candidate_still_passes_shape_gate():
    rng = np.random.default_rng(7)
    lags = np.arange(1, 601) * 0.01
    acf = window_refit._lorentz2(lags, 0.45, 0.08, 0.35, 1.2, 0.0)
    acf += rng.normal(0.0, 0.002, lags.size)

    result = window_refit._fit_subband(lags, acf)

    assert result["model_sel"] == "2L"
    assert result["shape_ok"]
    assert result["dbic_line"] >= 6.0
    assert abs(result["gamma"] - 0.08) < 0.03


def test_smooth_linear_artifact_is_not_resolved():
    rng = np.random.default_rng(8)
    lags = np.arange(1, 501) * 0.01
    acf = 0.7 - 0.08 * lags + rng.normal(0.0, 0.002, lags.size)

    result = window_refit._fit_subband(lags, acf)

    assert not result["shape_ok"]
    assert not result["resolved"]
