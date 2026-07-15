"""Regression checks for the p1 common-mode mechanism gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE = Path(__file__).with_name("measure_common_mode.py")
SELECTOR = Path(__file__).with_name("select_variant.py")


def _module():
    spec = importlib.util.spec_from_file_location("measure_common_mode", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _selector():
    spec = importlib.util.spec_from_file_location("select_variant", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_measurement_recovers_seeded_common_lorentzian():
    module = _module()
    rng = np.random.default_rng(20260714)
    channels = 64 * 64
    times = 240
    width_bins = 5.0
    frequencies = 650.0 + np.arange(channels) * 0.006103515625
    modes = np.fft.rfftfreq(channels)
    common = np.fft.irfft(
        np.sqrt(np.exp(-2.0 * np.pi * width_bins * modes))
        * (rng.normal(size=modes.size) + 1j * rng.normal(size=modes.size)),
        n=channels,
    )
    common /= np.std(common)
    pol0 = 10.0 + common[:, None] + rng.normal(0.0, 0.2, (channels, times))
    pol1 = 10.0 + common[:, None] + rng.normal(0.0, 0.2, (channels, times))

    result = module.measure_common_mode(pol0, pol1, frequencies)

    assert result["cross_correlation"][0] > 0.5
    assert result["lorentzian_fit"]["amplitude"] > 0.2
    assert not result["mechanism_gate"]["eligible"]


def test_selector_stops_when_no_variant_passes():
    selector = _selector()
    failed = {
        "variant": {"window": "hann", "oversample": 2},
        "mechanism_gate": {"eligible": False},
        "lorentzian_fit": {"amplitude": 0.2},
        "cross_correlation": [0.3],
    }

    verdict = selector.select_variant([failed])

    assert verdict["status"] == "DOCUMENTED-FAIL"
    assert verdict["selected_variant"] is None


def test_selector_uses_frozen_tie_break_order():
    selector = _selector()

    def candidate(window, oversample, amplitude, lag1):
        return {
            "variant": {"window": window, "oversample": oversample},
            "mechanism_gate": {"eligible": True},
            "lorentzian_fit": {"amplitude": amplitude},
            "cross_correlation": [lag1],
        }

    verdict = selector.select_variant(
        [
            candidate("hann", 4, 0.03, 0.04),
            candidate("blackmanharris", 4, 0.02, 0.03),
            candidate("hann", 2, 0.02, 0.03),
        ]
    )

    assert verdict["selected_variant"] == {"window": "hann", "oversample": 2}
