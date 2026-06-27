"""Tests for scintillation ↔ tau_consistency bridge."""

import math

import numpy as np

from galaxies.foreground.scintillation_bridge import (
    _burst_spectrum_for_fit,
    build_scintillation_source_block,
    consistency_failed_for_component,
    format_two_screen_coherence,
    merge_source_into_config,
)
from scintillation.scint_analysis.core import DynamicSpectrum


def test_source_block_wilhelm_has_tau_and_distance():
    block = build_scintillation_source_block("wilhelm")
    assert math.isfinite(block["ra_deg"])
    assert math.isfinite(block["dec_deg"])
    assert block.get("tau_d_ms", 0) > 0 or "tau_d_ms" in block
    assert math.isfinite(block.get("distance_mpc", np.nan))


def test_merge_source_preserves_existing_config():
    cfg = {"source": {"tau_d_ms": 9.9, "distance_mpc": 100.0}, "analysis": {}}
    out = merge_source_into_config(cfg, "wilhelm")
    assert out["source"]["tau_d_ms"] == 9.9
    assert out["source"]["distance_mpc"] == 100.0


def test_consistency_failed_when_tau_large_dnu_small():
    comp = {"bw_at_ref_mhz": 0.01, "subband_measurements": [{}]}
    cfg = {
        "source": {"nickname": "wilhelm", "tau_d_ms": 0.5},
        "analysis": {"fitting": {}},
    }
    assert consistency_failed_for_component(comp, cfg, band="chime") is True


def test_burst_spectrum_for_fit_from_dynamic_spectrum():
    freqs = np.linspace(400, 800, 64)
    power = np.ones((64, 100))
    ds = DynamicSpectrum(power, freqs, np.linspace(0, 1, 100))
    spec, ch = _burst_spectrum_for_fit(ds, (10, 90))
    assert spec is not None
    assert spec.shape == (64,)
    assert ch == ds.channel_width_mhz
