"""Truth-known single/two-screen injections through the production ACF path.

A1 trigger-calibration Phase 3 (Faber2026
docs/rse/specs/plan-a1-trigger-calibration.md): injections render to
per-subband ACFs via the production ``calculate_acfs_for_subbands`` (sim_gate
pattern), stamped with the HWHM convention and their truth parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "simulation"))
sys.path.insert(0, str(_root / "scintillation"))

from trigger_calibration import inject_single_screen, inject_two_screen  # noqa: E402


def test_single_screen_recovers_injected_hwhm():
    out = inject_single_screen(
        dnu_hwhm_mhz=0.4, band_width_mhz=6.0, channel_width_mhz=0.05,
        snr=50.0, num_subbands=4, seed=7,
    )
    assert out["convention"] == "HWHM"
    assert out["truth"]["dnu_hwhm_mhz"] == 0.4
    subs = out["subbands"]
    assert len(subs) >= 3  # at least 3 of 4 subbands must yield an ACF
    for s in subs:
        assert np.all(s["lags"] > 0)  # one-sided, lag-0 excluded
    med = float(np.median([s["dnu_hwhm_est_mhz"] for s in subs]))
    # generous window: ~5 scintles per 1.5 MHz subband gives large scatter;
    # the absolute bias is tracked against the P1.2 dnu-recovery entry
    assert 0.1 < med < 1.2


def test_two_screen_injection_exposes_both_scales():
    out = inject_two_screen(
        dnu1_hwhm_mhz=0.15, dnu2_hwhm_mhz=1.5, m2_ratio=1.0,
        band_width_mhz=6.0, channel_width_mhz=0.05, snr=50.0,
        num_subbands=1, seed=8,
    )
    assert out["truth"]["f"] == 10.0
    assert out["truth"]["dnu1_hwhm_mhz"] == 0.15
    assert len(out["subbands"]) == 1
    s = out["subbands"][0]
    # narrow screen must be resolved: estimated HWHM sits between the scales
    assert 0.05 < s["dnu_hwhm_est_mhz"] < 1.5
