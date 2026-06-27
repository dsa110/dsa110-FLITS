"""Tests for tau consistency and attribution matrix."""

import math

from galaxies.foreground.attribution_matrix import build_attribution_matrix
from galaxies.foreground.tau_consistency import (
    consistency_status,
    scale_tau_1ghz_ms,
)


def test_scale_tau_alpha4():
    tau = scale_tau_1ghz_ms(1.0, 600.0, alpha=4.0)
    assert tau > 1.0


def test_consistency_gate_matches_physics():
    # C = 2π τ Δν ∈ (0.628, 12.57) for consistent
    assert consistency_status(0.05, 0.01) == "consistent"
    assert consistency_status(0.1, 50.0) == "inconsistent"


def test_attribution_matrix_twelve_rows():
    df = build_attribution_matrix()
    assert len(df) == 12
    assert set(df.nickname) == {
        "zach",
        "whitney",
        "oran",
        "isha",
        "wilhelm",
        "phineas",
        "freya",
        "hamilton",
        "mahi",
        "chromatica",
        "casey",
        "johndoeii",
    }


def test_freya_inverse_dnu_flagged():
    df = build_attribution_matrix()
    freya = df[df.nickname == "freya"].iloc[0]
    assert freya.dnu_status == "inverse_scaling"
    assert "inverse_dnu_scaling" in freya.multi_screen_triggers


def test_unmeasured_dnu_placeholder():
    df = build_attribution_matrix()
    isha = df[df.nickname == "isha"].iloc[0]
    assert str(isha.dnu_chime_mhz).startswith("N/A —")


def test_joint_tau_loaded_for_casey():
    df = build_attribution_matrix()
    casey = df[df.nickname == "casey"].iloc[0]
    assert math.isfinite(casey.tau_joint_1ghz_ms)
