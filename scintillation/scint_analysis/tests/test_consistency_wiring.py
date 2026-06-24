"""Checks for the scattering-scintillation consistency wiring (band_consistency)."""

from __future__ import annotations

import numpy as np
import pytest

from scint_analysis.consistency import band_consistency, consistency_table


def _dnu_for_C(tau_1ghz_ms, alpha, nu0_ghz, target_C):
    """Dnu (MHz) that makes 2*pi*tau_band*Dnu == target_C exactly."""
    tau_band_s = (tau_1ghz_ms * nu0_ghz ** (-alpha)) * 1e-3
    return target_C / (2 * np.pi * tau_band_s) / 1e6


def test_band_consistency_oracle_single_screen():
    """Construct Dnu so C_implied == 1 -> consistent (one screen does both)."""
    tau_1ghz, alpha, nu0 = 0.5, 4.0, 1.4
    dnu = _dnu_for_C(tau_1ghz, alpha, nu0, 1.0)
    r = band_consistency(tau_1ghz, alpha, nu0, dnu)
    assert r["C_implied"] == pytest.approx(1.0, rel=1e-6)
    assert r["consistent"] is True
    assert r["tau_band_ms"] == pytest.approx(tau_1ghz * nu0 ** (-alpha), rel=1e-12)


def test_band_consistency_flags_multiscreen():
    """C_implied > 2π·2.0 (canonical upper bound) -> inconsistent (>=2 screens)."""
    tau_1ghz, alpha, nu0 = 0.5, 4.0, 1.4
    dnu = _dnu_for_C(tau_1ghz, alpha, nu0, 15.0)
    r = band_consistency(tau_1ghz, alpha, nu0, dnu)
    assert r["C_implied"] == pytest.approx(15.0, rel=1e-6)
    assert r["consistent"] is False


@pytest.mark.slow
def test_consistency_table_real_files():
    """Reader parses the real multiscale JSONs (casey/freya/wilhelm)."""
    df = consistency_table()
    if df.empty:
        pytest.skip("no multiscale results present")
    assert "burst" in df.columns
    assert any(c.startswith("C_implied_") for c in df.columns)
