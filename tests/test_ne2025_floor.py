"""Checks for the NE2025 per-burst Galactic-floor wiring (query_ne2025_scint)."""

import importlib.util
import pathlib

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

_MOD = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scintillation"
    / "ne2025"
    / "query_ne2025_scint.py"
)
_spec = importlib.util.spec_from_file_location("query_ne2025_scint", _MOD)
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)


def test_band_scaling_is_analytic():
    """tau ~ nu^-alpha and Dnu ~ nu^+alpha, so both band ratios equal
    (f_DSA/f_CHIME)^alpha. Catches an exponent-sign error in the wiring."""
    alpha = 4.4
    coord = SkyCoord(ra=169.98 * u.deg, dec=70.68 * u.deg, frame="icrs")  # casey
    floor = q.galactic_floor(coord, q.BAND_CENTERS_MHZ, alpha=alpha)
    r = (q.BAND_CENTERS_MHZ["DSA"] / q.BAND_CENTERS_MHZ["CHIME"]) ** alpha
    assert floor["CHIME"]["tau_ms"] / floor["DSA"]["tau_ms"] == pytest.approx(r, rel=1e-6)
    assert floor["DSA"]["bw_kHz"] / floor["CHIME"]["bw_kHz"] == pytest.approx(r, rel=1e-6)
    # Floor must be a real, positive prediction.
    assert floor["DSA"]["tau_ms"] > 0 and np.isfinite(floor["DSA"]["tau_ms"])


@pytest.mark.slow
def test_floor_for_all_bursts():
    """All 12 catalog bursts get a finite, positive MW floor at both bands."""
    df = q.floor_for_bursts()
    assert len(df) == 12
    for col in ("tau_ms_CHIME", "bw_kHz_CHIME", "tau_ms_DSA", "bw_kHz_DSA", "l_deg", "b_deg"):
        assert col in df.columns
    for col in ("tau_ms_CHIME", "bw_kHz_CHIME", "tau_ms_DSA", "bw_kHz_DSA"):
        assert np.all(np.isfinite(df[col])) and np.all(df[col] > 0)
