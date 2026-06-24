"""Phase 5: NE2025 Galactic-floor wiring + extragalactic-excess flag.

The real floor needs the optional ``mwprop`` package (or ``pygedm``), so the wiring
proof injects a synthetic ``query_ne2025_scint`` module rather than skipping the whole
file when the dep is absent. A final test exercises the real floor under
``importorskip`` so a complete environment still checks the analytic scaling.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

import numpy as np
import pytest

from scint_analysis.floor_wiring import (
    attach_galactic_floor,
    attach_galactic_floor_all,
    extragalactic_excess,
)

_BANDS = {"CHIME": 600.19, "DSA": 1405.0}


def _fake_query_module(monkeypatch, *, chime_bw_khz=1e5, dsa_bw_khz=5e4):
    """Inject a synthetic query_ne2025_scint so the wiring runs without mwprop."""
    fake = types.ModuleType("scintillation.ne2025.query_ne2025_scint")
    fake.BAND_CENTERS_MHZ = dict(_BANDS)
    fake.galactic_floor = lambda coord, bands, alpha=4.4, model="ne2025": {
        "CHIME": {"tau_ms": 1e-4, "bw_kHz": chime_bw_khz},
        "DSA": {"tau_ms": 1e-5, "bw_kHz": dsa_bw_khz},
    }
    monkeypatch.setitem(sys.modules, "scintillation.ne2025.query_ne2025_scint", fake)
    return fake


def test_excess_flag_logic():
    # measured 2.7 MHz = 2700 kHz; below a 50000 kHz MW floor -> extragalactic excess
    assert extragalactic_excess(2.7, 50000.0) is True
    # measured Δν above the MW floor -> consistent with the Galaxy (no excess)
    assert extragalactic_excess(2.7, 1000.0) is False
    # unusable inputs -> None (caller omits the flag)
    assert extragalactic_excess(None, 5.0) is None
    assert extragalactic_excess(2.7, 0.0) is None
    assert extragalactic_excess(np.nan, 5.0) is None


def test_attach_flags_extragalactic_with_synthetic_floor(monkeypatch):
    _fake_query_module(monkeypatch, dsa_bw_khz=5e4)
    comp = {"scaling_index": 4.4, "subband_measurements": [{"freq_mhz": 1405.0, "bw": 2.7}]}
    attach_galactic_floor(comp, coord=None)  # coord unused by the fake
    assert comp["galactic_floor"]["DSA"]["bw_kHz"] == 5e4
    assert comp["extragalactic_excess"] is True  # 2700 kHz < 50000 kHz


def test_attach_no_excess_when_measured_above_floor(monkeypatch):
    _fake_query_module(monkeypatch, dsa_bw_khz=100.0)  # tiny floor -> measured is above it
    comp = {"subband_measurements": [{"freq_mhz": 1405.0, "bw": 2.7}]}
    attach_galactic_floor(comp, coord=None)
    assert comp["extragalactic_excess"] is False  # 2700 kHz > 100 kHz


def test_attach_noop_without_dep(monkeypatch):
    # Force the lazy import to fail -> clean no-op, floor=None, no flag.
    monkeypatch.setitem(sys.modules, "scintillation.ne2025.query_ne2025_scint", None)
    comp = {"subband_measurements": [{"freq_mhz": 1405.0, "bw": 2.7}]}
    attach_galactic_floor(comp, coord=None)
    assert comp["galactic_floor"] is None
    assert "extragalactic_excess" not in comp


def test_attach_all_iterates_components(monkeypatch):
    _fake_query_module(monkeypatch)
    fr = {
        "components": {
            "scint_scale": {"subband_measurements": [{"freq_mhz": 600.0, "bw": 0.05}]},
        }
    }
    attach_galactic_floor_all(fr, ra_deg=170.0, dec_deg=70.0)
    assert "galactic_floor" in fr["components"]["scint_scale"]


@pytest.mark.slow
def test_real_floor_band_scaling():
    """With the real NE2025 floor, the DSA band carries a finite positive floor and
    the excess flag resolves. Mirrors tests/test_ne2025_floor.py's analytic check."""
    pytest.importorskip(
        "mwprop.nemod.NE2025",
        reason="real NE2025 floor needs the optional mwprop package (pip install mwprop)",
    )
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    coord = SkyCoord(ra=170.0 * u.deg, dec=70.0 * u.deg, frame="icrs")
    comp = {"scaling_index": 4.4, "subband_measurements": [{"freq_mhz": 1405.0, "bw": 2.7}]}
    attach_galactic_floor(comp, coord)
    assert comp["galactic_floor"]["DSA"]["bw_kHz"] > 0
    assert isinstance(comp.get("extragalactic_excess"), bool)
