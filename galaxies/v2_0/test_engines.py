import os
import sys

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from galaxies.v2_0 import engines as eng
from galaxies.v2_0.config import VIZIER_CATALOGS
from galaxies.v2_0.engines import NedEngine, VizierEngine


def test_glade_catalog_id_is_gladep():
    # Vizier renamed the GLADE+ table VII/291/glade -> VII/291/gladep (the old
    # suffix returns 0 tables, silently dropping all GLADE+ galaxies).
    assert VIZIER_CATALOGS["GLADE+"] == "VII/291/gladep"


def test_glade_plus_standardization_maps_zcmb_and_converts_mass():
    # gladep schema: redshift as zcmb/zhelio (no literal 'z'); M* in 1e10 Msun
    # (linear), unlike the old glade M_star which was already log10(M/Msun).
    df = pd.DataFrame(
        {
            "ra": [10.0, 11.0],
            "dec": [20.0, 21.0],
            "zcmb": [0.0188, 0.05],
            "zhelio": [0.022, 0.051],
            "M*": [4.9, 0.0],
        }
    )
    out = eng._add_desi_stellar_mass(df.copy(), "VII/291/gladep")
    assert list(out["z"]) == [0.0188, 0.05]  # prefer CMB frame
    assert abs(out.loc[0, "M_star"] - np.log10(4.9e10)) < 1e-6  # 1e10 Msun -> log10
    assert pd.isna(out.loc[1, "M_star"])  # non-positive mass -> NaN
    assert out.loc[0, "mass_source"] == "catalog"


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    assert eng._retry(flaky, attempts=3, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_exhausting_attempts():
    def always_down():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        eng._retry(always_down, attempts=2, base_delay=0.0)


def test_engines():
    # Test with M31 (Andromeda) to verify engines are working
    coord = SkyCoord(10.6847, 41.2687, unit="deg", frame="icrs")
    radius = 10.0 * u.arcmin

    print(f"Testing M31 with {radius} radius")

    ned = NedEngine()
    ned_df = ned.query(coord, radius)
    print(f"NED returned {len(ned_df)} rows")

    # Try AllWISE as a fallback/test
    wise = VizierEngine("II/328/allwise")
    wise_df = wise.query(coord, radius)
    print(f"AllWISE returned {len(wise_df)} rows")
    if not wise_df.empty:
        print("AllWISE columns:", wise_df.columns.tolist())


if __name__ == "__main__":
    test_engines()
