#!/usr/bin/env python3
"""
DESI DR8 (VII/292) pencil‑beam host search via Vizier
====================================================
Given a list of (RA, Dec, z_max) sight‑lines this script:

1. Converts a *rest‑frame* impact‑parameter ceiling (default 100 kpc, override
   with `--impact`) into an angular radius for each beam using **Planck18**.
2. Queries the **DESI DR8 northern catalogue** (`VII/292/north`) with
   **astroquery.vizier**, requesting only the columns we need.
3. Casts all numeric columns to floats, discards any rows with non‑finite
   values **before** formatting, and filters on photo‑z ≤ z_max.
4. Computes the impact parameter for every remaining source and keeps those
   within the ceiling, streaming results to one CSV per beam plus a summary.

The code is fully self‑contained and memory‑light; the bottleneck is Vizier
latency (≈1 s per beam).  NumPy/Pandas formatting warnings are suppressed by
explicit NaN masks and `float_format="%.6g"`.
"""
from __future__ import annotations

import argparse
import pathlib
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astroquery.vizier import Vizier

# ---------------------------------------------------------------------------
# USER CONFIG: list of (RA, Dec, z_max)
# ---------------------------------------------------------------------------
TARGETS: List[Tuple[str, str, float]] = [
    ("20h40m47.886s", "+72d52m56.378s", 0.0430),
    ("08h58m52.92s",  "+73d29m27.0s",   0.4790),
    ("21h12m10.760s", "+72d49m38.20s",  0.3005),
    ("04h45m38.64s",  "+70d18m26.6s",   0.2505),
    ("21h00m31.09s",  "+72d02m15.22s",  0.5100),
    ("11h51m07.52s",  "+71d41m44.3s",   0.2710),
    ("05h52m45.12s",  "+74d12m01.7s",   1.0000),
    ("20h20m08.92s",  "+70d47m33.96s",  0.3024),
    ("02h39m03.96s",  "+71d01m04.3s",   1.0000),
    ("20h50m28.59s",  "+73d54m00.0s",   0.0740),
    ("11h19m56.05s",  "+70d40m34.4s",   0.2870),
    ("22h23m53.94s",  "+73d01m33.26s",  1.0000),
]

# ---------------------------------------------------------------------------
# CONSTANTS & VIZIER SETUP
# ---------------------------------------------------------------------------
#CATALOG   = "VII/292/north"  # DESI DR8 northern catalogue
#RA_COL    = "RAJ2000"
#DEC_COL   = "DEJ2000"
#ZPHOT_COL = "zphot"
#OBJ_COL   = "id"
#PSTAR_COL = "pstar"  # only star‑prob present in DR8 north

CATALOG = "VII/275/glade1"
RA_COL = "RAJ2000"
DEC_COL = "DEJ2000"
ZPHOT_COL = "zph2MPZ"
OBJ_COL = "id"

COSMO = Planck18
Vizier.row_limit = -1  # no limit, trust radius cone to keep it tiny

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def theta_max_kpc(z: float, impact_kpc: float) -> float:
    """Angular radius (rad) subtending *impact_kpc* at redshift *z*."""
    return (impact_kpc * u.kpc / COSMO.angular_diameter_distance(z))\
        .to(u.rad, equivalencies=u.dimensionless_angles()).value


def query_beam(centre: SkyCoord, radius_rad: float) -> pd.DataFrame:
    """Query Vizier and return a pandas DataFrame (can be empty)."""
    Vizier.columns = [OBJ_COL, RA_COL, DEC_COL, ZPHOT_COL, PSTAR_COL]
    res = Vizier.query_region(centre, radius=radius_rad * u.rad, catalog=CATALOG)
    return res[0].to_pandas() if res else pd.DataFrame()

# ---------------------------------------------------------------------------
# MAIN ROUTINE
# ---------------------------------------------------------------------------

def main(output_dir: pathlib.Path, impact_kpc: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: defaultdict[int, int] = defaultdict(int)

    t0 = time.time()
    for idx, (ra_str, dec_str, z_max) in enumerate(TARGETS, start=1):
        centre = SkyCoord(ra_str, dec_str, frame="icrs")
        theta  = theta_max_kpc(z_max, impact_kpc)

        df = query_beam(centre, theta)
        if df.empty:
            continue

        # --- cast numeric columns & basic z cut
        for col in (ZPHOT_COL, PSTAR_COL):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[(df[ZPHOT_COL].notna()) & (df[ZPHOT_COL] >= 0) & (df[ZPHOT_COL] <= z_max)]
        if df.empty:
            continue

        # --- separation & impact
        coords = SkyCoord(df[RA_COL].to_numpy() * u.deg, df[DEC_COL].to_numpy() * u.deg, frame="icrs")
        theta_arr = coords.separation(centre).radian
        d_a       = COSMO.angular_diameter_distance(df[ZPHOT_COL].to_numpy()).to(u.kpc)

        finite_mask = np.isfinite(theta_arr) & np.isfinite(d_a.value) & np.isfinite(df[PSTAR_COL])
        if not np.any(finite_mask):
            continue
        df          = df.loc[finite_mask].reset_index(drop=True)
        theta_arr   = theta_arr[finite_mask]
        d_a         = d_a[finite_mask]

        impact = theta_arr * d_a
        df["impact_kpc"] = impact.value
        df = df[impact <= impact_kpc * u.kpc].reset_index(drop=True)
        if df.empty:
            continue

        out_name = output_dir / f"beam_{idx:02d}_{ra_str.replace(' ', '')}{dec_str.replace(' ', '')}_matches.csv"
        df.to_csv(out_name, index=False, float_format="%.6g", na_rep="nan")
        counts[idx] = len(df)
        print(f"Beam {idx:2d}: {len(df):3d} matches → {out_name.name}")

    # summary
    summary = [(i+1, ra, dec, z, counts.get(i+1, 0)) for i, (ra, dec, z) in enumerate(TARGETS)]
    pd.DataFrame(summary, columns=["beam#", "RA", "Dec", "z_max", "N_gal"])\
        .to_csv(output_dir / "desi_beam_summary.csv", index=False)
    print(f"Completed in {time.time() - t0:.1f} s → summary written.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DESI DR8 Vizier pencil‑beam search")
    ap.add_argument("output_dir", type=pathlib.Path, help="Directory for per‑beam CSVs")
    ap.add_argument("--impact", type=float, default=100.0,
                    help="Maximum rest‑frame impact parameter in kpc (default 100)")
    args = ap.parse_args()

    main(args.output_dir, impact_kpc=args.impact)
