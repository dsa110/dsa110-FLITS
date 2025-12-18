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

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

# Shared configuration
from config import (
    TARGETS_TUPLE as TARGETS,
    angular_diameter_distance_fast,
    COSMO,
)

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
PSTAR_COL = "Bmag"  # Use Bmag as placeholder since GLADE1 has no pstar column

Vizier.row_limit = -1  # no limit, trust radius cone to keep it tiny

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def theta_max_kpc(z: float, impact_kpc: float) -> float:
    """Angular radius (rad) subtending *impact_kpc* at redshift *z*. Uses fast lookup."""
    d_a_mpc = angular_diameter_distance_fast(np.array([z]))[0]
    return (impact_kpc / 1000.0) / d_a_mpc  # kpc -> Mpc, result in radians


def query_beam(centre: SkyCoord, radius_rad: float) -> pd.DataFrame:
    """Query Vizier and return a pandas DataFrame (can be empty)."""
    # Request only columns that exist in the catalog
    Vizier.columns = [OBJ_COL, RA_COL, DEC_COL, ZPHOT_COL]
    if PSTAR_COL:
        Vizier.columns.append(PSTAR_COL)
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
        cols_to_cast = [ZPHOT_COL]
        if PSTAR_COL and PSTAR_COL in df.columns:
            cols_to_cast.append(PSTAR_COL)
        for col in cols_to_cast:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[(df[ZPHOT_COL].notna()) & (df[ZPHOT_COL] >= 0) & (df[ZPHOT_COL] <= z_max)]
        if df.empty:
            continue

        # --- separation & impact
        coords = SkyCoord(df[RA_COL].to_numpy() * u.deg, df[DEC_COL].to_numpy() * u.deg, frame="icrs")
        theta_arr = coords.separation(centre).radian
        d_a       = COSMO.angular_diameter_distance(df[ZPHOT_COL].to_numpy()).to(u.kpc)

        finite_mask = np.isfinite(theta_arr) & np.isfinite(d_a.value)
        if PSTAR_COL and PSTAR_COL in df.columns:
            finite_mask &= np.isfinite(df[PSTAR_COL])
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
