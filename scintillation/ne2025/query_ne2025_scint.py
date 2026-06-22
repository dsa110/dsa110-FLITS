#!/usr/bin/env python3
"""
query_ne2025_scint.py  –  v3, NE2025, sexagesimal-aware
-------------------------------------------------------
Predict the Milky-Way scattering floor (tau_scatt and scintillation bandwidth
Dnu_scint) for a list of sky positions using NE2025
[Ocker & Cordes, ADS:2026ApJ..1002....3O] via mwprop.nemod.NE2025.

For an extragalactic source (FRB) the relevant quantity is the TOTAL Galactic
scattering integrated to the edge of the disk, so we call NE2025 in d->DM mode
(ndir<0) with a large distance (30 kpc, capped at the Galactic boundary). The MW
floor is independent of the source DM, so no 'dm' column is needed.

INPUT  : CSV/TSV with either …
         • a single column  'coord'  containing ICRS strings
           e.g.  01h24m50.45s +72d39m14.1s
         • OR columns  ra, dec   (sexagesimal or degrees)
         • OR legacy  ra_deg, dec_deg  (degrees)

OUTPUT : CSV with tau_ms_<freq>MHz and bw_kHz_<freq>MHz (the predicted MW floor).
"""

import argparse
import re

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from mwprop.nemod.NE2025 import ne2025

EDGE_KPC = 30.0  # integrate to the Galactic boundary (NE2025 caps at the edge)


# ---------------------------------------------------------------------------
def coord_from_row(row):
    """
    Return an astropy SkyCoord from a dataframe row containing any of:
      - 'coord'  (full ICRS string)
      - 'ra', 'dec'           (sexagesimal or degrees)
      - 'ra_str', 'dec_str'
      - 'ra_deg', 'dec_deg'
    """
    # Case 1: single 'coord' column
    if "coord" in row and pd.notna(row["coord"]):
        return SkyCoord(row["coord"], frame="icrs")

    # Helper: find any matching RA/Dec pair
    variants = [("ra", "dec"), ("ra_str", "dec_str"), ("ra_deg", "dec_deg")]
    for ra_key, dec_key in variants:
        if ra_key in row and dec_key in row and pd.notna(row[ra_key]):
            ra_val, dec_val = row[ra_key], row[dec_key]
            # Detect degrees vs sexagesimal by simple regex (contains h or : ?)
            sexa = bool(re.search("[hms:]", str(ra_val)))
            unit = (u.deg, u.deg) if not sexa else None
            return SkyCoord(ra_val, dec_val, unit=unit, frame="icrs")

    raise ValueError("Row is missing a recognisable coordinate column set.")


def query_single(coord_icrs, freq_mhz, alpha=4.4):
    """Return (tau_scatt_ms, Dnu_scint_kHz) of the MW floor at freq_mhz.

    Uses NE2025's native TAU and SBW (both reported @1 GHz) and scales them by
    nu^-alpha / nu^+alpha (alpha=4.4 = 22/5; scattering_functions2020.py:90,
    tauiss ~ nu^-4.4). Taking SBW from the model preserves NE2025's C1=1.16,
    unlike re-deriving Dnu_d = 1/(2*pi*tau) which implicitly assumes C1=1.
    """
    gl, gb = coord_icrs.galactic.l.value, coord_icrs.galactic.b.value
    Dk, Dv, Du, Dd = ne2025(ldeg=gl, bdeg=gb, dmd=EDGE_KPC, ndir=-1, classic=False, dmd_only=False)
    nu_ghz = freq_mhz / 1000.0
    tau_ms = Dv["TAU"] * nu_ghz ** (-alpha)  # ms
    bw_kHz = Dv["SBW"] * nu_ghz**alpha * 1e3  # MHz @1GHz -> kHz at freq
    return tau_ms, bw_kHz


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("infile", help="Input CSV/TSV with sky positions")
    p.add_argument("--freq", type=float, default=600.0, help="Frequency in MHz (default 600)")
    p.add_argument("--alpha", type=float, default=4.4, help="Scattering index α (default 4.4)")
    p.add_argument("--out", default="ne2025_results.csv", help="Output CSV file name")
    args = p.parse_args()

    # Load table (auto-detect separator)
    df = pd.read_csv(args.infile, sep=None, engine="python")

    taus, bws = [], []
    for _idx, row in df.iterrows():
        coord = coord_from_row(row)
        tau_ms, bw_kHz = query_single(coord, args.freq, args.alpha)
        taus.append(tau_ms)
        bws.append(bw_kHz)

    df[f"tau_ms_{args.freq:.0f}MHz"] = taus
    df[f"bw_kHz_{args.freq:.0f}MHz"] = bws
    df.to_csv(args.out, index=False)
    print(f"[ok] {len(df)} positions processed -> '{args.out}'")


if __name__ == "__main__":
    main()
