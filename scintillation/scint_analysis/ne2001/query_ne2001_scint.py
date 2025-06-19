#!/usr/bin/env python3
"""
query_ne2001_scint.py  –  v2, sexagesimal-aware
------------------------------------------------
Compute NE2001 τ_scatt and Δν_scint for a list of sky positions.

INPUT  : CSV/TSV with either …
         • a single column  'coord'  containing ICRS strings
           e.g.  01h24m50.45s +72d39m14.1s
         • OR columns  ra, dec   (sexagesimal or degrees)
         • OR legacy  ra_deg, dec_deg  (degrees)

OPTION  : 'dm' column for custom DM; otherwise DM=20 pc cm⁻³.

OUTPUT : CSV with τ_scatt_ms_xMHz and bw_kHz_xMHz
"""

import argparse, re
import pandas as pd, numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from mwprop.ne2001p.NE2001 import ne2001


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
    if 'coord' in row and pd.notna(row['coord']):
        return SkyCoord(row['coord'], frame='icrs')

    # Helper: find any matching RA/Dec pair
    variants = [('ra', 'dec'),
                ('ra_str', 'dec_str'),
                ('ra_deg', 'dec_deg')]
    for ra_key, dec_key in variants:
        if ra_key in row and dec_key in row and pd.notna(row[ra_key]):
            ra_val, dec_val = row[ra_key], row[dec_key]
            # Detect degrees vs sexagesimal by simple regex (contains h or : ?)
            sexa = bool(re.search('[hms:]', str(ra_val)))
            unit = (u.deg, u.deg) if not sexa else None
            return SkyCoord(ra_val, dec_val, unit=unit, frame='icrs')

    raise ValueError("Row is missing a recognisable coordinate column set.")


def query_single(coord_icrs, dm_pc_cm3, freq_mhz, alpha=4.0):
    """Return (τ_scatt_ms, Δν_scint_kHz) at freq_mhz for one SkyCoord."""
    l, b = coord_icrs.galactic.l.value, coord_icrs.galactic.b.value
    # NE2001p call
    Dk, Dv, Du, Dd = ne2001(ldeg=l, bdeg=b, dmd=dm_pc_cm3,
                            ndir=-1, classic=False, dmd_only=False)
    tau1_ms = Dv['TAU']                # at 1 GHz
    tau_ms  = tau1_ms * (1_000./freq_mhz)**alpha
    bw_kHz  = 1. / (2*np.pi * tau_ms*1e-3) / 1e3
    return tau_ms, bw_kHz


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("infile",  help="Input CSV/TSV with sky positions")
    p.add_argument("--freq",  type=float, default=600.,
                   help="Frequency in MHz (default 600)")
    p.add_argument("--alpha", type=float, default=4.0,
                   help="Scattering index α (default 4)")
    p.add_argument("--out",   default="ne2001_results.csv",
                   help="Output CSV file name")
    args = p.parse_args()

    # Load table (auto-detect separator)
    df = pd.read_csv(args.infile, sep=None, engine='python')
    if 'dm' not in df.columns:
        df['dm'] = 20.0

    taus, bws = [], []
    for idx, row in df.iterrows():
        coord = coord_from_row(row)
        τ_ms, Δν_kHz = query_single(coord, row['dm'], args.freq, args.alpha)
        taus.append(τ_ms); bws.append(Δν_kHz)

    df[f"tau_ms_{args.freq:.0f}MHz"] = taus
    df[f"bw_kHz_{args.freq:.0f}MHz"] = bws
    df.to_csv(args.out, index=False)
    print(f"✓  {len(df)} positions processed →  '{args.out}'")

if __name__ == "__main__":
    main()
