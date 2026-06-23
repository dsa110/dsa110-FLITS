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

# Band centres (MHz) = midpoints of f_min/f_max in configs/telescopes.yaml.
# CHIME 400.19-800.19, DSA 1311.25-1498.75.
BAND_CENTERS_MHZ = {"CHIME": 600.19, "DSA": 1405.0}


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


def galactic_floor(coord_icrs, bands=BAND_CENTERS_MHZ, alpha=4.4):
    """MW scattering floor at each band centre for one sky position.

    Returns {band: {"tau_ms": .., "bw_kHz": ..}}. Integrated to the Galactic
    edge, so it is z-independent and applies to every burst regardless of host
    redshift. The floor is the Galactic-vs-extragalactic discriminator: measured
    tau/Dnu well above this floor implies an extragalactic (host/intervening)
    screen.
    """
    return {
        b: dict(zip(("tau_ms", "bw_kHz"), query_single(coord_icrs, f, alpha), strict=True))
        for b, f in bands.items()
    }


def floor_for_bursts(catalog_path="configs/bursts.yaml", bands=BAND_CENTERS_MHZ, alpha=4.4):
    """Per-burst MW floor table from the burst catalog (one row per burst)."""
    from pathlib import Path

    import yaml

    cat = Path(catalog_path)
    if not cat.is_absolute():
        cat = Path(__file__).resolve().parents[2] / catalog_path
    bursts = yaml.safe_load(cat.read_text())["bursts"]

    rows = []
    for name, b in bursts.items():
        coord = SkyCoord(ra=b["ra_deg"] * u.deg, dec=b["dec_deg"] * u.deg, frame="icrs")
        row = {
            "burst": name,
            "l_deg": coord.galactic.l.value,
            "b_deg": coord.galactic.b.value,
        }
        for band, vals in galactic_floor(coord, bands, alpha).items():
            row[f"tau_ms_{band}"] = vals["tau_ms"]
            row[f"bw_kHz_{band}"] = vals["bw_kHz"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "infile", nargs="?", help="Input CSV/TSV with sky positions (omit with --bursts)"
    )
    p.add_argument(
        "--bursts",
        action="store_true",
        help="Emit per-burst MW floor at CHIME+DSA band centres from configs/bursts.yaml (ignores infile/--freq)",
    )
    p.add_argument("--freq", type=float, default=600.0, help="Frequency in MHz (default 600)")
    p.add_argument("--alpha", type=float, default=4.4, help="Scattering index α (default 4.4)")
    p.add_argument("--out", default="ne2025_results.csv", help="Output CSV file name")
    args = p.parse_args()

    if args.bursts:
        df = floor_for_bursts(alpha=args.alpha)
        df.to_csv(args.out, index=False)
        print(f"[ok] {len(df)} bursts -> '{args.out}' (MW floor at CHIME+DSA)")
        return

    if not args.infile:
        p.error("infile is required unless --bursts is given")

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
