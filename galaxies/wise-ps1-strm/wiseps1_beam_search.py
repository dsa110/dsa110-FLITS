#!/usr/bin/env python3
"""
Multi-catalog host-galaxy search for FRB sight-lines
====================================================
Version 3  ─ adds DESI Legacy DR8 photo-z support
-------------------------------------------------
Flags
-----
--strm          path to WISE-PS1-STRM CSV **or** Parquet  (optional)
--legacy        path to Legacy DR8/DR9 photo-z FITS file  (optional)
--outer-kpc     impact-parameter cut (kpc)                [default: 200]
--make-parquet  one-time STRM CSV → Parquet conversion
--clean-output  delete previous per-beam CSVs before run

Example
-------
python frb_host_search.py \
  --strm   wiseps1_strm_dec70_75.csv \
  --legacy dr8_photoz_dec70_75_full.fits \
  --outer-kpc 500 --make-parquet --clean-output
"""
from __future__ import annotations

import argparse
import pathlib
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.table import Table

# ─────────────────────────── FRB sight-lines ──────────────────────────────
TARGETS = [
    ("20h40m47.886s", "+72d52m56.378s", 0.0430),
    ("08h58m52.92s",  "+73d29m27.0s",   0.4790),
    ("21h12m10.760s", "+72d49m38.20s",  0.3005),
    ("04h45m38.64s",  "+70d18m26.6s",   0.2505),
    ("21h00m31.09s",  "+72d02m15.22s",  0.5100),
    ("11h51m07.52s",  "+71d41m44.3s",   0.2710),
    ("05h52m45.12s",  "+74d12m01.7s",   1.0000),
    ("22h23m53.94s",  "+73d01m33.26s",  1.0000),
    ("20h20m08.92s",  "+70d47m33.96s",  0.3024),
    ("02h39m03.96s",  "+71d01m04.3s",   1.0000),
    ("20h50m28.59s",  "+73d54m00.0s",   0.0740),
    ("11h19m56.05s",  "+70d40m34.4s",   0.2870),
]

# ──────────────── WISE-PS1-STRM (header-less CSV) info ────────────────────
STRM_COLNAMES = [
    "objID", "raMean", "decMean", "class",
    "prob_Galaxy", "prob_Star", "prob_QSO", "z_phot0",
]
STRM_COL_IDX = [0, 1, 3, 197, 198, 199, 200, 206]

# ──────────────── other constants ─────────────────────────────────────────
LEGACY_KEEP_COLS = ["RA", "DEC", "OBJID", "TYPE", "z_phot_mean"]
CHUNK_ROWS = 2_000_000
COSMO = Planck18

# ───────────────────────── geometry helpers ───────────────────────────────
def build_beam_metadata(outer_kpc: float):
    centres = [SkyCoord(r, d, frame="icrs") for r, d, _ in TARGETS]
    theta_max = [
        (outer_kpc * u.kpc / COSMO.angular_diameter_distance(z))
        .to(u.rad, equivalencies=u.dimensionless_angles()).value
        for *_, z in TARGETS
    ]
    ra0 = np.array([c.ra.rad  for c in centres])
    dec0 = np.array([c.dec.rad for c in centres])
    return centres, theta_max, ra0, dec0


def rect_mask(ra_rad, dec_rad, idx, ra0, dec0, theta_max):
    dra  = np.abs((ra_rad - ra0[idx] + np.pi) % (2*np.pi) - np.pi)
    ddec = np.abs(dec_rad - dec0[idx])
    return (dra <= theta_max[idx] / np.cos(dec0[idx])) & (ddec <= theta_max[idx])

# ───────────────────────── STRM utilities ─────────────────────────────────
def csv_to_parquet(csv_path: pathlib.Path, parquet_path: pathlib.Path):
    import pyarrow as pa
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    tot, t0 = 0, time.time()
    for chunk in pd.read_csv(
        csv_path, comment="#", header=None,
        usecols=STRM_COL_IDX, names=STRM_COLNAMES,
        dtype={"class": "category"}, chunksize=CHUNK_ROWS, low_memory=True,
    ):
        tot += len(chunk)
        tbl = pa.Table.from_pandas(chunk)
        writer = pa.parquet.ParquetWriter(
            parquet_path, tbl.schema, compression="zstd"
        ) if writer is None else writer
        writer.write_table(tbl)
    writer.close()
    print(f"[done] STRM Parquet {tot:,} rows in {(time.time()-t0)/60:.1f} min")


def iter_strm(path: pathlib.Path):
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(
            columns=STRM_COLNAMES, batch_size=CHUNK_ROWS
        ):
            yield batch.to_pandas()
    else:
        for chunk in pd.read_csv(
            path, comment="#", header=None,
            usecols=STRM_COL_IDX, names=STRM_COLNAMES,
            dtype={"class": "category"},
            chunksize=CHUNK_ROWS, low_memory=True,
        ):
            yield chunk

# ───────────────────────── Legacy DR8 iterator ────────────────────────────
def iter_legacy(path: pathlib.Path, chunk_size: int = 2_000_000):
    """
    Stream a large Legacy DR8/DR9 photo-z FITS file in DataFrame chunks.

    Parameters
    ----------
    path : Path
        .fits or .fits.fz file downloaded from the Legacy Surveys portal.
    chunk_size : int
        Number of rows to yield per chunk.
    """
    # Read the whole table header-only (mem-mapped to avoid RAM blow-up)
    tab = Table.read(path, memmap=True)       # <-- no 'columns=' kwarg

    # Keep just the four columns we care about
    tab = tab[LEGACY_KEEP_COLS]               # slice after reading

    nrows = len(tab)
    for start in range(0, nrows, chunk_size):
        df = tab[start : start + chunk_size].to_pandas()
        df.rename(columns={"RA": "ra", "DEC": "dec", "OBJID": "objid"},
                  inplace=True)
        yield df

# ───────────────────────── matcher (generic) ──────────────────────────────
def match(df: pd.DataFrame,
          ra_col: str, dec_col: str, z_col: str,
          centres, theta_max, ra0, dec0,
          outer_kpc: float, prefix: str,
          counts_run: dict[int, dict[str, int]],
          out_paths: dict[int, dict[str, pathlib.Path]]):
    # basic cleaning
    df[z_col] = pd.to_numeric(df[z_col], errors="coerce")
    df = df[(df[z_col].notna()) & (df[z_col] >= 0)]
    if df.empty:
        return

    ra_rad  = np.deg2rad(df[ra_col].to_numpy())
    dec_rad = np.deg2rad(df[dec_col].to_numpy())
    z_arr   = df[z_col].to_numpy()

    for idx, (_, _, z_host) in enumerate(TARGETS):
        mask_z = z_arr <= 1.5 * z_host
        if not mask_z.any():
            continue
        win = rect_mask(ra_rad, dec_rad, idx, ra0, dec0, theta_max) & mask_z
        if not win.any():
            continue

        sel = np.flatnonzero(win)
        coords = SkyCoord(ra=ra_rad[sel]*u.rad,
                          dec=dec_rad[sel]*u.rad, frame="icrs")
        theta = coords.separation(centres[idx]).radian
        d_a   = COSMO.angular_diameter_distance(z_arr[sel]).to(u.kpc)
        impact = theta * d_a
        close = impact <= outer_kpc * u.kpc
        if not close.any():
            continue

        sub = df.iloc[sel[close]].copy()
        sub["impact_kpc"] = impact[close].value

        outfile = out_paths[idx][prefix]
        header_needed = not outfile.exists()
        sub.to_csv(outfile, mode="a", header=header_needed, index=False)
        counts_run[idx][prefix] += len(sub)

# ───────────────────────── CLI & main ─────────────────────────────────────
def parse_cli():
    p = argparse.ArgumentParser(
        description="Cross-match FRB beams with STRM and/or Legacy DR8 photo-z.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--strm",   type=str, help="WISE-PS1-STRM catalogue")
    p.add_argument("--legacy", type=str, help="Legacy DR8/DR9 FITS with photo-z")
    p.add_argument("--outer-kpc", type=float, default=200.0,
                   help="Impact-parameter cut in kpc")
    p.add_argument("--make-parquet", action="store_true",
                   help="Convert STRM CSV→Parquet if needed")
    p.add_argument("--clean-output", action="store_true",
                   help="Delete existing per-beam CSVs before run")
    return p.parse_args()


def main():
    args = parse_cli()
    centres, theta_max, ra0, dec0 = build_beam_metadata(args.outer_kpc)

    # initialise per-beam output maps
    out_paths = {
        i: {
            "strm":   pathlib.Path(f"beam_{i+1:02d}_ip{int(args.outer_kpc):03d}kpc_strm.csv"),
            "legacy": pathlib.Path(f"beam_{i+1:02d}_ip{int(args.outer_kpc):03d}kpc_legacy.csv"),
        }
        for i in range(len(TARGETS))
    }
    counts_run = {i: {"strm": 0, "legacy": 0} for i in range(len(TARGETS))}

    if args.clean_output:
        for d in out_paths.values():
            for p in d.values():
                p.unlink(missing_ok=True)

    # ── STRM catalogue ────────────────────────────────────────────────────
    if args.strm:
        spath = pathlib.Path(args.strm)
        if args.make_parquet and spath.suffix != ".parquet":
            pq_path = spath.with_suffix(".parquet")
            if not pq_path.exists():
                print("[info] Converting STRM CSV → Parquet …")
                csv_to_parquet(spath, pq_path)
            spath = pq_path
        print(f"[info] Matching STRM catalogue  : {spath.name}")
        for chunk in iter_strm(spath):
            match(chunk, "raMean", "decMean", "z_phot0",
                  centres, theta_max, ra0, dec0,
                  args.outer_kpc, "strm",
                  counts_run, out_paths)

    # ── Legacy DR8 catalogue ──────────────────────────────────────────────
    if args.legacy:
        lpath = pathlib.Path(args.legacy)
        print(f"[info] Matching Legacy DR8 file: {lpath.name}")
        for chunk in iter_legacy(lpath):
            match(chunk, "ra", "dec", "z_phot_mean",
                  centres, theta_max, ra0, dec0,
                  args.outer_kpc, "legacy",
                  counts_run, out_paths)

    # ── summary CSV ───────────────────────────────────────────────────────
    rows = []
    for i, (ra, dec, z_host) in enumerate(TARGETS):
        rows.append((
            i+1, ra, dec, z_host,
            counts_run[i]["strm"], counts_run[i]["legacy"]
        ))
    pd.DataFrame(
        rows,
        columns=["beam#", "RA", "Dec", "z_host", "N_strm", "N_legacy"],
    ).to_csv("beam_summary_combined.csv", index=False)
    print("[done] Written beam_summary_combined.csv")

if __name__ == "__main__":
    main()
