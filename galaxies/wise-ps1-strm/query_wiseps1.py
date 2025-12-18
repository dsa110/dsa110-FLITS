#!/usr/bin/env python3
"""
Search the WISE‑PS1‑STRM galaxy catalogue for hosts lying ≤ 100 kpc (rest‑frame)
from a set of pencil‑beam sight‑lines.

Key features
------------
* **Planck18 cosmology** (built into *astropy*).
* **Rectangular θ‑pre‑cut** for a 10–100 × speed‑up.
* **CSV → Parquet one‑time converter**.
* **Robust Parquet reader** – streams row‑groups with *pyarrow* so it works with
  any pandas version (no `iterator=` kwarg required).

Typical run‑time on a laptop (M1 Max, 64 GB RAM) for the full 30 GB northern cap:
≈ 50 s wall‑clock.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

# Optional: pyarrow for Parquet support (faster than CSV)
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pq = None
    HAS_PYARROW = False

# Add parent directory to path for shared config import
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "v1_0"))
from config import (
    TARGETS_TUPLE as TARGETS,
    angular_diameter_distance_fast,
    R_PHYS_KPC,
)

# ───────────────────────────── CONFIGURATION ──────────────────────────────
CSV_USECOLS = [1, 3, 205]            # 0-based indices: raMean, decMean, z_phot0
CSV_COLNAMES = ["raMean", "decMean", "z_phot0"]
CHUNK_ROWS = 2_000_000               # rows to process at once

# ────────────────────────────── Helper functions ──────────────────────────────

def build_beam_metadata():
    """Build beam metadata using fast cosmology lookup."""
    centres = [SkyCoord(ra, dec, frame="icrs") for ra, dec, _ in TARGETS]
    # Use fast lookup: theta = impact_kpc / D_A(z)
    z_arr = np.array([z for _, _, z in TARGETS])
    d_a = angular_diameter_distance_fast(z_arr)  # Mpc
    theta_max = (R_PHYS_KPC / 1000.0) / d_a  # radians
    ra0 = np.array([c.ra.rad for c in centres])
    dec0 = np.array([c.dec.rad for c in centres])
    return centres, list(theta_max), ra0, dec0


def rect_mask(ra_rad, dec_rad, idx, ra0, dec0, theta_max):
    dra = np.abs((ra_rad - ra0[idx] + np.pi) % (2 * np.pi) - np.pi)
    ddec = np.abs(dec_rad - dec0[idx])
    return (dra * np.cos(dec0[idx]) <= theta_max[idx]) & (ddec <= theta_max[idx])

# ─────────────────────────────────── Main ─────────────────────────────────────

def main(catalog_path: pathlib.Path, make_parquet: bool = False):
    t0 = time.time()

    parquet_path = catalog_path.with_suffix(".parquet")
    if make_parquet and not parquet_path.exists():
        if not HAS_PYARROW:
            print("[error] pyarrow is required for Parquet conversion. Install with: pip install pyarrow")
            return
        print(f"[info] Converting {catalog_path.name} -> Parquet (one-time)...")
        to_parquet(catalog_path, parquet_path)
    use_parquet = HAS_PYARROW and parquet_path.exists()

    if use_parquet:
        print(f"[info] Using Parquet file {parquet_path.name}")
        parq = pq.ParquetFile(parquet_path)
        def chunk_iter():
            for batch in parq.iter_batches(columns=CSV_COLNAMES,
                                           batch_size=CHUNK_ROWS):
                yield batch.to_pandas()
    else:
        print(f"[info] Reading CSV in chunks of {CHUNK_ROWS:,} rows")
        def chunk_iter():
            reader = pd.read_csv(
                catalog_path, header=0, comment="#", usecols=CSV_USECOLS,
                names=CSV_COLNAMES, chunksize=CHUNK_ROWS, low_memory=True,
            )
            yield from reader

    centres, theta_max, ra0, dec0 = build_beam_metadata()
    matches = {i: [] for i in range(len(TARGETS))}
    total_rows = 0

    for chunk in chunk_iter():
        total_rows += len(chunk)
        chunk["z_phot0"] = pd.to_numeric(chunk["z_phot0"], errors="coerce")
        chunk.dropna(subset=["z_phot0"], inplace=True)
        if chunk.empty:
            continue

        ra_rad = np.deg2rad(chunk["raMean"].values)
        dec_rad = np.deg2rad(chunk["decMean"].values)
        z_arr = chunk["z_phot0"].values
        d_a = COSMO.angular_diameter_distance(z_arr)  # Quantity[Mpc]

        for idx, (_, _, z_lim) in enumerate(TARGETS):
            zmask = z_arr <= z_lim
            if not np.any(zmask):
                continue
            pre = rect_mask(ra_rad, dec_rad, idx, ra0, dec0, theta_max)
            pre &= zmask
            if not np.any(pre):
                continue
            coords = SkyCoord(ra=ra_rad[pre] * u.rad, dec=dec_rad[pre] * u.rad,
                              frame="icrs")
            sep = coords.separation(centres[idx])
            phys = (sep.to(u.rad) * d_a[pre]).to(u.kpc)
            fin = phys <= 100 * u.kpc
            if not np.any(fin):
                continue
            sub = chunk.loc[pre].iloc[fin].copy()
            sub["impact_kpc"] = phys[fin].value
            matches[idx].append(sub)

    write_results(matches)
    print(f"Processed {total_rows:,} rows in {time.time() - t0:.1f} s")

# ───────────────────────────── I/O helpers ────────────────────────────────────

def to_parquet(csv_path: pathlib.Path, parquet_path: pathlib.Path):
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0; t0 = time.time()
    for chunk in pd.read_csv(
        csv_path, header=0, comment="#", usecols=CSV_USECOLS,
        names=CSV_COLNAMES, chunksize=CHUNK_ROWS, low_memory=True,
    ):
        rows += len(chunk)
        if writer is None:
            import pyarrow as pa
            writer = pq.ParquetWriter(parquet_path,
                                      pa.Table.from_pandas(chunk).schema,
                                      compression="zstd")
        writer.write_table(pa.Table.from_pandas(chunk))
    if writer:  # close the file
        writer.close()
    print(f"[done] Parquet written ({rows:,} rows in {(time.time()-t0)/60:.1f} min)")


def write_results(matches: dict[int, list[pd.DataFrame]]):
    summary = []
    for idx, (ra, dec, zmax) in enumerate(TARGETS):
        if matches[idx]:
            df = pd.concat(matches[idx], ignore_index=True)
            fname = f"beam_{idx+1:02d}_{ra}_{dec}_matches.csv".replace(" ", "")
            df.to_csv(fname, index=False)
            summary.append((idx+1, ra, dec, zmax, len(df)))
        else:
            summary.append((idx+1, ra, dec, zmax, 0))
    (pd.DataFrame(summary, columns=["beam#", "RA", "Dec", "z_max", "N_gal"])
      .to_csv("wiseps1_beam_summary.csv", index=False))
    print("Per‑beam CSVs written where matches ≥ 1.")

# ─────────────────────────────────── CLI ──────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=pathlib.Path, required=True,
                   help="Path to wiseps1_cat_*.csv or .parquet")
    p.add_argument("--make-parquet", action="store_true",
                   help="Convert CSV → Parquet the first time, then exit")
    args = p.parse_args()
    main(args.catalog, make_parquet=args.make_parquet)
