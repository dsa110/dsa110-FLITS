#!/usr/bin/env python3
"""
WISE‑PS1‑STRM 100 kpc host‑search
---------------------------------
* **Planck18** cosmology
* Rectangular θ pre‑cut for ≳10 × speed‑up
* Auto CSV → Parquet converter and streaming reader
* **O(1) memory**: matches stream straight to per‑beam CSVs
* `angular_diameter_distance` evaluated only on post‑cut galaxies for extra speed
* **Now stores**: `objID`, galaxy/star/QSO probabilities (`p_gal`, `p_star`,
  `p_qso`) along with RA/Dec, photo‑z, and impact parameter for every match.
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

# ───────────────────────────── CONFIGURATION ────────────────────────────────
TARGETS: list[tuple[str, str, float]] = [
    ("20h40m47.886s", "+72d52m56.378s", 0.0430+0.5), #310.1995250000, 72.8823272222
    ("08h58m52.92s",  "+73d29m27.0s",   0.4790+0.5), #134.7205000000, 73.4908333333
    ("21h12m10.760s", "+72d49m38.20s",  0.3005+0.5), #318.0448333333, 72.8272777778
    ("04h45m38.64s",  "+70d18m26.6s",   0.2505+0.5), #71.4110000000, 70.3073888889
    ("21h00m31.09s",  "+72d02m15.22s",  0.5100+0.5), #315.1295416667, 72.0375611111
    ("11h51m07.52s",  "+71d41m44.3s",   0.2710+0.5), #177.7813333333, 71.6956388889
    ("05h52m45.12s",  "+74d12m01.7s",   1.0000+0.5), #88.1880000000, 74.2004722222
    ("22h23m53.94s",  "+73d01m33.26s",  1.0000+0.5), #335.9747500000, 73.0259055600
    ("20h20m08.92s",  "+70d47m33.96s",  0.3024+0.5), #305.0371666667, 70.7927666667
    ("02h39m03.96s",  "+71d01m04.3s",   1.0000+0.5), #39.7665000000, 71.0178611111
    ("20h50m28.59s",  "+73d54m00.0s",   0.0740+0.5), #312.6191250000, 73.9000000000
    ("11h19m56.05s",  "+70d40m34.4s",   0.2870+0.5), #169.9835417, 70.67622222
]

# Column indices we need from the giant CSV
# 0=objID, 1=raMean, 3=decMean, 197–199=probabilities, 205=z_phot0
CSV_USECOLS  = [0, 1, 3, 197, 198, 199, 200, 205]
CSV_COLNAMES = [
    "objID", "raMean", "decMean", "class", "p_gal", "p_star", "p_qso", "z_phot0"
]

CHUNK_ROWS = 2_000_000        # rows per chunk
COSMO      = Planck18         # cosmology instance
max_kpc    = 500              # maximum impact parameter in kpc

# ───────────────────────────── Helper functions ─────────────────────────────

def build_beam_metadata():
    centres = [SkyCoord(ra, dec, frame="icrs") for ra, dec, _ in TARGETS]
    theta_max = [
        (max_kpc * u.kpc / COSMO.angular_diameter_distance(z))
        .to(u.rad, equivalencies=u.dimensionless_angles()).value
        for *_, z in TARGETS
    ]
    ra0  = np.array([c.ra.rad  for c in centres])
    dec0 = np.array([c.dec.rad for c in centres])
    return centres, theta_max, ra0, dec0


def rect_mask(ra_rad: np.ndarray, dec_rad: np.ndarray, idx: int,
              ra0: np.ndarray, dec0: np.ndarray, theta_max: list[float]):
    """Cheap rectangular window around beam centre."""
    dra  = np.abs((ra_rad - ra0[idx] + np.pi) % (2 * np.pi) - np.pi)
    ddec = np.abs(dec_rad - dec0[idx])
    return (dra * np.cos(dec0[idx]) <= theta_max[idx]) & (ddec <= theta_max[idx])

# ────────────────────────────────── Main ─────────────────────────────────────

def main(catalog_path: pathlib.Path, make_parquet: bool = False):
    t0 = time.time()

    # ── Choose CSV or Parquet ──
    parquet_path = catalog_path.with_suffix(".parquet")
    if make_parquet and not parquet_path.exists():
        print(f"[info] Converting {catalog_path.name} → Parquet (one‑time)…")
        csv_to_parquet(catalog_path, parquet_path)
    use_parquet = parquet_path.exists()

    # ── Chunk iterator ──
    if use_parquet:
        print(f"[info] Using Parquet file {parquet_path.name}")
        parq = pq.ParquetFile(parquet_path)
        chunk_iter = (
            b.to_pandas()
            for b in parq.iter_batches(columns=CSV_COLNAMES, batch_size=CHUNK_ROWS)
        )
    else:
        print(f"[info] Reading CSV in chunks of {CHUNK_ROWS:,} rows")
        chunk_iter = pd.read_csv(
            catalog_path,
            header=0,
            comment="#",
            usecols=CSV_USECOLS,
            names=CSV_COLNAMES,
            chunksize=CHUNK_ROWS,
            low_memory=True,
        )

    centres, theta_max, ra0, dec0 = build_beam_metadata()

    out_paths = {
        i: pathlib.Path(
            f"beam_{i+1:02d}_{ra}_{dec}_matches.csv".replace(" ", "")
        )
        for i, (ra, dec, _) in enumerate(TARGETS)
    }
    counts = defaultdict(int)

    processed = 0
    for chunk in chunk_iter:
        processed += len(chunk)
        # Clean up photo‑z column
        chunk["z_phot0"] = pd.to_numeric(chunk["z_phot0"], errors="coerce")
        chunk = chunk[(chunk["z_phot0"].notna()) & (chunk["z_phot0"] >= 0)]
        if chunk.empty:
            continue

        ra_rad  = np.deg2rad(chunk["raMean"].to_numpy())
        dec_rad = np.deg2rad(chunk["decMean"].to_numpy())
        z_arr   = chunk["z_phot0"].to_numpy()

        for idx, (_, _, z_lim) in enumerate(TARGETS):
            mask_z = z_arr <= z_lim
            if not np.any(mask_z):
                continue
            mask_window = rect_mask(ra_rad, dec_rad, idx, ra0, dec0, theta_max) & mask_z
            if not np.any(mask_window):
                continue

            sel_idx = np.flatnonzero(mask_window)
            coords = SkyCoord(
                ra=ra_rad[sel_idx] * u.rad, dec=dec_rad[sel_idx] * u.rad, frame="icrs"
            )
            theta = coords.separation(centres[idx]).radian  # pure floats
            d_a_sel = COSMO.angular_diameter_distance(z_arr[sel_idx]).to(u.kpc)
            phys = theta * d_a_sel  # Quantity[kpc]
            close = phys <= max_kpc * u.kpc
            if not np.any(close):
                continue

            final_idx = sel_idx[close]
            sub = chunk.iloc[final_idx].copy()
            sub["impact_kpc"] = phys[close].value

            header_needed = not out_paths[idx].exists()
            sub.to_csv(out_paths[idx], mode="a", header=header_needed, index=False)
            counts[idx] += len(sub)

    # ── Write summary ──
    summary = [
        (i + 1, ra, dec, zmax, counts[i]) for i, (ra, dec, zmax) in enumerate(TARGETS)
    ]
    pd.DataFrame(summary, columns=["beam#", "RA", "Dec", "z_max", "N_gal"]).to_csv(
        "wiseps1_beam_summary.csv", index=False
    )

    dt = time.time() - t0
    print(f"Processed {processed:,} rows in {dt:.1f} s")
    print("Per‑beam CSVs written where matches ≥ 1.")

# ─────────────────────── CSV → Parquet helper ───────────────────────────────

def csv_to_parquet(csv_path: pathlib.Path, parquet_path: pathlib.Path):
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa

    writer = None
    rows = 0
    t0 = time.time()
    for chunk in pd.read_csv(
        csv_path,
        header=0,
        comment="#",
        usecols=CSV_USECOLS,
        names=CSV_COLNAMES,
        chunksize=CHUNK_ROWS,
        low_memory=True,
    ):
        rows += len(chunk)
        if writer is None:
            writer = pq.ParquetWriter(
                parquet_path, pa.Table.from_pandas(chunk).schema, compression="zstd"
            )
        writer.write_table(pa.Table.from_pandas(chunk))
    if writer:
        writer.close()
    print(f"[done] Parquet written ({rows:,} rows in {(time.time() - t0)/60:.1f} min)")

# ────────────────────────────────── CLI ─────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=pathlib.Path,
        required=True,
        help="Path to wiseps1_cat_*.csv or .parquet",
    )
    parser.add_argument(
        "--make-parquet",
        action="store_true",
        help="Convert CSV → Parquet if needed before running search",
    )
    args = parser.parse_args()

    main(args.catalog, make_parquet=args.make_parquet)
