"""Build the checked-in Figure 3 input from the expanded foreground catalog."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.foreground.census_registry import load_census_duplicates

DATA_DIR = Path(__file__).resolve().parent / "data"
EXPANDED_CSV = DATA_DIR / "expanded_catalog_cross_references.csv"
OUTPUT_CSV = DATA_DIR / "sightline_halo_grid.csv"
BURSTS_CSV = DATA_DIR / "frozen_census" / "bursts.csv"


def build_frame() -> pd.DataFrame:
    catalog = pd.read_csv(EXPANDED_CSV, dtype={"object_id": str})
    catalog["nickname"] = catalog["nickname"].str.lower()
    bursts = pd.read_csv(BURSTS_CSV)
    duplicates = set(load_census_duplicates())
    records: list[dict] = []

    for _, burst in bursts.iterrows():
        nick = str(burst.nickname).lower()
        group = catalog[catalog.nickname == nick]
        host_z = pd.to_numeric(burst.z_spec, errors="coerce")
        common = {
            "nickname": nick,
            "frb_name": str(burst.tns),
            "frb_z": float(host_z) if np.isfinite(host_z) else None,
            "frb_ra_deg": float(burst.ra_deg),
            "frb_dec_deg": float(burst.dec_deg),
        }
        records.append(
            {
                **common,
                "row_kind": "host",
                "object_id": None,
                "system_type": None,
                "final_verdict": None,
                "system_z": None,
                "candidate_ra_deg": None,
                "candidate_dec_deg": None,
                "impact_kpc": None,
                "mass_msun": None,
                "radius_kpc": None,
                "radius_definition": None,
                "geometry_status": "host_roster",
                "budget_eligible": False,
            }
        )
        for _, row in group.iterrows():
            key = (nick, str(row.object_id))
            if row.final_verdict != "confirmed" or key in duplicates:
                continue
            z = pd.to_numeric(row.best_z, errors="coerce")
            mass = radius = np.nan
            radius_definition = None
            geometry_status = "pass"
            if (
                not np.isfinite(host_z)
                or not np.isfinite(z)
                or float(z) <= 0
                or float(z) >= float(host_z)
            ):
                geometry_status = "invalid_foreground_redshift"
            elif row.type == "halo":
                mass = pd.to_numeric(row.m200c_msun, errors="coerce")
                radius = pd.to_numeric(row.r200c_kpc, errors="coerce")
                radius_definition = "R200c"
            else:
                mass500 = pd.to_numeric(row.m500_1e14msun, errors="coerce")
                radius500 = pd.to_numeric(row.r500_mpc, errors="coerce")
                mass = float(mass500) * 1.0e14 if np.isfinite(mass500) else np.nan
                radius = float(radius500) * 1.0e3 if np.isfinite(radius500) else np.nan
                radius_definition = "R500c_catalog"
            if geometry_status == "pass" and (
                not np.isfinite(mass) or not np.isfinite(radius) or float(mass) <= 0 or float(radius) <= 0
            ):
                geometry_status = "missing_sourced_geometry"
            records.append(
                {
                    **common,
                    "row_kind": "system",
                    "object_id": str(row.object_id),
                    "system_type": str(row.type),
                    "final_verdict": str(row.final_verdict),
                    "system_z": float(z) if np.isfinite(z) else None,
                    "candidate_ra_deg": float(row.ra_deg),
                    "candidate_dec_deg": float(row.dec_deg),
                    "impact_kpc": float(row.impact_kpc) if pd.notna(row.impact_kpc) else None,
                    "mass_msun": float(mass) if np.isfinite(mass) else None,
                    "radius_kpc": float(radius) if np.isfinite(radius) else None,
                    "radius_definition": radius_definition,
                    "geometry_status": geometry_status,
                    "budget_eligible": bool(row.budget_eligible),
                }
            )
    return pd.DataFrame(records)


def build(path: Path = OUTPUT_CSV) -> Path:
    frame = build_frame()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
