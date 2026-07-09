#!/usr/bin/env python3
"""Aggregate per-sightline foreground replay outputs into revalidation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord


def _read_parts(root: Path, name: str) -> pd.DataFrame:
    parts = []
    for path in sorted(root.glob(f"*/{name}")):
        frame = pd.read_csv(path)
        frame["source_path"] = str(path)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _read_candidates(root: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(root.glob("*/*_galaxies.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "nickname", path.name.removesuffix("_galaxies.csv"))
        frame["source_path"] = str(path)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _candidate_diff(candidates: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matched_registry_indexes: set[int] = set()
    for candidate_index, candidate in candidates.iterrows():
        nickname = str(candidate.get("nickname", "")).lower()
        registry_subset = registry[registry["nickname"].astype(str).str.lower() == nickname].copy()
        row = {
            "status": "new_candidate",
            "candidate_index": candidate_index,
            "nickname": nickname,
            "candidate_catalog": candidate.get("catalog"),
            "candidate_ra_deg": candidate.get("ra"),
            "candidate_dec_deg": candidate.get("dec"),
            "candidate_z": candidate.get("z"),
            "candidate_impact_kpc": candidate.get("impact_kpc"),
            "registry_index": pd.NA,
            "registry_obj": pd.NA,
            "registry_best_z": pd.NA,
            "registry_final_verdict": pd.NA,
            "separation_arcsec": pd.NA,
        }
        if not registry_subset.empty:
            cand_coord = SkyCoord(float(candidate["ra"]) * u.deg, float(candidate["dec"]) * u.deg)
            reg_coord = SkyCoord(
                registry_subset["ra_deg"].to_numpy(dtype=float) * u.deg,
                registry_subset["dec_deg"].to_numpy(dtype=float) * u.deg,
            )
            sep = cand_coord.separation(reg_coord).arcsec
            nearest_pos = int(sep.argmin())
            nearest = registry_subset.iloc[nearest_pos]
            nearest_index = int(registry_subset.index[nearest_pos])
            if sep[nearest_pos] <= 10.0:
                matched_registry_indexes.add(nearest_index)
                row.update(
                    {
                        "status": "matched_registry",
                        "registry_index": nearest_index,
                        "registry_obj": nearest.get("obj"),
                        "registry_best_z": nearest.get("best_z"),
                        "registry_final_verdict": nearest.get("final_verdict"),
                        "separation_arcsec": float(sep[nearest_pos]),
                    }
                )
        rows.append(row)

    for registry_index, registry_row in registry.iterrows():
        if int(registry_index) in matched_registry_indexes:
            continue
        rows.append(
            {
                "status": "registry_not_in_replay",
                "candidate_index": pd.NA,
                "nickname": registry_row.get("nickname"),
                "candidate_catalog": pd.NA,
                "candidate_ra_deg": pd.NA,
                "candidate_dec_deg": pd.NA,
                "candidate_z": pd.NA,
                "candidate_impact_kpc": pd.NA,
                "registry_index": int(registry_index),
                "registry_obj": registry_row.get("obj"),
                "registry_best_z": registry_row.get("best_z"),
                "registry_final_verdict": registry_row.get("final_verdict"),
                "separation_arcsec": pd.NA,
            }
        )

    return pd.DataFrame(rows)


def aggregate(input_dir: Path, output_dir: Path, registry_csv: Path | None = None) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_parts(input_dir, "search_summary.csv")
    coverage = _read_parts(input_dir, "survey_coverage.csv")
    candidates = _read_candidates(input_dir)

    summary.to_csv(output_dir / "foreground_live_replay_search_summary.csv", index=False)
    coverage.to_csv(output_dir / "foreground_live_replay_survey_coverage.csv", index=False)
    candidates.to_csv(output_dir / "foreground_live_replay_candidates.csv", index=False)

    counts = {
        "sightlines": int(summary["name"].nunique()) if "name" in summary.columns else 0,
        "candidate_rows": int(len(candidates)),
        "coverage_rows": int(len(coverage)),
    }
    if "num_galaxies" in summary.columns:
        counts["summary_candidate_rows"] = int(summary["num_galaxies"].sum())

    if registry_csv is not None:
        registry = pd.read_csv(registry_csv)
        diff = _candidate_diff(candidates, registry)
        diff.to_csv(output_dir / "foreground_live_replay_registry_diff.csv", index=False)
        counts["registry_rows"] = int(len(registry))
        counts["matched_registry_rows"] = int((diff["status"] == "matched_registry").sum())
        counts["new_candidate_rows"] = int((diff["status"] == "new_candidate").sum())
        counts["registry_not_in_replay_rows"] = int((diff["status"] == "registry_not_in_replay").sum())

    (output_dir / "foreground_live_replay_summary.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path)
    args = parser.parse_args()

    counts = aggregate(args.input_dir, args.output_dir, args.registry_csv)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
