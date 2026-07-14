#!/usr/bin/env python
"""Stage immutable configs for the DM-locked joint-fit campaign."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    rows = list(csv.DictReader((here / "fit_roster.csv").open(newline="")))
    configs = args.output / "configs"
    for directory in (configs, args.output / "data" / "joint", args.output / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    staged = {}
    for row in rows:
        burst = row["burst"]
        if burst in staged:
            continue
        staged[burst] = {}
        for band in ("chime", "dsa"):
            source = args.source_configs / f"{burst}_{band}_run.yaml"
            payload = yaml.safe_load(source.read_text())
            payload["telcfg_path"] = str(args.repo / "scattering" / "configs" / "telescopes.yaml")
            payload["sampcfg_path"] = str(args.repo / "scattering" / "configs" / "sampler.yaml")
            if band == "dsa":
                payload["dm_init"] = float(row["adopted_dm"])
            target = configs / source.name
            target.write_text(yaml.safe_dump(payload, sort_keys=True))
            data_path = Path(payload["path"])
            if not data_path.exists():
                raise FileNotFoundError(data_path)
            staged[burst][band] = {
                "config": str(target),
                "data": str(data_path),
                "size_bytes": data_path.stat().st_size,
                "dm_init": float(payload.get("dm_init", 0.0)),
            }

    shutil.copy2(here / "fit_roster.csv", args.output / "fit_roster.csv")
    (args.output / "staging_provenance.json").write_text(
        json.dumps({"repo": str(args.repo.resolve()), "bursts": staged}, indent=2)
    )
    print(f"staged {len(staged)} bursts under {args.output}")


if __name__ == "__main__":
    main()
