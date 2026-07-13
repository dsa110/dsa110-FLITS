from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from dispersion.dm_power_analysis import CHIME_DT_S, DSA_DT_S

from .diagnostics import render_campaign
from .injection import run_full_injections, run_quick_injections
from .measurement import run_measurements
from .oracle import build_oracle_report
from .synthesis import synthesize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def preflight(config_path: Path) -> int:
    root = Path.cwd()
    config = yaml.safe_load(config_path.read_text())
    data_dir = Path(config["data_dir"]).expanduser().resolve()
    manifest_path = (root / config["manifest"]).resolve()
    output = (root / config["output_dir"]).resolve()
    rows = list(csv.DictReader(manifest_path.open()))
    records = []
    failures = []
    for row in rows:
        path = data_dir / row["filename"]
        record = {"burst": row["burst"], "telescope": row["telescope"], "path": str(path)}
        if not path.is_file():
            record.update(status="MISSING")
            failures.append(f"missing {path}")
        else:
            array = np.load(path, mmap_mode="r")
            expected_channels = 1024 if row["telescope"] == "chime" else 6144
            expected_time = 32000 if row["telescope"] == "chime" else 2500
            expected_shape = (expected_channels, expected_time)
            record.update(
                status="PRESENT" if tuple(array.shape) == expected_shape else "BAD_SHAPE",
                shape=list(array.shape),
                dtype=str(array.dtype),
                bytes=path.stat().st_size,
                sha256=_sha256(path),
                dm_reference=float(row["dm_pc_cm3"]),
                native_dt_s=CHIME_DT_S if row["telescope"] == "chime" else DSA_DT_S,
            )
            if tuple(array.shape) != expected_shape:
                failures.append(f"bad shape {path}: {array.shape} != {expected_shape}")
        records.append(record)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    run_manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "dirty": bool(dirty),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "input_count": len(records),
        "failure_count": len(failures),
    }
    _atomic_json(output / "inputs.json", records)
    _atomic_json(output / "run_manifest.json", run_manifest)
    snapshot = output / "config.snapshot.yaml"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(yaml.safe_dump(config, sort_keys=True))
    print(json.dumps({"ok": not failures and len(records) == 24, **run_manifest}, indent=2))
    if len(records) != 24:
        failures.append(f"manifest has {len(records)} rows, expected 24")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="flits-dm-phase")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--config", type=Path, required=True)
    oracle_parser = commands.add_parser("oracle")
    oracle_parser.add_argument("--config", type=Path, required=True)
    injection_parser = commands.add_parser("inject")
    injection_parser.add_argument("--config", type=Path, required=True)
    injection_parser.add_argument("--tier", choices=("quick", "full"), default="quick")
    measurement_parser = commands.add_parser("measure")
    measurement_parser.add_argument("--config", type=Path, required=True)
    measurement_parser.add_argument("--pilot", action="store_true")
    measurement_parser.add_argument("--all", action="store_true")
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--config", type=Path, required=True)
    render_parser.add_argument("--pilot", action="store_true")
    synthesize_parser = commands.add_parser("synthesize")
    synthesize_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        return preflight(args.config)
    if args.command == "oracle":
        config = yaml.safe_load(args.config.read_text())
        output = Path(config["output_dir"]) / "oracle" / "report.json"
        report = build_oracle_report(output)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1
    if args.command == "inject":
        config = yaml.safe_load(args.config.read_text())
        output = Path(config["output_dir"]) / "injections" / f"{args.tier}.json"
        report = (
            run_quick_injections(output)
            if args.tier == "quick"
            else run_full_injections(output)
        )
        print(json.dumps({"tier": args.tier, "pass": report["pass"], "summaries": report["summaries"]}, indent=2))
        return 0 if report["pass"] else 1
    if args.command == "measure":
        if args.pilot == args.all:
            raise SystemExit("choose exactly one of --pilot or --all")
        summary = run_measurements(args.config, pilot=args.pilot)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "render":
        paths = render_campaign(args.config, pilot=args.pilot)
        print(json.dumps({"rendered": len(paths), "paths": paths}, indent=2))
        return 0
    if args.command == "synthesize":
        report = synthesize(args.config)
        print(json.dumps(report, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
