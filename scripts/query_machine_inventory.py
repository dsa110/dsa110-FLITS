#!/usr/bin/env python3
"""Query machine_inventory.yaml for paths, hosts, kinds, and migration state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML required: conda run -n py312 python -m pip install pyyaml") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPO_ROOT / "machine_inventory.yaml"

# h23 chime_dsa_codetections subtrees that must appear in migration_map source_path
H23_CODETECTION_REQUIRED = [
    "/media/ubuntu/ssd/jfaber/chime_dsa_codetections/bursts",
    "/media/ubuntu/ssd/jfaber/chime_dsa_codetections/data/stokes_I_npys",
    "/media/ubuntu/ssd/jfaber/chime_dsa_codetections/dm_budget",
    "/media/ubuntu/ssd/jfaber/chime_dsa_codetections/scattering",
    "/media/ubuntu/ssd/jfaber/chime_dsa_codetections/dm",
    "/media/ubuntu/ssd/jfaber/chime_dsa_codetections/localizations",
]


def load_inventory(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def walk_locations(node: dict, machine: str, hits: list[dict]) -> None:
    loc = dict(node)
    loc.setdefault("machine", machine)
    hits.append(loc)
    for child in loc.get("children") or []:
        if isinstance(child, str):
            hits.append({"machine": machine, "path": child})
        elif isinstance(child, dict):
            walk_locations(child, machine, hits)


def flatten(inventory: dict) -> list[dict]:
    rows: list[dict] = []
    for machine, meta in (inventory.get("machines") or {}).items():
        for loc in meta.get("locations") or []:
            walk_locations(loc, machine, rows)
    return rows


def migration_map_rows(inventory: dict) -> list[dict]:
    rows: list[dict] = []
    for entry in inventory.get("migration_map") or []:
        row = dict(entry)
        row.setdefault("machine", row.get("source_host"))
        rows.append(row)
    return rows


def filter_location_rows(inv: dict, args: argparse.Namespace) -> list[dict]:
    rows = flatten(inv)
    if args.machine:
        rows = [r for r in rows if r.get("machine") == args.machine]
    if args.kind:
        rows = [r for r in rows if r.get("kind") == args.kind]
    if args.path_contains:
        needle = args.path_contains.lower()
        rows = [r for r in rows if needle in str(r.get("path", "")).lower()]
    if args.migration_status and args.migration_status not in MIGRATION_MAP_STATUSES:
        machines = inv.get("machines") or {}
        allowed = {
            name
            for name, meta in machines.items()
            if meta.get("migration_status") == args.migration_status
        }
        rows = [r for r in rows if r.get("machine") in allowed]
    if args.migration_target:
        rows = [
            r
            for r in rows
            if r.get("migration_target") == args.migration_target
            or r.get("target_host") == args.migration_target
        ]
    return rows


def filter_map_rows(inv: dict, args: argparse.Namespace) -> list[dict]:
    rows = migration_map_rows(inv)
    if args.machine:
        rows = [
            r
            for r in rows
            if r.get("source_host") == args.machine or r.get("target_host") == args.machine
        ]
    if args.migration_status:
        rows = [r for r in rows if r.get("status") == args.migration_status]
    if args.migration_target:
        rows = [
            r
            for r in rows
            if r.get("target_host") == args.migration_target
            or r.get("migration_target") == args.migration_target
        ]
    if args.path_contains:
        needle = args.path_contains.lower()
        rows = [
            r
            for r in rows
            if needle in str(r.get("source_path", "")).lower()
            or needle in str(r.get("target_path", "")).lower()
        ]
    return rows


def check_retired_coverage(inv: dict) -> list[str]:
    """Return list of uncovered required paths (empty == pass)."""
    source_paths = [e.get("source_path", "") for e in inv.get("migration_map") or []]
    gaps: list[str] = []
    for required in H23_CODETECTION_REQUIRED:
        if not any(required in sp or sp == required for sp in source_paths):
            gaps.append(required)
    retired = set(inv.get("migration", {}).get("retired_hosts") or [])
    for entry in inv.get("migration_map") or []:
        if entry.get("source_host") in retired and entry.get("status") == "pending":
            if not entry.get("source_path"):
                gaps.append(f"migration_map:{entry.get('id')}:missing source_path")
    return gaps


MIGRATION_MAP_STATUSES = frozenset({"pending", "out_of_scope"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--machine", help="Filter by machine id (iacobus, h17, arc, ...)")
    parser.add_argument("--kind", help="Filter by location kind")
    parser.add_argument("--path-contains", help="Substring match on path")
    parser.add_argument(
        "--migration-status",
        help="Filter migration_map by status (pending, out_of_scope, ...) or machines by migration_status",
    )
    parser.add_argument(
        "--migration-target",
        help="Filter by migration_target / target_host (locations; add --migration-map for map-only)",
    )
    parser.add_argument(
        "--migration-map",
        action="store_true",
        help="Query migration_map entries only (default with --migration-status pending)",
    )
    parser.add_argument(
        "--check-retired-coverage",
        action="store_true",
        help="Exit 1 if required retired-host codetection paths lack migration_map entries",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    inv = load_inventory(args.inventory)

    if args.check_retired_coverage:
        gaps = check_retired_coverage(inv)
        if args.json:
            json.dump({"gaps": gaps, "ok": not gaps}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        elif gaps:
            for g in gaps:
                print(f"UNMAPPED: {g}")
        else:
            print("retired-coverage: OK")
        return 1 if gaps else 0

    use_migration_map = args.migration_map or (
        args.migration_status in MIGRATION_MAP_STATUSES
    )

    if use_migration_map:
        rows = filter_map_rows(inv, args)
    elif args.migration_target:
        loc_rows = filter_location_rows(inv, args)
        map_rows = filter_map_rows(inv, args)
        rows = loc_rows + map_rows
    else:
        rows = filter_location_rows(inv, args)

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    for row in rows:
        if row.get("source_path") is not None and row.get("id"):
            status = row.get("status", "")
            action = row.get("action", "")
            src = row.get("source_path", "?")
            tgt = row.get("target_host") or row.get("target_path") or "-"
            print(f"{row.get('id')}: {status} {action} {src} -> {tgt}")
            continue

        path = row.get("path", "?")
        kind = row.get("kind", "")
        size = row.get("size_human", "")
        extra = f" [{kind}]" if kind else ""
        size_s = f" {size}" if size else ""
        mig = row.get("migration_action")
        mig_s = f" ({mig}->{row.get('migration_target')})" if mig else ""
        prefix = row.get("machine", "")
        if str(path).startswith("arc:"):
            label = str(path)
        else:
            label = f"{prefix}:{path}"
        print(f"{label}{extra}{size_s}{mig_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
