#!/usr/bin/env python3
"""Audit h23 vs iacobus delta for Phase 2 migration_map entries."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML required") from exc

REPO = Path(__file__).resolve().parents[2]
INVENTORY = REPO / "machine_inventory.yaml"
DEFAULT_OUT = REPO / "reports" / "phase2_audit.json"

H23 = "h23"
IACOBUS = "iacobus"


def sh(host: str | None, cmd: str) -> str:
    if host:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, cmd],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{host or 'local'}: {cmd}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def remote_stats(host: str, path: str) -> dict:
    cmd = (
        f'P="{path}"; '
        'if [ ! -e "$P" ]; then echo "MISSING"; exit 0; fi; '
        'find "$P" -type f 2>/dev/null | wc -l | tr -d " "; '
        'echo ---; '
        'du -sk "$P" 2>/dev/null | awk "{print \\$1}"'
    )
    out = sh(host, cmd)
    if out == "MISSING":
        return {"exists": False, "files": 0, "kib": 0}
    files_s, kib_s = out.split("\n---\n", 1)
    return {"exists": True, "files": int(files_s), "kib": int(kib_s or 0)}


def recommend(source: dict, target: dict, entry_id: str) -> str:
    if not source["exists"]:
        return "skip"
    if not target["exists"]:
        return "rsync_delta"
    if source["files"] == 0:
        return "skip"
    if target["kib"] >= source["kib"] and target["files"] >= source["files"]:
        if entry_id in ("h23_old_chime_archive",):
            return "dedupe_first"
        return "skip"
    if entry_id == "h23_chime_bursts":
        return "reconcile"
    if entry_id == "h23_old_chime_archive":
        return "dedupe_first"
    return "rsync_delta"


def audit_entry(entry: dict) -> dict:
    src = entry["source_path"]
    tgt = entry.get("target_path") or ""
    if entry["id"] == "h23_dm_budget" and not tgt.rstrip("/").endswith("h23_dm_budget"):
        tgt = f"{tgt.rstrip('/')}/h23_dm_budget"
    source = remote_stats(H23, src)
    target = remote_stats(IACOBUS, tgt) if tgt else {"exists": False, "files": 0, "kib": 0}
    rec = recommend(source, target, entry["id"])
    return {
        "id": entry["id"],
        "source_path": src,
        "target_path": tgt,
        "source": source,
        "target": target,
        "recommendation": rec,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--id", help="Single migration_map id")
    parser.add_argument("--json", type=Path, default=DEFAULT_OUT, help="Write JSON report")
    parser.add_argument("--stdout", action="store_true", help="Also print summary")
    args = parser.parse_args()

    inv = yaml.safe_load(args.inventory.read_text())
    entries = [
        e
        for e in inv.get("migration_map", [])
        if e.get("phase") == 2 and e.get("source_host") == "h23"
    ]
    if args.id:
        entries = [e for e in entries if e["id"] == args.id]
        if not entries:
            raise SystemExit(f"unknown id: {args.id}")

    results = [audit_entry(e) for e in entries]
    report = {
        "generated_by": "scripts/migration/audit_h23_delta.py",
        "entries": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")

    if args.stdout:
        for r in results:
            s, t = r["source"], r["target"]
            print(
                f"{r['id']}: {r['recommendation']} "
                f"(h23 {s['files']}f/{s['kib']}K -> iacobus {t['files']}f/{t['kib']}K)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
