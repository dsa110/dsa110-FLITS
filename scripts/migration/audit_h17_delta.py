#!/usr/bin/env python3
"""Audit h17 compute/staging paths vs iacobus for Phase 4 migration entries."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML required") from exc

REPO = Path(__file__).resolve().parents[2]
INVENTORY = REPO / "machine_inventory.yaml"
DEFAULT_OUT = REPO / "reports" / "phase4_audit.json"

H17 = "h17"
IACOBUS = "iacobus"

# Phase 4 audits: h17 paths + optional iacobus compare targets
AUDITS = [
    {
        "id": "h17_compute_workspace",
        "h17_path": "/data/research/astrophysics/frbs/chime-dsa-codetections",
        "iacobus_path": None,
        "notes": "CHIME docker workspace; baseband staging; keep on h17",
        "children": True,
    },
    {
        "id": "h17_upchan_products",
        "h17_path": "/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections",
        "iacobus_path": None,
        "notes": "5-target upchan .npy; promote to arc/iacobus when stable",
    },
    {
        "id": "h17_arc_archive_copy",
        "h17_path": "/data/research/astrophysics/frbs/chime-dsa-codetections/archive/arc_trash_2026-06",
        "iacobus_path": "/Users/iacobus/Research/CHIME_DSA_Codetections/archive/arc_trash_2026-06",
        "notes": "consolidated from /data/jfaber/ 2026-06-27; optional copy to iacobus after dedupe",
        "children": True,
    },
    {
        "id": "h17_ubuntu_stub",
        "h17_path": "/data/ubuntu/chime-dsa-codetections",
        "iacobus_path": None,
        "notes": "empty stub; remove after compute workspace canonical",
    },
    {
        "id": "h17_chime_singlebeam_empty",
        "h17_path": "/data/jfaber/chime_singlebeam",
        "iacobus_path": None,
        "notes": "empty dir; baseband staged under compute workspace",
    },
    {
        "id": "iacobus_chime_canfar_archive",
        "h17_path": None,
        "iacobus_path": "/Users/iacobus/Archives/CHIME_canfar",
        "iacobus_compare_path": "/Users/iacobus/Research/CHIME_DSA_Codetections/archive",
        "notes": "iacobus-only dedupe; migration_map phase 4",
    },
]


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
    quoted = path.replace("'", "'\\''")
    cmd = (
        f"P='{quoted}'; "
        'if [ ! -e "$P" ]; then echo "MISSING"; exit 0; fi; '
        'find "$P" -type f 2>/dev/null | wc -l | tr -d " "; '
        'echo ---; '
        'du -sk "$P" 2>/dev/null | awk "{print \\$1}"'
    )
    out = sh(host, cmd)
    if out == "MISSING":
        return {"exists": False, "files": 0, "bytes": 0}
    files_s, kib_s = out.split("\n---\n", 1)
    return {"exists": True, "files": int(files_s), "bytes": int(kib_s or 0) * 1024}


def remote_children(host: str, path: str, limit: int = 12) -> list[dict]:
    quoted = path.replace("'", "'\\''")
    cmd = (
        f"P='{quoted}'; "
        'if [ ! -d "$P" ]; then exit 0; fi; '
        f'du -sk "$P"/* 2>/dev/null | sort -nr | head -{limit} | '
        'awk \'{printf "%s\\t%s\\n", $1, $2}\''
    )
    try:
        out = sh(host, cmd)
    except RuntimeError:
        return []
    children: list[dict] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        kib_s, child_path = line.split("\t", 1)
        children.append({"path": child_path, "bytes": int(kib_s) * 1024})
    return children


def recommend(audit_id: str, h17: dict, iacobus: dict, compare: dict | None) -> str:
    if audit_id == "h17_compute_workspace":
        return "keep"
    if audit_id == "h17_upchan_products":
        return "keep" if h17.get("exists") else "missing"
    if audit_id == "h17_arc_archive_copy":
        if not h17.get("exists"):
            return "skip"
        if not iacobus.get("exists"):
            return "dedupe_then_copy"
        if iacobus["bytes"] >= h17["bytes"]:
            return "dedupe_first"
        return "rsync_delta"
    if audit_id == "h17_ubuntu_stub":
        if not h17.get("exists"):
            return "skip"
        if h17.get("files", 0) == 0:
            return "remove_stub"
        return "reconcile"
    if audit_id == "h17_chime_singlebeam_empty":
        if h17.get("exists") and h17.get("files", 0) == 0:
            return "remove_stub"
        return "keep"
    if audit_id == "iacobus_chime_canfar_archive":
        if not iacobus.get("exists"):
            return "skip"
        if compare and compare.get("exists") and compare["bytes"] > 0:
            return "dedupe_into_research"
        return "audit"
    return "audit"


def audit_entry(entry: dict) -> dict:
    h17 = (
        remote_stats(H17, entry["h17_path"])
        if entry.get("h17_path")
        else {"exists": False, "files": 0, "bytes": 0}
    )
    iacobus = (
        remote_stats(IACOBUS, entry["iacobus_path"])
        if entry.get("iacobus_path")
        else {"exists": False, "files": 0, "bytes": 0}
    )
    compare = (
        remote_stats(IACOBUS, entry["iacobus_compare_path"])
        if entry.get("iacobus_compare_path")
        else None
    )
    result = {
        "id": entry["id"],
        "h17_path": entry.get("h17_path"),
        "iacobus_path": entry.get("iacobus_path"),
        "iacobus_compare_path": entry.get("iacobus_compare_path"),
        "notes": entry.get("notes"),
        "h17": h17,
        "iacobus": iacobus,
        "recommendation": recommend(entry["id"], h17, iacobus, compare),
    }
    if compare is not None:
        result["iacobus_compare"] = compare
    if entry.get("children") and h17.get("exists"):
        result["h17_children"] = remote_children(H17, entry["h17_path"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--id", help="Single audit id")
    parser.add_argument("--json", type=Path, default=DEFAULT_OUT, help="Write JSON report")
    parser.add_argument("--stdout", action="store_true", help="Print summary")
    args = parser.parse_args()

    audits = AUDITS
    if args.id:
        audits = [a for a in AUDITS if a["id"] == args.id]
        if not audits:
            raise SystemExit(f"unknown id: {args.id}")

    inv = yaml.safe_load(args.inventory.read_text())
    results = [audit_entry(a) for a in audits]
    report = {
        "generated_by": "scripts/migration/audit_h17_delta.py",
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "h17_host": inv.get("machines", {}).get("h17", {}).get("hostname"),
        "h17_tailscale": inv.get("machines", {}).get("h17", {}).get("access", {}).get("tailscale"),
        "phase2_parallel_note": "Do not touch h23/iacobus rsync while Phase 2 runs elsewhere",
        "entries": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")

    if args.stdout:
        for r in results:
            h, i = r["h17"], r["iacobus"]
            print(
                f"{r['id']}: {r['recommendation']} "
                f"(h17 {h['files']}f/{h['bytes']//1024//1024}M "
                f"iacobus {i['files']}f/{i['bytes']//1024//1024}M)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
