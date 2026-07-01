#!/usr/bin/env python3
"""Audit arc VOSpace vs iacobus/jakob-mbp for Phase 3 migration entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML required") from exc

REPO = Path(__file__).resolve().parents[2]
INVENTORY = REPO / "machine_inventory.yaml"
MANIFEST = REPO / "codetections_manifest.yaml"
DEFAULT_OUT = REPO / "reports" / "phase3_audit.json"

IACOBUS = "iacobus"
VLS = "vls"
VCAT = "vcat"
DATE_PREFIX = re.compile(r"^\w{3}\s+\d{1,2}(?:\s+\d{4}|\s+\d{2}:\d{2})\s+")
VLS_LINE = re.compile(r"^([drw-]+)\s+\S+\s+\S+\s+\S+\s+(\d+)\s+(.+)$")

# Phase 3 audits: arc path vs local comparison targets
AUDITS = [
    {
        "id": "arc_dsa_bursts",
        "arc_path": "arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/data/DSA_bursts",
        "local_host": None,
        "local_path": "/Users/jakobfaber/Data/Faber2026/dsa110/DSA_bursts",
        "iacobus_path": None,
        "deep": True,
    },
    {
        "id": "arc_chime_bursts",
        "arc_path": "arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/data/CHIME_bursts",
        "local_host": None,
        "local_path": None,
        "iacobus_path": "/Users/iacobus/Research/CHIME_DSA_Codetections/burst_npys",
        "notes": "CHIME-side .npy; iacobus burst_npys is CHIME+mixed namespace",
    },
    {
        "id": "arc_old_chime_dedupe",
        "arc_path": "arc:home/jfaber/baseband_morphologies/OLD_CHIME_DSA_Codetections",
        "local_host": None,
        "local_path": None,
        "iacobus_path": "/Users/iacobus/Research/CHIME_DSA_Codetections/archive/OLD_CHIME_DSA_Codetections",
        "sentinel_subdir": "archive",
    },
    {
        "id": "arc_flits_checkout",
        "arc_path": "arc:home/jfaber/dsa110-FLITS",
        "local_host": None,
        "local_path": str(REPO),
        "iacobus_path": None,
        "notes": "diff vs GitHub canonical jakob-mbp checkout",
    },
    {
        "id": "arc_codetection_flits_tree",
        "arc_path": "arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS",
        "local_host": None,
        "local_path": str(REPO),
        "iacobus_path": None,
        "notes": "legacy ~5G arc-side FLITS; diff vs GitHub only",
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


def parse_vls_line(line: str) -> tuple[int, str, bool] | None:
    m = VLS_LINE.match(line)
    if not m:
        return None
    perms, size_s, rest = m.group(1), m.group(2), m.group(3)
    dm = DATE_PREFIX.match(rest)
    if not dm:
        return None
    name = rest[dm.end() :]
    return int(size_s), name, perms.startswith("d")


def arc_list(uri: str) -> tuple[list[tuple[int, str, bool]], str | None]:
    proc = subprocess.run([VLS, "-l", uri], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()[0] if proc.stderr else "vls failed"
        return [], err
    entries: list[tuple[int, str, bool]] = []
    for line in proc.stdout.splitlines():
        parsed = parse_vls_line(line)
        if not parsed:
            continue
        size, name, is_dir = parsed
        if name.startswith("."):
            continue
        entries.append((size, name, is_dir))
    return entries, None


def arc_shallow_stats(uri: str) -> dict:
    """Sum vls -l entry sizes; dir lines report subtree bytes (no recursion)."""
    entries, err = arc_list(uri)
    if err:
        return {"exists": False, "files": 0, "bytes": 0, "error": err}
    files = sum(1 for _, _, is_dir in entries if not is_dir)
    total = sum(size for size, _, _ in entries)
    return {"exists": True, "files": files, "files_note": "top_level_only", "bytes": total}


def arc_deep_stats(uri: str, depth: int = 0, max_depth: int = 6) -> dict:
    entries, err = arc_list(uri)
    if err:
        return {"exists": False, "files": 0, "bytes": 0, "error": err}
    files, total = 0, 0
    skipped: list[str] = []
    for size, name, is_dir in entries:
        child = f"{uri.rstrip('/')}/{name}"
        if is_dir:
            if depth >= max_depth:
                skipped.append(name)
                total += size
                continue
            sub = arc_deep_stats(child, depth + 1, max_depth)
            if not sub["exists"]:
                skipped.append(name)
                total += size
                continue
            files += sub["files"]
            total += sub["bytes"]
        else:
            files += 1
            total += size
    out: dict = {"exists": True, "files": files, "bytes": total, "mode": "deep"}
    if skipped:
        out["skipped_dirs"] = skipped[:8]
    return out


def arc_stats(uri: str, deep: bool) -> dict:
    if deep:
        return arc_deep_stats(uri)
    return arc_shallow_stats(uri)


def local_stats(host: str | None, path: str) -> dict:
    if not path:
        return {"exists": False, "files": 0, "bytes": 0}
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


def arc_sha256_prefix(uri: str, nbytes: int) -> str | None:
    proc = subprocess.run(
        [VCAT, uri],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    h = hashlib.sha256()
    h.update(proc.stdout[:nbytes])
    return h.hexdigest()


def remote_sha256_prefix(host: str, path: str, nbytes: int) -> str | None:
    safe_path = path.replace("'", "'\\''")
    cmd = f"head -c {nbytes} '{safe_path}' | shasum -a 256 | awk '{{print $1}}'"
    try:
        return sh(host, cmd)
    except RuntimeError:
        return None


def sentinel_check(entry: dict, manifest: dict) -> dict | None:
    subdir = entry.get("sentinel_subdir")
    if not subdir:
        return None
    spec = manifest.get("subdirs", {}).get(subdir)
    if not spec:
        return None
    rel = spec["sentinel_path"]
    nbytes = spec.get("sentinel_sha256_prefix_bytes", 67108864)
    expected = spec.get("sentinel_sha256")
    iacobus_base = manifest.get("base", "")
    iacobus_file = f"{iacobus_base}/{rel}"
    arc_rel = rel.replace("archive/OLD_CHIME_DSA_Codetections/", "")
    arc_file = f"{entry['arc_path']}/{arc_rel}"
    iacobus_hash = remote_sha256_prefix(IACOBUS, iacobus_file, nbytes)
    arc_hash = arc_sha256_prefix(arc_file, nbytes)
    return {
        "sentinel_path": rel,
        "expected_sha256": expected,
        "iacobus_sha256": iacobus_hash,
        "arc_sha256": arc_hash,
        "iacobus_ok": iacobus_hash == expected if iacobus_hash else False,
        "arc_ok": arc_hash == expected if arc_hash else False,
        "arc_sentinel_exists": arc_hash is not None,
    }


def recommend(audit_id: str, arc: dict, local: dict, iacobus: dict) -> str:
    if not arc.get("exists"):
        return "skip"
    if audit_id in ("arc_flits_checkout", "arc_codetection_flits_tree"):
        return "diff_github"
    if audit_id == "arc_old_chime_dedupe":
        if iacobus.get("exists") and iacobus["bytes"] > arc.get("bytes", 0):
            return "iacobus_canonical"
        if arc.get("bytes", 0) > iacobus.get("bytes", 0):
            return "arc_superset_reconcile"
        return "dedupe_first"
    if audit_id == "arc_dsa_bursts":
        if local.get("exists") and local["files"] >= arc.get("files", 0):
            return "arc_authoritative_for_fits"
        return "sync_local_replica"
    if audit_id == "arc_chime_bursts":
        return "reconcile"
    return "audit"


def audit_entry(entry: dict, manifest: dict, check_sentinel: bool, deep: bool) -> dict:
    arc = arc_stats(entry["arc_path"], deep or bool(entry.get("deep")))
    local = local_stats(entry.get("local_host"), entry["local_path"]) if entry.get("local_path") else {
        "exists": False,
        "files": 0,
        "bytes": 0,
    }
    iacobus = local_stats(IACOBUS, entry["iacobus_path"]) if entry.get("iacobus_path") else {
        "exists": False,
        "files": 0,
        "bytes": 0,
    }
    result = {
        "id": entry["id"],
        "arc_path": entry["arc_path"],
        "local_path": entry.get("local_path"),
        "iacobus_path": entry.get("iacobus_path"),
        "notes": entry.get("notes"),
        "arc": arc,
        "local": local,
        "iacobus": iacobus,
        "recommendation": recommend(entry["id"], arc, local, iacobus),
    }
    if check_sentinel and entry.get("sentinel_subdir"):
        result["sentinel"] = sentinel_check(entry, manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--id", help="Single audit id")
    parser.add_argument("--json", type=Path, default=DEFAULT_OUT, help="Write JSON report")
    parser.add_argument("--sentinel", action="store_true", help="SHA-256 prefix sentinel checks")
    parser.add_argument("--deep", action="store_true", help="Recursive arc walk (slow; default shallow)")
    parser.add_argument("--stdout", action="store_true", help="Print summary")
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text()) if args.manifest.exists() else {}
    audits = AUDITS
    if args.id:
        audits = [a for a in AUDITS if a["id"] == args.id]
        if not audits:
            raise SystemExit(f"unknown id: {args.id}")

    results = [audit_entry(a, manifest, args.sentinel, args.deep) for a in audits]
    inv = yaml.safe_load(args.inventory.read_text())
    report = {
        "generated_by": "scripts/migration/audit_arc_delta.py",
        "generated_utc": inv.get("generated_utc"),
        "arc_cert_exp": inv.get("machines", {}).get("arc", {}).get("access", {}).get("cert_exp"),
        "quota_note": "Do not bulk-upload 218G iacobus tree; arc quota ~200G",
        "entries": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")

    if args.stdout:
        for r in results:
            a, loc, i = r["arc"], r["local"], r["iacobus"]
            print(
                f"{r['id']}: {r['recommendation']} "
                f"(arc {a.get('files',0)}f/{a.get('bytes',0)//1024}K "
                f"local {loc['files']}f/{loc['bytes']//1024}K "
                f"iacobus {i['files']}f/{i['bytes']//1024}K)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
