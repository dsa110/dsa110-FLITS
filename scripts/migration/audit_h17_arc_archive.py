#!/usr/bin/env python3
"""D3 read-only audit: h17 arc_archive_2026-06 hash-map vs iacobus archive trees."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

H17 = "h17"
IACOBUS = "iacobus"
H17_ROOT = "/data/jfaber/arc_archive_2026-06"
COMPARE_ROOTS = [
    ("old_chime", "/Users/iacobus/Research/CHIME_DSA_Codetections/archive/OLD_CHIME_DSA_Codetections"),
    ("chime_canfar", "/Users/iacobus/Research/CHIME_DSA_Codetections/archive/chime_canfar"),
]
HASH_EXTS = {".pkl", ".npy"}

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "reports" / "d3_h17_arc_inventory.csv"
DEFAULT_JSON = REPO / "reports" / "d3_h17_arc_inventory.json"

REMOTE = r"""
import csv, json, os, subprocess, sys
from pathlib import Path

root = Path({root!r})
side = {side!r}
hash_exts = set({hash_exts!r})
rows = []
hash_paths = []
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        p = Path(dirpath) / fn
        try:
            st = p.stat()
        except OSError:
            continue
        ext = p.suffix.lower()
        row = {{
            "side": side,
            "relpath": str(p.relative_to(root)),
            "basename": fn,
            "bytes": st.st_size,
            "ext": ext,
        }}
        rows.append(row)
        if ext in hash_exts:
            hash_paths.append(str(p))

hashes = {{}}
if hash_paths:
    proc = subprocess.run(
        ["xargs", "-0", "-P", "4", "-n", "1", "sha256sum"],
        input="\0".join(hash_paths),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    for line in proc.stdout.splitlines():
        digest, path = line.split(None, 1)
        hashes[path] = digest

for row in rows:
    full = str(root / row["relpath"])
    if full in hashes:
        row["sha256"] = hashes[full]

summary = {{
    "side": side,
    "root": str(root),
    "files": len(rows),
    "bytes": sum(r["bytes"] for r in rows),
    "hashed_files": sum(1 for r in rows if r.get("sha256")),
    "hashed_bytes": sum(r["bytes"] for r in rows if r.get("sha256")),
}}
print(json.dumps(summary))
w = csv.DictWriter(sys.stdout, fieldnames=["side", "relpath", "basename", "bytes", "ext", "sha256"], extrasaction="ignore")
w.writeheader()
w.writerows(rows)
"""


def remote_python(host: str, script: str) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "python3"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{host}: python3\n{proc.stderr.strip()}")
    return proc.stdout


def remote_inventory(host: str, root: str, side: str) -> tuple[dict, list[dict]]:
    script = REMOTE.format(root=root, side=side, hash_exts=sorted(HASH_EXTS))
    out = remote_python(host, script)
    summary_line, _, body = out.partition("\n")
    summary = json.loads(summary_line)
    rows = list(csv.DictReader(body.splitlines()))
    for row in rows:
        row["bytes"] = int(row["bytes"])
    return summary, rows


def classify(h17_rows: list[dict], compare_rows: list[dict]) -> dict:
    hash_index: dict[str, list[dict]] = {}
    fp_index: dict[tuple[str, int], list[dict]] = {}
    basename_index: dict[str, list[dict]] = {}
    for row in compare_rows:
        if row.get("sha256"):
            hash_index.setdefault(row["sha256"], []).append(row)
        fp_index.setdefault((row["basename"], row["bytes"]), []).append(row)
        basename_index.setdefault(row["basename"], []).append(row)

    hash_dup_bytes = 0
    for row in h17_rows:
        b = row["bytes"]
        sha = row.get("sha256")
        if sha and sha in hash_index:
            row["match"] = "hash"
            row["match_side"] = hash_index[sha][0]["side"]
            hash_dup_bytes += b
        elif (row["basename"], b) in fp_index:
            row["match"] = "basename_size"
            row["match_side"] = fp_index[(row["basename"], b)][0]["side"]
        elif row["basename"] in basename_index:
            row["match"] = "basename_only"
            row["match_side"] = basename_index[row["basename"]][0]["side"]
        else:
            row["match"] = "unique"

    hashed = [r for r in h17_rows if r.get("sha256")]
    hashed_bytes = sum(r["bytes"] for r in hashed)
    hash_unique_bytes = sum(r["bytes"] for r in hashed if r.get("match") == "unique")
    dup_ratio = hash_dup_bytes / hashed_bytes if hashed_bytes else 0.0
    return {
        "h17_files": len(h17_rows),
        "h17_bytes": sum(r["bytes"] for r in h17_rows),
        "hashed_sample_files": len(hashed),
        "hashed_sample_bytes": hashed_bytes,
        "hash_duplicate_files": sum(1 for r in hashed if r.get("match") == "hash"),
        "hash_duplicate_bytes": hash_dup_bytes,
        "hash_unique_files": sum(1 for r in hashed if r.get("match") == "unique"),
        "hash_unique_bytes": hash_unique_bytes,
        "basename_size_duplicate_files": sum(1 for r in h17_rows if r.get("match") == "basename_size"),
        "basename_only_files": sum(1 for r in h17_rows if r.get("match") == "basename_only"),
        "unique_files": sum(1 for r in h17_rows if r.get("match") == "unique"),
        "unique_bytes": sum(r["bytes"] for r in h17_rows if r.get("match") == "unique"),
        "duplicate_pct_by_bytes": round(100 * dup_ratio, 2),
        "unique_pct_by_bytes": round(100 * (1 - dup_ratio), 2) if hashed_bytes else 0.0,
        "recommendation": "skip_copy" if hashed_bytes and dup_ratio >= 0.9 else "copy",
    }


def audit() -> tuple[dict, list[dict]]:
    h17_summary, h17_rows = remote_inventory(H17, H17_ROOT, "h17")
    compare_rows: list[dict] = []
    compare_summaries = []
    for label, path in COMPARE_ROOTS:
        summary, rows = remote_inventory(IACOBUS, path, label)
        compare_summaries.append(summary)
        compare_rows.extend(rows)
    stats = classify(h17_rows, compare_rows)
    report = {
        "generated_by": "scripts/migration/audit_h17_arc_archive.py",
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "h17_root": H17_ROOT,
        "compare_roots": [{"label": a, "path": b} for a, b in COMPARE_ROOTS],
        "hash_sample_note": "sha256 on all .pkl/.npy via xargs -P 4 sha256sum (245 h17 + 555 compare files)",
        "h17_summary": h17_summary,
        "compare_summaries": compare_summaries,
        **stats,
    }
    return report, h17_rows + compare_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="Print summary JSON")
    parser.add_argument("-o", "--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    report, rows = audit()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["side", "relpath", "basename", "bytes", "ext", "sha256", "match", "match_side"]
    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(report, indent=2) + "\n")

    if args.stdout:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(
            f"wrote {args.csv} ({len(rows)} rows); "
            f"hash unique {report['unique_pct_by_bytes']}% → {report['recommendation']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
