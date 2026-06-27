#!/usr/bin/env python3
"""D2 pre-move audit: iacobus CHIME_canfar vs Research/archive basename inventory."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

IACOBUS = "iacobus"
SRC = "/Users/iacobus/Archives/CHIME_canfar"
ARCHIVE = "/Users/iacobus/Research/CHIME_DSA_Codetections/archive"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "reports" / "d2_chime_canfar_inventory.csv"

REMOTE = rf"""
import csv, json, os, sys
from pathlib import Path
src = Path({SRC!r})
archive = Path({ARCHIVE!r})

def walk(root, side):
    if not root.exists():
        return
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            yield {{
                "side": side,
                "relpath": str(p.relative_to(root)),
                "basename": fn,
                "bytes": st.st_size,
                "root": str(root),
            }}

rows = list(walk(src, "source"))
for child in sorted(archive.iterdir()) if archive.exists() else []:
    rows.extend(walk(child, "archive"))
src_b = {{r["basename"] for r in rows if r["side"] == "source"}}
arc_b = {{r["basename"] for r in rows if r["side"] == "archive"}}
summary = {{
    "source_files": sum(1 for r in rows if r["side"] == "source"),
    "source_bytes": sum(r["bytes"] for r in rows if r["side"] == "source"),
    "archive_files": sum(1 for r in rows if r["side"] == "archive"),
    "archive_bytes": sum(r["bytes"] for r in rows if r["side"] == "archive"),
    "basename_overlap_count": len(src_b & arc_b),
}}
print(json.dumps(summary))
w = csv.DictWriter(sys.stdout, fieldnames=["side", "relpath", "basename", "bytes", "root"])
w.writeheader()
w.writerows(rows)
"""


def sh(host: str, cmd: str) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{host}: {proc.stderr.strip()}")
    return proc.stdout


def audit() -> tuple[dict, list[dict]]:
    out = sh(IACOBUS, f"python3 -c {json.dumps(REMOTE)}")
    summary_line, _, csv_body = out.partition("\n")
    summary = json.loads(summary_line)
    rows = list(csv.DictReader(csv_body.splitlines()))
    for row in rows:
        row["bytes"] = int(row["bytes"])
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="Print summary JSON to stdout")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    summary, rows = audit()
    summary["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary["source_path"] = SRC
    summary["archive_path"] = ARCHIVE

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["side", "relpath", "basename", "bytes", "root"])
        writer.writeheader()
        writer.writerows(rows)

    if args.stdout:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"wrote {args.output} ({len(rows)} rows, overlap={summary['basename_overlap_count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
