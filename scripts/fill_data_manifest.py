"""Fill sha256/bytes in data-manifest.csv from locally reachable files.

P0.1 step only: writes status=HASHED_LOCAL. The subsequent P2.2 arc byte
cross-check (which upgrades HASHED_LOCAL -> ARC_BYTE_MATCH) was a one-shot
`vcp` download + sha256 comparison documented in
`scattering/scat_analysis/builders_arc/ORIGIN.md`; it is not wrapped by this
script. The manifest's ARC_BYTE_MATCH state is pinned by
`tests/test_data_manifest.py`.
"""

import argparse
import csv
import hashlib
import pathlib
import sys

MANIFEST = pathlib.Path(__file__).parents[1] / "data-manifest.csv"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="e.g. ~/Data/Faber2026")
    args = ap.parse_args()
    root = pathlib.Path(args.data_root).expanduser()
    with MANIFEST.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    missing = []
    for r in rows:
        hits = list(root.rglob(r["filename"]))
        if not hits:
            missing.append(r["filename"])
            continue
        r["sha256"] = sha256(hits[0])
        r["bytes"] = str(hits[0].stat().st_size)
        r["status"] = "HASHED_LOCAL"
    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"filled {len(rows) - len(missing)}/{len(rows)}; still pending: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
