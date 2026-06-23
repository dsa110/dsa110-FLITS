#!/usr/bin/env python
"""Build dsa_pointing.csv (burst, pointing_dec_deg, source) from the DSA detection-log
compilation of primary-beam pointings (dsa_primary_beam_pointings.csv, "Dec ibeam" column).

DSA-110 is a transit array, so the primary-beam pointing Dec is ~constant (~71.6 deg here); the
beam offset for each burst is |Dec_src - Dec_pointing| (analysis/flux_cal.dsa_beam_offset). Only
the 12 nicknames in configs/bursts.yaml are emitted; the source CSV also lists extra/repeater
events (gertrude, pingu, FRB20220912A2) that are out of sample.
"""

import csv
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = HERE / "dsa_primary_beam_pointings.csv"
OUT = HERE / "dsa_pointing.csv"
ALIAS = {"johndoe (ii)": "johndoeii"}  # CSV label -> bursts.yaml nickname


def main() -> None:
    sample = set(yaml.safe_load((REPO / "configs" / "bursts.yaml").read_text())["bursts"])
    rows = []
    for r in csv.DictReader(SRC.open()):
        name = (r.get("Names") or "").strip().lower()
        nick = ALIAS.get(name, name)
        dec = (r.get("Dec ibeam") or "").strip()
        if nick in sample and dec:
            rows.append((nick, float(dec)))
    rows.sort()
    missing = sample - {n for n, _ in rows}
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["burst", "pointing_dec_deg", "source"])
        for nick, dec in rows:
            w.writerow(
                [nick, dec, "DSA detection-log compilation (dsa_primary_beam_pointings.csv)"]
            )
    print(f"wrote {OUT} with {len(rows)} bursts; missing from source: {sorted(missing) or 'none'}")


if __name__ == "__main__":
    main()
