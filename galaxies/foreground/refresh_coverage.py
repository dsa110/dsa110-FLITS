#!/usr/bin/env python
"""Reclassify the burst x survey coverage table with exact-MOC footprints.

The HPCC repro run (scratch/repro-foreground-search-hpcc/survey_coverage.csv)
recorded raw/foreground counts per burst x survey, but its in_footprint column
came from the declination-only rules, which return True for the entire
+70..+74 deg sample and so mislabel genuinely-uncovered positions as
"searched and empty". This offline pass keeps the recorded counts (no
re-query -- results/*_galaxies.csv are curated and must not be regenerated)
and recomputes in_footprint with the cached CDS MOCs, then reclassifies via
classify_coverage (data-beats-footprint ordering).

  conda run -n flits python galaxies/foreground/refresh_coverage.py

Writes results/survey_coverage.csv (canonical, committed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from galaxies.foreground.survey_coverage import (  # noqa: E402
    classify_coverage,
    survey_in_footprint,
)

SRC = REPO / "scratch/repro-foreground-search-hpcc/survey_coverage.csv"
DST = REPO / "results/survey_coverage.csv"


def main() -> int:
    df = pd.read_csv(SRC)
    out = []
    for _, r in df.iterrows():
        coord = SkyCoord(str(r["ra"]), str(r["dec"]))  # sexagesimal (20h40m47s, +72d52m56s)
        in_fp = survey_in_footprint(str(r["survey"]), coord)
        status = classify_coverage(
            in_footprint=bool(in_fp),
            raw_count=int(r["raw_count"]),
            foreground_count=int(r["foreground_count"]),
        )
        out.append({**r.to_dict(), "in_footprint": bool(in_fp), "status": status})
    res = pd.DataFrame(out)
    res.to_csv(DST, index=False)
    changed = (res["status"] != df["status"]).sum()
    print(f"wrote {DST} ({len(res)} rows; {changed} statuses changed vs the dec-rule run)")
    for _, r in res[res["status"] != df["status"]].iterrows():
        print(f"  {r['nickname']:12s} {r['survey']:16s} -> {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
