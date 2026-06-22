"""Independent verification of ps1_strm_resolution.csv.

The resolver confirmed each objID by a POSITION cone search. This checks the same
claim by the orthogonal path: look each objID up DIRECTLY BY ID in MAST PanSTARRS DR2
and assert the catalog's own coordinates fall within 1" of the sheet coordinates.
If the by-ID coords match the sheet position, the objID genuinely is that source.
Exit non-zero on any mismatch.
"""

import os
import socket
import sys
import warnings

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(120)

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs

HERE = os.path.dirname(os.path.abspath(__file__))
res = pd.read_csv(os.path.join(HERE, "ps1_strm_resolution.csv"))
fails = []
n = 0
for _, r in res.iterrows():
    try:
        t = Catalogs.query_criteria(
            catalog="Panstarrs",
            data_release="dr2",
            table="mean",
            objID=int(r.obj),
            columns=["objID", "raMean", "decMean"],
        )
        if t is None or len(t) == 0:
            fails.append(f"{r.nickname}/{r.obj}: objID not found by direct ID lookup")
            continue
        row = t[0]
        sep = (
            SkyCoord(r.ra_deg * u.deg, r.dec_deg * u.deg)
            .separation(SkyCoord(float(row["raMean"]) * u.deg, float(row["decMean"]) * u.deg))
            .arcsec
        )
        n += 1
        ok = sep < 1.0
        print(
            f"{r.nickname:11s} objID={r.obj} byID_coords=({float(row['raMean']):.5f},"
            f"{float(row['decMean']):.5f}) sep_to_sheet={sep:.3f}as ok={ok}"
        )
        if not ok:
            fails.append(f"{r.nickname}/{r.obj}: by-ID coords {sep:.2f}as from sheet (>1as)")
    except Exception as ex:
        fails.append(f"{r.nickname}/{r.obj}: {ex.__class__.__name__}: {str(ex)[:80]}")

print(f"\nby-ID identity checks: {n}/{len(res)}")
print(
    "legacy-footprint claim (all 0 within 60as):", bool((res.legacy_sources_within_60as == 0).all())
)
if fails:
    print(f"\nFAILED ({len(fails)}):")
    for x in fails:
        print("  " + x)
    sys.exit(1)
print("\nALL PS1 IDENTITY CHECKS PASS")
