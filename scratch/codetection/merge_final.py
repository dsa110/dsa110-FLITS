"""Combine the catalog validation and the PS1-STRM closure into one authoritative table.

foreground_validated.csv carries verdicts for all 49 from ls_dr9/DESI/NED. For the 9
'WISE,PS1,STRM' halos that came back with no independent redshift, ps1_strm_resolution.csv
carries the direct PS1-STRM adjudication (class + z_phot +/- z_photErr + extrapolation).
This merges them into foreground_final.csv with a single final_verdict (+ reason) and
self-checks the headline counts.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
val = pd.read_csv(os.path.join(HERE, "foreground_validated.csv"))
strm = pd.read_csv(os.path.join(HERE, "ps1_strm_resolution.csv"))
val["obj"] = val.obj.astype(str)
strm["obj"] = strm.obj.astype(str)
strm_map = strm.set_index("obj")

KEEP_STRM = [
    "strm_class",
    "strm_zphot",
    "strm_zphoterr",
    "strm_extrapolated",
    "sheet_vs_strm_zphot_diff",
    "strm_verdict",
]


def collapse(verdict):
    s = str(verdict)
    if s.startswith("confirmed"):
        return "confirmed"
    if s.startswith("refuted"):
        return "refuted"
    return "inconclusive"


rows = []
for _, r in val.iterrows():
    d = r.to_dict()
    if r.obj in strm_map.index:  # one of the 9 STRM halos: PS1-STRM is authoritative
        s = strm_map.loc[r.obj]
        for k in KEEP_STRM:
            d[k] = s[k]
        d["final_verdict"] = collapse(s.strm_verdict)
        d["final_reason"] = s.strm_verdict
    else:
        for k in KEEP_STRM:
            d[k] = np.nan
        d["final_verdict"] = collapse(r.foreground_verdict)
        if d["final_verdict"] == "inconclusive":
            d["final_reason"] = "photo-z within 1sigma of host (Legacy/Zhou DR9)"
        else:
            d["final_reason"] = f"{r.best_z_source} vs host z"
    rows.append(d)

final = pd.DataFrame(rows)
cols = [
    "nickname",
    "type",
    "obj",
    "survey",
    "ra_deg",
    "dec_deg",
    "host_z_spec",
    "sheet_zphot",
    "internal_flags",
    "exists",
    "best_z",
    "best_z_source",
    "foreground_verdict",
    *KEEP_STRM,
    "final_verdict",
    "final_reason",
]
final = final[[c for c in cols if c in final.columns]]
final.to_csv(os.path.join(HERE, "foreground_final.csv"), index=False)

vc = final.final_verdict.value_counts().to_dict()
print(final.groupby(["final_verdict"]).size().to_string())
print("\nby type:")
print(final.groupby(["type", "final_verdict"]).size().to_string())
print(f"\ntotal rows: {len(final)}")
# self-check against the independently established numbers
assert len(final) == 49, f"expected 49, got {len(final)}"
assert vc.get("confirmed") == 29, vc
assert vc.get("refuted") == 7, vc
assert vc.get("inconclusive") == 13, vc
assert final[final.final_verdict == "confirmed"].exists.all(), "confirmed row not in any catalog"
# the 9 STRM halos must all be inconclusive and carry a strm_class
strm_rows = final[final.obj.isin(strm.obj)]
assert len(strm_rows) == 9 and (strm_rows.final_verdict == "inconclusive").all()
assert strm_rows.strm_class.notna().all()
print(
    "\nSELF-CHECK PASS: 49 rows, 29 confirmed / 7 refuted / 13 inconclusive; "
    "9 STRM halos carry PS1-STRM class and are inconclusive."
)
