"""Independent verification of the PS1-STRM adjudication (ps1_strm_resolution.csv).

Three orthogonal checks:
 A) COLUMN EXTRACTION: re-parse the full 19-column raw rows (strm_catalog_rows_full.csv,
    schema = README order) and assert class/z_phot/z_photErr/prob_Galaxy/extrap match the
    cut-column file the adjudicator used. Catches an off-by-one in the `cut -f` extraction
    (i.e. proves z_phot really is column 14, not a neighbour).
 B) OBJECT IDENTITY: assert each STRM row's own (raMean,decMean) is within 1" of the sheet
    coordinates for that objID (from foreground.csv) -> the grep matched the right source.
 C) VERDICT LOGIC: re-derive strm_verdict from the catalog columns via an independent
    reimplementation and assert it matches ps1_strm_resolution.csv; check headline counts.
Exit non-zero on any disagreement.
"""

import os
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

HERE = os.path.dirname(os.path.abspath(__file__))
full = pd.read_csv(os.path.join(HERE, "strm_catalog_rows_full.csv"))
cut = pd.read_csv(os.path.join(HERE, "strm_catalog_rows.csv"))
res = pd.read_csv(os.path.join(HERE, "ps1_strm_resolution.csv"))
fg = pd.read_csv(os.path.join(HERE, "foreground.csv"))
fg["obj"] = fg.obj.astype(str)
fails = []

# ---- A) column-extraction correctness: full reparse vs cut file ----
full["objID"] = full.objID.astype("int64")
cut["objID"] = cut.objID.astype("int64")
fa = full.set_index("objID")
ca = cut.set_index("objID")
for oid in ca.index:
    for col in ["class", "z_phot", "z_photErr", "prob_Galaxy", "extrapolation_Photoz"]:
        a, b = fa.loc[oid, col], ca.loc[oid, col]
        same = (a == b) if isinstance(a, str) else abs(float(a) - float(b)) < 1e-9
        if not same:
            fails.append(f"COLEXTRACT {oid} {col}: full={a} cut={b}")

# ---- B) object identity: STRM coords vs sheet coords ----
for _, r in full.iterrows():
    srow = fg[fg.obj == str(r.objID)]
    if srow.empty:
        fails.append(f"IDENTITY {r.objID}: not in foreground.csv")
        continue
    s = srow.iloc[0]
    sep = (
        SkyCoord(s.ra_deg * u.deg, s.dec_deg * u.deg)
        .separation(SkyCoord(r.raMean * u.deg, r.decMean * u.deg))
        .arcsec
    )
    if sep > 1.0:
        fails.append(f"IDENTITY {r.objID}: STRM row {sep:.2f}as from sheet coords (>1as)")


# ---- C) verdict re-derivation ----
def rederive(r):
    cls = r["class"]
    if cls != "GALAXY":
        return "inconclusive-not-galaxy"
    if r.extrapolation_Photoz == 1:
        return "inconclusive-extrapolated-photoz"
    host = r.host_z_spec
    if not np.isfinite(host):
        return "inconclusive-host-z-unknown"
    z, dz = r.z_phot, r.z_photErr
    if z + dz < host:
        return "confirmed"
    if z - dz > host:
        return "refuted"
    return "inconclusive-borderline"


fa2 = full.set_index("objID")
for _, rv in res.iterrows():
    fr = fa2.loc[int(rv.obj)]
    merged = pd.Series(
        dict(
            **{k: fr[k] for k in ("class", "extrapolation_Photoz", "z_phot", "z_photErr")},
            host_z_spec=rv.host_z_spec,
        )
    )
    indep = rederive(merged)
    if not str(rv.strm_verdict).startswith(indep):
        fails.append(f"VERDICT {rv.obj}: script='{rv.strm_verdict}' indep='{indep}'")

# headline counts
n_unsure = (full["class"] != "GALAXY").sum()
n_extrap = ((full["class"] == "GALAXY") & (full.extrapolation_Photoz == 1)).sum()
n_conf = res.strm_verdict.str.startswith("confirmed").sum()
print(f"A) column-extraction checks: {len(ca) * 5} comparisons")
print(f"B) identity checks: {len(full)} rows vs sheet coords")
print(f"C) verdict re-derivations: {len(res)}")
print(f"headline: UNSURE={n_unsure} extrapolated={n_extrap} confirmed-foreground={n_conf}")
assert n_unsure == 4 and n_extrap == 2 and n_conf == 0, "headline counts changed!"
if fails:
    print(f"\nFAILED ({len(fails)}):")
    for x in fails:
        print("  " + x)
    sys.exit(1)
print("\nALL STRM ADJUDICATION CHECKS PASS")
