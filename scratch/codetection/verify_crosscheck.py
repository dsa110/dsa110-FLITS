"""Independent cross-check of the normalizer output.

Oracle sources are the OTHER sheets, parsed by a different code path than
normalize_codetection.py:
  - bursts.csv      vs Sheet3 (independent clean burst table)
  - foreground.csv  vs Sheet2 (independent wide layout; per-object values)
A column mis-map, dropped row, or duplicated row in the Sheet1 parser would
surface here as a value mismatch or a set-difference. Exit non-zero on any failure.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source")
S2 = os.path.join(SRC, "DSA110_CHIME_Codetection_BurstProperties_Foreground  - Sheet2.csv")
S3 = os.path.join(SRC, "DSA110_CHIME_Codetection_BurstProperties_Foreground  - Sheet3.csv")

bursts = pd.read_csv(os.path.join(HERE, "bursts.csv"))
fg = pd.read_csv(os.path.join(HERE, "foreground.csv"))

fails = []


def close(a, b, tol=1e-6):
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


# ---- bursts vs Sheet3 (keyed by TNS) ----
s3 = pd.read_csv(S3, header=0, dtype=str)
s3.columns = ["tns", "mjd", "loc", "ra_deg", "dec_deg", "z_spec"]
s3 = s3.set_index(s3.tns.str.strip())
n_burst_checks = 0
for _, b in bursts.iterrows():
    o = s3.loc[b.tns.strip()]
    for col, ocol in [
        ("mjd", "mjd"),
        ("ra_deg", "ra_deg"),
        ("dec_deg", "dec_deg"),
        ("z_spec", "z_spec"),
    ]:
        ov = float(o[ocol]) if str(o[ocol]).strip() not in ("", "nan") else np.nan
        if not close(b[col], ov, tol=1e-4):
            fails.append(f"BURST {b.nickname} {col}: out={b[col]} vs Sheet3={ov}")
        n_burst_checks += 1
assert len(bursts) == 12, f"expected 12 bursts, got {len(bursts)}"
assert set(bursts.tns.str.strip()) == set(s3.index), "burst TNS set != Sheet3 TNS set"

# ---- foreground vs Sheet2 (keyed by obj id / obj name) ----
s2 = pd.read_csv(S2, header=None, dtype=str).fillna("")
oracle = {}  # key -> (impact, ra, dec, zphot)
for i in range(1, len(s2)):
    r = s2.iloc[i]
    if r[12].strip() or r[13].strip():  # halo: label col 12, obj_ID col 13
        oracle[("halo", r[13].strip())] = (r[15], r[16], r[17], r[18])
    if r[24].strip():  # cluster: obj_name col 24
        oracle[("cluster", r[24].strip())] = (r[26], r[27], r[28], r[29])

n_fg_checks = 0
for _, e in fg.iterrows():
    key = (e.type, str(e.obj).strip())
    if key not in oracle:
        fails.append(f"FG {e.nickname}/{e.type} obj={e.obj}: not found in Sheet2 oracle")
        continue
    imp, ra, dec, zp = oracle[key]
    for val, ov, name in [
        (e.impact_kpc_listed, imp, "impact"),
        (e.ra_deg, ra, "ra"),
        (e.dec_deg, dec, "dec"),
        (e.z_phot, zp, "zphot"),
    ]:
        ovf = float(ov) if str(ov).strip() not in ("", "nan") else np.nan
        if not close(val, ovf, tol=1e-3):
            fails.append(f"FG {e.nickname}/{e.type} obj={e.obj} {name}: out={val} vs Sheet2={ovf}")
        n_fg_checks += 1

# set equality: same objects in both representations
out_halos = set(fg[fg.type == "halo"].obj.astype(str).str.strip())
ora_halos = {k[1] for k in oracle if k[0] == "halo"}
out_cl = set(fg[fg.type == "cluster"].obj.astype(str).str.strip())
ora_cl = {k[1] for k in oracle if k[0] == "cluster"}
if out_halos != ora_halos:
    fails.append(f"HALO set diff: only-out={out_halos - ora_halos} only-S2={ora_halos - out_halos}")
if out_cl != ora_cl:
    fails.append(f"CLUSTER set diff: only-out={out_cl - ora_cl} only-S2={ora_cl - out_cl}")

print(f"burst field checks: {n_burst_checks}  (12 bursts x 4 fields vs Sheet3)")
print(f"foreground value checks: {n_fg_checks}  ({len(fg)} objects x 4 fields vs Sheet2)")
print(f"halo set: out={len(out_halos)} oracle={len(ora_halos)} equal={out_halos == ora_halos}")
print(f"cluster set: out={len(out_cl)} oracle={len(ora_cl)} equal={out_cl == ora_cl}")
if fails:
    print(f"\nFAILED ({len(fails)}):")
    for x in fails:
        print("  " + x)
    sys.exit(1)
print("\nALL CROSS-CHECKS PASS")
