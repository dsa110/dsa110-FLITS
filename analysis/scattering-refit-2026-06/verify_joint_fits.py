#!/usr/bin/env python
"""Verify every co-detected joint fit ran successfully and report pathologies.

For each of the 12 co-detected bursts: confirm joint_fit.json + joint_samples.npz
exist; report alpha (with prior-rail flags), delta_dm sanity (vs +-50 prior),
zeta, lnZ. Flags missing/failed fits and rail/degeneracy pathologies so "all ran
successfully" is backed by evidence, not assumed.

  python verify_joint_fits.py
"""

import json
import os

RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
J = f"{RUNS}/data/joint"
BURSTS = (
    "casey chromatica freya hamilton isha johndoeII mahi oran phineas whitney wilhelm zach".split()
)
ALO, AHI = 1.0, 6.0
EDGE = 0.05
DDM_PRIOR = 50.0

hdr = (
    f"{'burst':12s} {'json':5s} {'npz':4s} {'alpha (p16,p84)':20s} "
    f"{'dDM_C':8s} {'dDM_D':8s} {'zC':6s} {'zD':6s} {'lnZ':10s} flags"
)
print(hdr)
print("-" * len(hdr))
nfail = nflag = 0
for b in BURSTS:
    jf, sf = f"{J}/{b}_joint_fit.json", f"{J}/{b}_joint_samples.npz"
    hasj, hass = os.path.exists(jf), os.path.exists(sf)
    if not hasj:
        print(f"{b:12s} {'NO':5s} {('yes' if hass else 'NO'):4s} *** FIT MISSING / JOB FAILED ***")
        nfail += 1
        continue
    d = json.load(open(jf))
    p = d["percentiles"]
    am, alo, ahi = p["alpha"]["median"], p["alpha"]["lower"], p["alpha"]["upper"]
    flags = []
    if alo <= ALO + EDGE:
        flags.append("aRAIL_LO")
    if ahi >= AHI - EDGE:
        flags.append("aRAIL_HI")
    ddmc, ddmd = p["delta_dm_C"]["median"], p["delta_dm_D"]["median"]
    if abs(ddmc) > 0.9 * DDM_PRIOR:
        flags.append("dDMc_rail")
    if abs(ddmd) > 0.9 * DDM_PRIOR:
        flags.append("dDMd_rail")
    if d.get("shared_zeta"):
        zc = zd = p["zeta_1ghz"]["median"]  # shared zeta(nu): report the 1-GHz width
        flags.append("shared")
    else:
        zc, zd = p["zeta_C"]["median"], p["zeta_D"]["median"]
    lnz = d.get("log_evidence", float("nan"))
    if not hass:
        flags.append("NO_SAMPLES")
    astr = f"{am:.2f} [{alo:.2f},{ahi:.2f}]"
    print(
        f"{b:12s} {'yes':5s} {('yes' if hass else 'NO'):4s} {astr:20s} "
        f"{ddmc:+8.1f} {ddmd:+8.1f} {zc:6.2f} {zd:6.2f} {lnz:10.1f} {','.join(flags)}"
    )
    if [f for f in flags if f != "shared"]:  # "shared" is informational, not a pathology
        nflag += 1

print()
print(
    f"summary: {len(BURSTS)} bursts | {nfail} missing/failed | {nflag} flagged "
    f"(alpha rails / dDM degeneracy / no samples)"
)
print(
    "note: aRAIL_LO = alpha posterior pressing the 1.0 prior floor -> floor may be "
    "truncating a genuinely shallower slope; consider lowering --alpha-lo for those."
)
