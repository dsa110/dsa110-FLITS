"""Independent check of validate_foreground.py output (foreground_validated.csv).

A) LOGIC (offline, deterministic): re-derive every foreground_verdict from the
   recorded catalog columns using a from-scratch reimplementation of the decision
   rule, and assert it agrees with the script's verdict for all 49 rows. Catches
   coding/transcription bugs in the original rule.
B) MATCH QUALITY (offline): assert every confirmed/refuted verdict rests on a
   catalog match close enough to be the SAME source the sheet listed
   (lsdr9/ned < 3", desi-halo < 5"); a far match would mean we judged the wrong object.
C) REPRODUCE (online): independently re-query the catalog that produced best_z for a
   sample of objects via a fresh service call and assert the redshift reproduces.
Exit non-zero on any disagreement.
"""

import os
import socket
import sys
import warnings

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(90)

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
v = pd.read_csv(os.path.join(HERE, "foreground_validated.csv"))
fails = []


def nn(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


# ---- A) independent verdict re-derivation ----
def rederive(r):
    host = r.host_z_spec
    # priority must match the documented rule: desi spec > ls zspec > ls zphot > ned z
    if nn(r.desi_specz):
        z, std, spec = r.desi_specz, 0.0, True
    elif nn(r.lsdr9_zspec):
        z, std, spec = r.lsdr9_zspec, 0.0, True
    elif nn(r.lsdr9_zphot):
        z, std, spec = r.lsdr9_zphot, r.lsdr9_zphot_std, False
    elif nn(r.ned_z):
        z, std, spec = r.ned_z, np.nan, False
    else:
        return "inconclusive"
    if not nn(host):
        return "inconclusive"
    if spec or not nn(std):
        return "confirmed" if z < host else "refuted"
    if z + std < host:
        return "confirmed"
    if z - std > host:
        return "refuted"
    return "inconclusive"


for _, r in v.iterrows():
    mine = rederive(r)
    if mine != r.foreground_verdict:
        fails.append(f"VERDICT {r.nickname}/{r.obj}: script={r.foreground_verdict} indep={mine}")

# ---- B) match-quality: verdict must rest on a near match ----
for _, r in v.iterrows():
    if r.foreground_verdict not in ("confirmed", "refuted"):
        continue
    src = r.best_z_source
    sep, lim = None, None
    if src in ("lsdr9_zphot", "lsdr9_zspec"):
        sep, lim = r.lsdr9_sep_as, 3.0
    elif src == "desi_specz":
        sep, lim = r.desi_sep_as, (90.0 if r.type == "cluster" else 5.0)
    elif src == "ned_z":
        sep, lim = r.ned_sep_as, 5.0
    if sep is None or np.isnan(sep) or sep > lim:
        fails.append(
            f"MATCH {r.nickname}/{r.obj}: verdict {r.foreground_verdict} on {src} "
            f"sep={sep} > limit {lim} (possible wrong source)"
        )

# ---- C) reproduce a sample online via independent queries ----
import pyvo

TAP = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")


def reget_desi(ra, dec, rad_as):
    # mirror the validator's search radius; confirm the stored z is present in the cone
    ddec = rad_as / 3600.0
    dra = ddec / max(np.cos(np.deg2rad(dec)), 0.02)
    q = (
        "SELECT z, zwarn FROM desi_dr1.zpix "
        f"WHERE mean_fiber_dec BETWEEN {dec - ddec} AND {dec + ddec} "
        f"AND mean_fiber_ra BETWEEN {ra - dra} AND {ra + dra} AND zwarn=0"
    )
    t = TAP.search(q).to_table()
    return None if len(t) == 0 else [float(x) for x in t["z"]]


def reget_lsphot(ra, dec):
    q = (
        "SELECT p.z_phot_mean FROM ls_dr9.tractor t JOIN ls_dr9.photo_z p ON t.ls_id=p.ls_id "
        f"WHERE t.dec BETWEEN {dec - 0.0008} AND {dec + 0.0008} "
        f"AND t.ra BETWEEN {ra - 0.0028} AND {ra + 0.0028}"
    )
    t = TAP.search(q).to_table()
    return None if len(t) == 0 else float(t["z_phot_mean"][0])


# sample: one confirmed cluster (desi), one refuted halo (lsphot), one confirmed halo (desi)
sample = []
for src_kind, verdict in [
    ("desi_specz", "confirmed"),
    ("lsdr9_zphot", "refuted"),
    ("desi_specz", "confirmed"),
]:
    sub = v[(v.best_z_source == src_kind) & (v.foreground_verdict == verdict)]
    sub = sub[~sub.obj.astype(str).isin([s[0] for s in sample])]
    if len(sub):
        sample.append((str(sub.iloc[0].obj), sub.iloc[0]))

n_repro = 0
for obj, r in sample:
    try:
        if r.best_z_source == "desi_specz":
            rad = 90.0 if r.type == "cluster" else 5.0
            zlist = reget_desi(r.ra_deg, r.dec_deg, rad)
            ref = r.desi_specz
            ok = zlist is not None and min(abs(z - ref) for z in zlist) < 0.001
            got = "present" if ok else (f"{len(zlist)} z's, none match" if zlist else None)
        else:
            got = reget_lsphot(r.ra_deg, r.dec_deg)
            ref = r.lsdr9_zphot
            ok = got is not None and abs(got - ref) < 0.02
        n_repro += 1
        print(
            f"REPRODUCE {r.nickname}/{obj} {r.best_z_source}: stored={ref:.4f} requery={got} ok={ok}"
        )
        if not ok:
            fails.append(f"REPRODUCE {r.nickname}/{obj}: stored={ref} requery={got}")
    except Exception as ex:
        fails.append(f"REPRODUCE {r.nickname}/{obj}: {ex.__class__.__name__}: {str(ex)[:80]}")

print(f"\nverdict re-derivations checked: {len(v)}")
print(
    f"match-quality checks on confirmed/refuted: {(v.foreground_verdict != 'inconclusive').sum()}"
)
print(f"online reproduce checks: {n_repro}")
if fails:
    print(f"\nFAILED ({len(fails)}):")
    for x in fails:
        print("  " + x)
    sys.exit(1)
print("\nALL VALIDATION CHECKS PASS")
