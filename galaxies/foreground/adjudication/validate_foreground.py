"""External-catalog validation of the 49 foreground entries.

For each intervening object the spreadsheet cannot self-verify three things:
  (1) existence  - a real source at the listed RA/Dec in a public catalog
  (2) redshift   - the catalog z (and, crucially, its error, which the sheet omits)
  (3) class      - galaxy vs star/QSO (vs cluster)

Engines (all confirmed reachable; matched by COORDINATES because the sheet's small
obj_IDs are per-FRB row indices, not catalog IDs):
  - NOIRLab Data Lab TAP  ls_dr9.tractor JOIN ls_dr9.photo_z  (Zhou+2021 DR9 photo-z
        with z_phot_std = the missing error; + morphological type)
  - NOIRLab Data Lab TAP  desi_dr1.zpix                       (DESI DR1 spec-z, zwarn/spectype)
  - NED cone search                                            (aggregate z + type; primary for clusters)

Verdict per object uses the best available redshift vs the FRB host spec-z:
  spec-z < host -> confirmed foreground ; spec-z >= host -> refuted (background)
  photo-z only  -> confirmed if z_phot_mean + z_phot_std < host
                   refuted   if z_phot_mean - z_phot_std > host
                   else inconclusive (within 1 sigma)
  host z unknown or no catalog match -> inconclusive
Read-only public queries; writes foreground_validated.csv next to this script.
"""

import os
import socket
import time
import warnings

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(90)

import numpy as np
import pandas as pd
import pyvo
from astropy import units as u
from astropy.coordinates import SkyCoord

HERE = os.path.dirname(os.path.abspath(__file__))
fg = pd.read_csv(os.path.join(HERE, "foreground.csv"))

TAP = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")
GAL_RADIUS_AS = 5.0  # galaxy positional match (STRM/PS1 vs Legacy astrometry slack)
CLU_RADIUS_AS = 90.0  # cluster centroid/BCG match (looser)


def _nearest(rows, ra, dec, racol, deccol, rad_as=None):
    """Return (row, sep_arcsec) of nearest row within rad_as, or (None, nan).

    Data Lab's ADQL backend does not translate POINT/CIRCLE geometry and rejects
    bare q3c in WHERE, so the SQL uses a cheap ra/dec bounding box; the true cone
    cut is enforced here on the exact angular separation.
    """
    if rows is None or len(rows) == 0:
        return None, np.nan
    c0 = SkyCoord(ra * u.deg, dec * u.deg)
    cs = SkyCoord(np.asarray(rows[racol], float) * u.deg, np.asarray(rows[deccol], float) * u.deg)
    sep = c0.separation(cs).arcsec
    i = int(np.argmin(sep))
    if rad_as is not None and sep[i] > rad_as:
        return None, np.nan
    return rows[i], float(sep[i])


def _bbox(ra, dec, rad_as):
    ddec = rad_as / 3600.0
    dra = ddec / max(np.cos(np.deg2rad(dec)), 0.02)
    return ra - dra, ra + dra, dec - ddec, dec + ddec


def q_lsdr9(ra, dec, rad_as):
    r0, r1, d0, d1 = _bbox(ra, dec, rad_as)
    q = (
        "SELECT t.ra, t.dec, t.type, p.z_phot_mean, p.z_phot_std, p.z_phot_median, p.z_spec "
        "FROM ls_dr9.tractor t JOIN ls_dr9.photo_z p ON t.ls_id = p.ls_id "
        f"WHERE t.dec BETWEEN {d0} AND {d1} AND t.ra BETWEEN {r0} AND {r1}"
    )
    r = TAP.search(q).to_table()
    row, sep = _nearest(r, ra, dec, "ra", "dec", rad_as)
    if row is None:
        return {}
    return dict(
        lsdr9_sep_as=round(sep, 2),
        lsdr9_type=str(row["type"]).strip(),
        lsdr9_zphot=float(row["z_phot_mean"]),
        lsdr9_zphot_std=float(row["z_phot_std"]),
        lsdr9_zspec=(
            float(row["z_spec"])
            if row["z_spec"] is not None and float(row["z_spec"]) > -90
            else np.nan
        ),
    )


def q_desi(ra, dec, rad_as):
    r0, r1, d0, d1 = _bbox(ra, dec, rad_as)
    q = (
        "SELECT mean_fiber_ra, mean_fiber_dec, z, zerr, zwarn, spectype "
        "FROM desi_dr1.zpix "
        f"WHERE mean_fiber_dec BETWEEN {d0} AND {d1} "
        f"AND mean_fiber_ra BETWEEN {r0} AND {r1} AND zwarn = 0"
    )
    r = TAP.search(q).to_table()
    row, sep = _nearest(r, ra, dec, "mean_fiber_ra", "mean_fiber_dec", rad_as)
    if row is None:
        return {}
    return dict(
        desi_sep_as=round(sep, 2),
        desi_specz=float(row["z"]),
        desi_zerr=float(row["zerr"]),
        desi_spectype=str(row["spectype"]).strip(),
    )


def q_ned(ra, dec, rad_as):
    from astroquery.ipac.ned import Ned

    t = Ned.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=rad_as * u.arcsec)
    if t is None or len(t) == 0:
        return {}
    row, sep = _nearest(t, ra, dec, "RA", "DEC")
    z = row["Redshift"]
    return dict(
        ned_sep_as=round(sep, 2),
        ned_name=str(row["Object Name"]).strip(),
        ned_type=str(row["Type"]).strip(),
        ned_z=(float(z) if z is not None and not np.ma.is_masked(z) else np.nan),
    )


def q_simbad(ra, dec, rad_as):
    # Best-effort secondary classification; SIMBAD is sparse for faint photo-z galaxies.
    # Defensive: votable-field names differ across astroquery versions -> never raise.
    from astroquery.simbad import Simbad

    s = Simbad()
    for fld in (("otype", "rvz_redshift"), ("otype", "rv_value"), ()):
        try:
            if fld:
                s.add_votable_fields(*fld)
            break
        except Exception:
            s = Simbad()
    t = s.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=rad_as * u.arcsec)
    if t is None or len(t) == 0:
        return {}
    racol = "ra" if "ra" in t.colnames else "RA"
    deccol = "dec" if "dec" in t.colnames else "DEC"
    try:
        row, sep = _nearest(t, ra, dec, racol, deccol, rad_as)
    except Exception:
        row, sep = t[0], np.nan
    if row is None:
        return {}
    otype = next((str(row[c]) for c in ("otype", "OTYPE") if c in t.colnames), "")
    zc = next((c for c in ("rvz_redshift", "z_value", "RVZ_REDSHIFT") if c in t.colnames), None)
    sz = np.nan
    if zc is not None and row[zc] is not None and not np.ma.is_masked(row[zc]):
        try:
            sz = float(row[zc])
        except (TypeError, ValueError):
            sz = np.nan
    return dict(
        simbad_sep_as=(round(sep, 2) if sep == sep else np.nan),
        simbad_otype=otype.strip(),
        simbad_z=sz,
    )


def best_z_and_verdict(host_z, sheet_zphot, d):
    """Pick best redshift; return (best_z, source, verdict)."""
    # priority: DESI spec-z > LS z_spec > LS z_phot(+std) > NED z
    if not np.isnan(d.get("desi_specz", np.nan)):
        bz, src, std = d["desi_specz"], "desi_specz", 0.0
    elif not np.isnan(d.get("lsdr9_zspec", np.nan)):
        bz, src, std = d["lsdr9_zspec"], "lsdr9_zspec", 0.0
    elif not np.isnan(d.get("lsdr9_zphot", np.nan)):
        bz, src, std = d["lsdr9_zphot"], "lsdr9_zphot", d.get("lsdr9_zphot_std", np.nan)
    elif not np.isnan(d.get("ned_z", np.nan)):
        bz, src, std = d["ned_z"], "ned_z", np.nan
    else:
        return np.nan, "none", "inconclusive"
    if np.isnan(host_z):
        return bz, src, "inconclusive"  # cannot judge foreground without host z
    if std == 0.0:  # spec-z
        return bz, src, ("confirmed" if bz < host_z else "refuted")
    if np.isnan(std):
        return bz, src, ("confirmed" if bz < host_z else "refuted")
    if bz + std < host_z:
        return bz, src, "confirmed"
    if bz - std > host_z:
        return bz, src, "refuted"
    return bz, src, "inconclusive"


out = []
for i, e in fg.iterrows():
    rad = CLU_RADIUS_AS if e.type == "cluster" else GAL_RADIUS_AS
    d = {}
    for name, fn in [("lsdr9", q_lsdr9), ("desi", q_desi), ("ned", q_ned), ("simbad", q_simbad)]:
        try:
            d.update(fn(e.ra_deg, e.dec_deg, rad))
        except Exception as ex:
            d[f"{name}_err"] = f"{ex.__class__.__name__}"
        time.sleep(0.3)
    exists = any(k in d for k in ("lsdr9_sep_as", "desi_sep_as", "ned_sep_as"))
    bz, src, verdict = best_z_and_verdict(e.host_z_spec, e.z_phot, d)
    rec = dict(
        nickname=e.nickname,
        type=e.type,
        obj=e.obj,
        survey=e.survey,
        ra_deg=e.ra_deg,
        dec_deg=e.dec_deg,
        host_z_spec=e.host_z_spec,
        sheet_zphot=e.z_phot,
        internal_flags=e["flags"],
        exists=exists,
        best_z=bz,
        best_z_source=src,
        foreground_verdict=verdict,
        **d,
    )
    out.append(rec)
    print(
        f"[{i + 1:2d}/49] {e.nickname:11s} {e.type:7s} {str(e.obj)[:20]:20s} "
        f"exists={exists} best_z={bz if isinstance(bz, str) else round(bz, 4) if not np.isnan(bz) else 'NA'} "
        f"({src}) -> {verdict}"
    )

vdf = pd.DataFrame(out)
vdf.to_csv(os.path.join(HERE, "foreground_validated.csv"), index=False)

print("\n=== SUMMARY ===")
print("exists in >=1 catalog: %d / %d" % (vdf.exists.sum(), len(vdf)))
print("\nforeground_verdict:")
print(vdf.foreground_verdict.value_counts().to_string())
print("\nverdict x internal_flag (ZPHOT_GE_ZHOST rows):")
sub = vdf[vdf.internal_flags.fillna("").str.contains("ZPHOT_GE_ZHOST")]
print(
    sub[
        [
            "nickname",
            "type",
            "obj",
            "sheet_zphot",
            "host_z_spec",
            "best_z",
            "best_z_source",
            "foreground_verdict",
        ]
    ].to_string(index=False)
)
