"""Resolve the 9 'WISE,PS1,STRM' halos that came back inconclusive.

Finding from probing: these positions are OUTSIDE the DESI Legacy / SDSS footprint
(Legacy DR9 tractor returns 0 sources within 600" of zach's halo) and outside DESI
spectroscopy, so no independent redshift exists in VizieR / NOIRLab Data Lab. Their
only redshift is the PS1-STRM photo-z (Beck+2021), a MAST HLSP not exposed to simple
cone queries. This script does what IS possible from accessible services:
  - confirm each is a real PS1 source at the sheet objID (MAST PanSTARRS DR2, best effort)
  - document the Legacy-footprint gap per object (Data Lab tractor count within 60")
  - leave the foreground verdict honestly 'inconclusive-no-independent-z' (not fabricated)
Writes ps1_strm_resolution.csv.
"""

import os
import socket
import warnings

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(120)

import numpy as np
import pandas as pd
import pyvo
from astropy import units as u

HERE = os.path.dirname(os.path.abspath(__file__))
fg = pd.read_csv(os.path.join(HERE, "foreground.csv"))
val = pd.read_csv(os.path.join(HERE, "foreground_validated.csv"))

# the 9 STRM halos that are inconclusive with no catalog z
strm = fg[fg.survey.fillna("").str.contains("STRM")].copy()
inconc = set(
    val[(val.foreground_verdict == "inconclusive") & (val.best_z_source == "none")].obj.astype(str)
)
strm = strm[strm.obj.astype(str).isin(inconc)]

svc = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")


def legacy_count(ra, dec, rad_as=60.0):
    ddec = rad_as / 3600.0
    dra = ddec / max(np.cos(np.deg2rad(dec)), 0.02)
    q = (
        "SELECT COUNT(*) AS n FROM ls_dr9.tractor "
        f"WHERE dec BETWEEN {dec - ddec} AND {dec + ddec} AND ra BETWEEN {ra - dra} AND {ra + dra}"
    )
    return int(svc.search(q).to_table()["n"][0])


def ps1_confirm(ra, dec, objid):
    from astroquery.mast import Catalogs

    try:
        t = Catalogs.query_region(
            f"{ra} {dec}",
            radius=3 * u.arcsec,
            catalog="Panstarrs",
            data_release="dr2",
            table="mean",
            columns=["objID", "raMean", "decMean"],
        )
        if t is None or len(t) == 0:
            return False, np.nan, np.nan
        ids = [int(x) for x in t["objID"]]
        match = int(objid) in ids
        # nearest
        from astropy.coordinates import SkyCoord

        c0 = SkyCoord(ra * u.deg, dec * u.deg)
        cs = SkyCoord(
            np.asarray(t["raMean"], float) * u.deg, np.asarray(t["decMean"], float) * u.deg
        )
        sep = float(np.min(c0.separation(cs).arcsec))
        return match, sep, len(t)
    except Exception as ex:
        return f"ERR:{ex.__class__.__name__}", np.nan, np.nan


rows = []
for _, e in strm.iterrows():
    lc = legacy_count(e.ra_deg, e.dec_deg)
    objmatch, sep, npix = ps1_confirm(e.ra_deg, e.dec_deg, e.obj)
    rows.append(
        dict(
            nickname=e.nickname,
            obj=e.obj,
            ra_deg=e.ra_deg,
            dec_deg=e.dec_deg,
            host_z_spec=e.host_z_spec,
            sheet_zphot=e.z_phot,
            legacy_sources_within_60as=lc,
            in_legacy_footprint=(lc > 0),
            ps1_objid_confirmed=objmatch,
            ps1_nearest_sep_as=(round(sep, 2) if sep == sep else np.nan),
            resolution="inconclusive-no-independent-z (outside Legacy/DESI/SDSS; PS1-STRM only)",
        )
    )
    print(
        f"{e.nickname:11s} obj={str(e.obj):20s} legacy60as={lc} "
        f"ps1_objid_confirmed={objmatch} sep={sep}"
    )

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "ps1_strm_resolution.csv"), index=False)
print("\n=== SUMMARY (9 STRM halos) ===")
print(
    "in Legacy footprint (>0 sources within 60as):",
    int(out.in_legacy_footprint.sum()),
    "/",
    len(out),
)
print("PS1 objID confirmed:", int((out.ps1_objid_confirmed == True).sum()), "/", len(out))  # noqa: E712
print("\nAll remain: inconclusive-no-independent-z (PS1-STRM photo-z is the only redshift).")
