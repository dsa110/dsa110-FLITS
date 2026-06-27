"""Recoverability probe for the 5 foreground galaxies dropped by the DESI
VII/292/north upstream refresh. For each, query current sources at the exact old
position (small cone) and report whether a counterpart still exists, with z + sep.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

GALAXIES = [
    ("freya", 88.196396, 74.198901, 0.767),
    ("hamilton", 305.035134, 70.793732, 0.222),
    ("johndoeii", 335.976327, 73.024977, 0.767),
    ("wilhelm_a", 315.132891, 72.035767, 0.46),
    ("wilhelm_b", 315.127412, 72.035548, 0.46),
]
CONE = 0.02 * u.deg  # 72 arcsec


def _nearest(coord, df, ra="ra", dec="dec"):
    if df is None or len(df) == 0:
        return None, 0
    m = SkyCoord(np.asarray(df[ra], float) * u.deg, np.asarray(df[dec], float) * u.deg)
    seps = coord.separation(m).arcsec
    return float(np.min(seps)), len(df)


def probe_ned(gid, coord, z):
    from galaxies.v2_0.engines_extra import NedTapEngine

    df = NedTapEngine().query(coord, CONE)
    sep, n = _nearest(coord, df)
    zr = None
    if n and sep is not None:
        m = SkyCoord(np.asarray(df["ra"], float) * u.deg, np.asarray(df["dec"], float) * u.deg)
        j = int(np.argmin(coord.separation(m).arcsec))
        zr = float(df.iloc[j]["z"]) if "z" in df and df.iloc[j]["z"] == df.iloc[j]["z"] else None
    return (gid, "NED_TAP", n, sep, zr)


def probe_desi_north(gid, coord, z):
    res = Vizier(row_limit=-1, timeout=45).query_region(coord, radius=CONE, catalog="VII/292/north")
    df = res[0].to_pandas() if res else None
    if df is not None and "RAJ2000" in df:
        df = df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})
    sep, n = _nearest(coord, df) if df is not None else (None, 0)
    return (gid, "VII/292/north", n, sep, None)


def probe_lsdr10(gid, coord, z):
    from galaxies.v2_0.engines_extra import DesiLsDr10Engine

    df = DesiLsDr10Engine().query(coord, CONE)
    sep, n = _nearest(coord, df)
    return (gid, "LS_DR10", n, sep, None)


tasks = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = []
    for gid, ra, dec, z in GALAXIES:
        c = SkyCoord(ra, dec, unit="deg")
        for fn in (probe_ned, probe_desi_north, probe_lsdr10):
            futs.append(ex.submit(fn, gid, c, z))
    for fut in as_completed(futs):
        try:
            tasks.append(fut.result())
        except Exception as e:
            tasks.append(("?", f"ERR:{type(e).__name__}", 0, None, str(e)[:60]))

for gid, _, _, _ in [(g[0], 0, 0, 0) for g in GALAXIES]:
    rows = [t for t in tasks if t[0] == gid]
    print(f"\n{gid}:")
    for _, src, n, sep, zr in sorted(rows, key=lambda r: r[1]):
        seps = f"{sep:.1f}in" if sep is not None else "-"
        zs = f"z={zr:.4f}" if zr is not None else ""
        hit = "FOUND" if (n and sep is not None and sep < 5.0) else "none"
        print(f"  {src:16s} n={n:3d} nearest={seps:8s} {zs} -> {hit}")
