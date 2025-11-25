#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Query four northern-sky catalogues for every galaxy (or spectrum /
# photometric entry) lying within ≤100 kpc *proper* of a foreground position.
#
# Catalogues queried
#   • NED   — region search, then per-object Redshift table
#   • Pan-STARRS PS1 DR2 (mean) — imaging + extended-source filter
#   • SDSS DR17 — spectra via SkyServer SQL API (no 3′ limit)
#   • DESI DR1 — spectra via VizieR (VII/378)
#
# Output
#   One Excel workbook “galaxies_100kpc_proper.xlsx”.
#   For every target, up to four sheets named  T##_<CAT>  with **exactly**
#       name | ra | dec | z | Mstar | Rproj_kpc | catalog
# ---------------------------------------------------------------------------

import time, warnings, sys
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from requests.exceptions import ConnectionError, ReadTimeout

# -------------------------- astroquery back-ends ---------------------------
from astroquery.ipac.ned import Ned            # NED
from astroquery.mast import Catalogs           # Pan-STARRS
from astroquery.sdss import SDSS               # SDSS SQL
from astroquery.vizier import Vizier           # DESI DR1

# ------------------------------- CONFIG ------------------------------------
Ned.TIMEOUT = 180                 # s per HTTP request
MAX_TRIES   = 5                   # network retries
BASE_DELAY  = 2                   # back-off base in seconds
PAUSE       = 0.5                 # polite pause between any two calls
R_PHYS      = 100 * u.kpc         # aperture in galaxy rest-frame
OUTFILE     = Path(f"galaxies_{int(R_PHYS.value)}kpc_proper.xlsx")

# silence harmless warnings
np.seterr(invalid="ignore")
warnings.filterwarnings("ignore",
        message="Field info are not available for this data release",
        module="astroquery.sdss")

# ---------------------------- TARGET LIST ----------------------------------
targets = [
    ("20h40m47.886s", "+72d52m56.378s", 0.0430),
    ("08h58m52.92s",  "+73d29m27.0s",   0.4790),
    ("21h12m10.760s", "+72d49m38.20s",  0.3005),
    ("04h45m38.64s",  "+70d18m26.6s",   0.2505),
    ("21h00m31.09s",  "+72d02m15.22s",  0.5100),
    ("11h51m07.52s",  "+71d41m44.3s",   0.2710),
    ("05h52m45.12s",  "+74d12m01.7s",   1.0000),
    ("20h20m08.92s",  "+70d47m33.96s",  0.3024),
    ("02h39m03.96s",  "+71d01m04.3s",   1.0000),
    ("20h50m28.59s",  "+73d54m00.0s",   0.0740),
    ("11h19m56.05s",  "+70d40m34.4s",   0.2870),
    ("22h23m53.94s",  "+73d01m33.26s",  1.0000),
]

# ----------------------- helper: aperture in arcmin ------------------------
def theta_proper(z):
    """Proper-radius (kpc) → angle (arcmin) at redshift z."""
    D_A = cosmo.angular_diameter_distance(z)          # Mpc
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        return (R_PHYS.to(u.Mpc) / D_A).to(u.arcmin)

# ------------------------------ retry wrapper ------------------------------
def _retry(fn, *args, **kw):
    for attempt in range(MAX_TRIES):
        try:
            return fn(*args, **kw)
        except (ConnectionError, ReadTimeout) as e:
            if attempt == MAX_TRIES - 1:
                raise
            wait = BASE_DELAY * 2**attempt
            print(f"   network glitch ({e}); retrying in {wait}s …")
            time.sleep(wait)

# ------------------------- NED: best redshift per object -------------------
def _best_redshift(name):
    try:
        ztbl = Ned.get_table(name, table="Redshift")
    except Exception:
        return None
    if len(ztbl) == 0:
        return None
    z    = ztbl["Redshift"]
    kind = ztbl["Velocity / Redshift Flag"]
    spec = z[(kind == "SPEC") & (~z.mask)]
    if len(spec):
        return float(spec[0])
    phot = z[(kind == "PHOTO") & (~z.mask)]
    return float(phot[0]) if len(phot) else None

# ----------------------- add projected separation column -------------------
def _add_proj(tab, ra_col, dec_col, coord, z):
    """
    Append column 'Rproj_kpc' = projected distance (proper) between
    each row (RA,DEC) and `coord` at redshift `z`.

    • Works even if RA/DEC already carry a (possibly unrecognised) unit tag.
    • Rows with NaN / masked coordinates get Rproj_kpc = NaN.
    """
    # 1. pull pure float arrays, preserving masked→nan
    ra_vals  = np.array(tab[ra_col], dtype=float)
    dec_vals = np.array(tab[dec_col], dtype=float)
    good = np.isfinite(ra_vals) & np.isfinite(dec_vals)

    Rproj = np.full(len(tab), np.nan)
    if z is not None and good.any():
        tgt   = SkyCoord(ra_vals[good], dec_vals[good], unit="deg")
        theta = tgt.separation(coord).to(u.rad).value      # radians
        D_A   = cosmo.angular_diameter_distance(z).to(u.kpc).value
        Rproj[good] = theta * D_A

    tab["Rproj_kpc"] = Rproj
    return tab

# ---------------------------- catalogue queries ----------------------------
def q_ned(coord, radius, z):
    t = _retry(Ned.query_region, coord, radius=radius)
    mask = [str(x).startswith("G") for x in t["Type"].filled("")]
    t = t[mask]
    if len(t):
        t["z_best"] = [_best_redshift(n) for n in t["Object Name"]]
        t = _add_proj(t, "RA", "DEC", coord, z)
    return t

def q_ps1(coord, radius, z):
    try:
        t = _retry(Catalogs.query_region, coord, radius=radius,
                   catalog="Panstarrs", table="mean", data_release="dr2")
    except Exception as e:
        print(f"   Pan-STARRS failed: {e}")
        return None
    if t is None or len(t) == 0:
        return None
    # extended-source bit
    mask_ext = (t["objInfoFlag"] & 0x400000000000) > 0

    def _psf_kron(b):
        psf  = next((c for c in t.colnames if c.lower()==f"{b}meanpsfmag"),  None)
        kron = next((c for c in t.colnames if c.lower()==f"{b}meankronmag"),None)
        if psf and kron:
            return (t[psf].filled(99) - t[kron].filled(99)) > 0.05
        return np.zeros(len(t), bool)

    mask = mask_ext | _psf_kron("g") | _psf_kron("r") | _psf_kron("i")
    t = t[mask]
    if len(t):
        t = _add_proj(t, "raMean", "decMean", coord, z)
    return t

def q_sdss(coord, radius, z):
    r = min(radius.to(u.arcmin).value, 2.99)
    sql = f"""
        SELECT TOP 50 s.objid, s.ra, s.dec, s.z, g.stellarMass
        FROM dbo.fGetNearbyObjEq({coord.ra.deg},{coord.dec.deg},{r}) AS nb
        JOIN SpecObj AS s   ON s.objid = nb.objid
        LEFT JOIN galSpecExtra AS g ON g.specobjid = s.specobjid
    """
    try:
        t = _retry(SDSS.query_sql, sql, timeout=90)
    except Exception as e:
        print(f"   SDSS SQL failed: {e}")
        return None
    if len(t):
        t = _add_proj(t, "ra", "dec", coord, z)
    return t

def q_desi(coord, radius, z):
    Vizier.ROW_LIMIT = -1
    rows = _retry(Vizier.query_region, coord, radius=radius,
                  catalog="VII/378/desi_dr1")
    if not rows:
        return None
    t = rows[0]
    if len(t):
        t = _add_proj(t, "RA", "DEC", coord, z)
    return t

# ----------------------- standardise to common columns ---------------------
def _std(src, mapping, cat):
    """
    Return a new table with columns:
        name, ra, dec, z, Mstar, Rproj_kpc, catalog
    `mapping` is {new_name : old_name or None}.
    Any missing column becomes an array of NaNs.
    """
    if src is None or len(src) == 0:
        return None

    n = len(src)
    cols = {}
    for new, old in mapping.items():
        if old and old in src.colnames:
            cols[new] = src[old]
        else:                        # make a NaN column of correct length
            cols[new] = np.full(n, np.nan)

    cols["catalog"] = np.full(n, cat)      # same length as others
    return Table(cols)

def std_ned(t):  return _std(t,
        {"name":"Object Name","ra":"RA","dec":"DEC","z":"z_best",
         "Mstar":None,"Rproj_kpc":"Rproj_kpc"}, "NED")

def std_ps1(t):  return _std(t,
        {"name":"objName","ra":"raMean","dec":"decMean","z":None,
         "Mstar":None,"Rproj_kpc":"Rproj_kpc"}, "PS1")

def std_sdss(t): return _std(t,
        {"name":"objid","ra":"ra","dec":"dec","z":"z",
         "Mstar":"stellarMass","Rproj_kpc":"Rproj_kpc"}, "SDSS")

def std_desi(t): return _std(t,
        {"name":"TARGETID","ra":"RA","dec":"DEC","z":"Z",
         "Mstar":"MASS_BEST","Rproj_kpc":"Rproj_kpc"}, "DESI")

STD_FUNCS = {"NED":std_ned, "PS1":std_ps1, "SDSS":std_sdss, "DESI":std_desi}
QUERY_FUNCS = {"NED":q_ned, "PS1":q_ps1, "SDSS":q_sdss, "DESI":q_desi}

# ------------------------------ MAIN LOOP ----------------------------------
sheets = {}          # {sheet_name : pandas DF}

try:
    for tid, (ra_s, dec_s, z_t) in enumerate(targets, 1):
        coord  = SkyCoord(ra_s, dec_s, unit=(u.hourangle,u.deg))
        radius = theta_proper(z_t)
        print(f"[{tid:02d}] {coord.to_string('hmsdms'):>22s}  "
              f"z={z_t:.3f}  θ={radius:.2f}")

        for cat in ("NED","PS1","SDSS","DESI"):
            raw = QUERY_FUNCS[cat](coord, radius, z_t)
            time.sleep(PAUSE)
            tab = STD_FUNCS[cat](raw)
            if tab is None or len(tab)==0:
                continue
            sheets[f"T{tid:02d}_{cat}"] = tab.to_pandas()
            print(f"   ↳ {cat:<4s}: {len(tab):3d} rows")

        print()

    # --------------------- WRITE EXCEL WORKBOOK ---------------------------
    if sheets:
        with pd.ExcelWriter(OUTFILE, engine="openpyxl") as xl:
            for name, df in sheets.items():
                df.to_excel(xl, sheet_name=name[:31], index=False)
        print(f"Results written to “{OUTFILE.name}”")
    else:
        print("No catalogue returned any galaxy within 100 kpc.")

except KeyboardInterrupt:
    print("\nInterrupted — writing collected data …")
    if sheets:
        with pd.ExcelWriter(OUTFILE, engine="openpyxl") as xl:
            for name, df in sheets.items():
                df.to_excel(xl, sheet_name=name[:31], index=False)
        print(f"Partial results written to “{OUTFILE.name}”")
    sys.exit(130)
