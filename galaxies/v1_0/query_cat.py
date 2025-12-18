#!/usr/bin/env python
"""
Query NED, Pan-STARRS DR2, SDSS DR17, DESI DR1 for galaxies within
<=100 kpc proper of each foreground position.

Output: Excel workbook (or CSV fallback) with one sheet per (target, catalogue)
containing columns: name | ra | dec | z | Mstar | Rproj_kpc | catalog
"""
from __future__ import annotations

import sys
import time
import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from requests.exceptions import ConnectionError as RequestsConnectionError, ReadTimeout

# astroquery back-ends
from astroquery.ipac.ned import Ned
from astroquery.mast import Catalogs
from astroquery.sdss import SDSS
from astroquery.vizier import Vizier

# Shared configuration
from config import (
    TARGETS_TUPLE as targets,
    TARGETS,
    R_PHYS, R_PHYS_KPC,
    MAX_TRIES, BASE_DELAY, PAUSE,
    NED_TIMEOUT, MAX_WORKERS,
    CACHE_DIR, CACHE_EXPIRY_HOURS,
    angular_diameter_distance_fast,
    theta_for_impact,
    filter_targets,
)

# ------------------------------- CONFIG ------------------------------------
Ned.TIMEOUT = NED_TIMEOUT
OUTFILE = Path(f"galaxies_{R_PHYS_KPC}kpc_proper.xlsx")

np.seterr(invalid="ignore")
warnings.filterwarnings("ignore",
        message="Field info are not available for this data release",
        module="astroquery.sdss")

def debug_table(table, name="Table"):
    """Print table info for debugging"""
    if table is None:
        print(f"\n{name} DEBUG: Table is None")
        return
        
    print(f"\n{name} DEBUG:")
    print(f"  Columns: {table.colnames}")
    print(f"  Length: {len(table)}")
    for col in table.colnames[:5]:  # First 5 columns
        print(f"  {col}: type={type(table[col])}, shape={table[col].shape}")
        if hasattr(table[col], 'mask'):
            print(f"    (masked array, {np.sum(table[col].mask)} masked values)")

# ----------------------- helper: aperture in arcmin ------------------------
def theta_proper(z):
    D_A = cosmo.angular_diameter_distance(z)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        return (R_PHYS.to(u.Mpc) / D_A).to(u.arcmin)

# ------------------------------ retry wrapper ------------------------------
def _retry(fn, *args, **kw):
    for attempt in range(MAX_TRIES):
        try:
            return fn(*args, **kw)
        except (RequestsConnectionError, ReadTimeout):
            if attempt == MAX_TRIES - 1:
                raise
            time.sleep(BASE_DELAY * 2**attempt)

# ------------------------- caching helpers ---------------------------------
def _cache_key(prefix: str, *args) -> Path:
    """Generate a cache file path based on query parameters."""
    key = hashlib.md5(f"{prefix}:{args}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{prefix}_{key}.json"

def _load_cache(cache_path: Path):
    """Load cached result if exists and not expired (24h)."""
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    return None

def _save_cache(cache_path: Path, data):
    """Save result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except (IOError, TypeError):
        pass  # silently skip if not JSON-serializable

# ------------------------- best redshift per NED obj -----------------------
def _best_redshift(name):
    """Get best redshift for a NED object, with caching."""
    cache_path = _cache_key("ned_z", name)
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached.get("z")
    
    try:
        ztbl = Ned.get_table(name, table="Redshift")
    except Exception:
        _save_cache(cache_path, {"z": None})
        return None
    if len(ztbl) == 0:
        _save_cache(cache_path, {"z": None})
        return None
    z    = ztbl["Redshift"]; kind = ztbl["Velocity / Redshift Flag"]
    spec = z[(kind == "SPEC") & (~z.mask)]
    if len(spec):
        result = float(spec[0])
        _save_cache(cache_path, {"z": result})
        return result
    phot = z[(kind == "PHOTO") & (~z.mask)]
    result = float(phot[0]) if len(phot) else None
    _save_cache(cache_path, {"z": result})
    return result

def _get_redshifts_parallel(names):
    """Fetch redshifts for multiple NED objects in parallel."""
    results = [None] * len(names)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(_best_redshift, name): i 
                         for i, name in enumerate(names)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None
    return results

def estimate_ps1_photoz(ps1_table):
    """Rough photo-z estimate from PS1 colors using empirical relations"""
    # Get magnitude columns
    g = ps1_table['gMeanKronMag'] if 'gMeanKronMag' in ps1_table.colnames else None
    r = ps1_table['rMeanKronMag'] if 'rMeanKronMag' in ps1_table.colnames else None
    i = ps1_table['iMeanKronMag'] if 'iMeanKronMag' in ps1_table.colnames else None
    z = ps1_table['zMeanKronMag'] if 'zMeanKronMag' in ps1_table.colnames else None
    
    if g is None or r is None or i is None or z is None:
        ps1_table['z_color_est'] = np.nan
        return ps1_table
    
    # Handle masked arrays
    if hasattr(g, 'filled'):
        g = g.filled(np.nan)
        r = r.filled(np.nan)
        i = i.filled(np.nan)
        z = z.filled(np.nan)
    else:
        g = np.array(g, dtype=float)
        r = np.array(r, dtype=float)
        i = np.array(i, dtype=float)
        z = np.array(z, dtype=float)
    
    # Simple color-redshift relation (very approximate!)
    gr = g - r
    ri = r - i
    iz = i - z
    
    # Rough estimate
    z_est = 0.3 * gr + 0.2 * ri + 0.1 * iz - 0.2
    
    # Clip to reasonable range
    z_est = np.clip(z_est, 0.0, 2.0)
    
    # Only use where all colors are available
    mask = np.isfinite(gr) & np.isfinite(ri) & np.isfinite(iz)
    
    ps1_table['z_color_est'] = np.full(len(ps1_table), np.nan)
    ps1_table['z_color_est'][mask] = z_est[mask]
    
    return ps1_table

# --------------- add projected separation (RA/DEC name-agnostic) ----------
def _add_proj(tab, coord, z):
    # Find RA/Dec columns (case-insensitive)
    ra_col = next((c for c in tab.colnames if c.lower() in ['ra', 'ramean']), None)
    dec_col = next((c for c in tab.colnames if c.lower() in ['dec', 'decmean']), None)
    
    if ra_col is None or dec_col is None:
        print(f"     Warning: Could not find RA/Dec columns in {tab.colnames}")
        tab["Rproj_kpc"] = np.full(len(tab), np.nan)
        return tab
        
    # Convert to numpy arrays and handle units
    ra_data = tab[ra_col]
    dec_data = tab[dec_col]
    
    # Handle potential astropy Quantity objects with units
    if hasattr(ra_data, 'value'):
        ra = ra_data.value
    elif hasattr(ra_data, 'filled'):
        ra = ra_data.filled(np.nan)
    else:
        ra = np.array(ra_data, dtype=float)
        
    if hasattr(dec_data, 'value'):
        dec = dec_data.value
    elif hasattr(dec_data, 'filled'):
        dec = dec_data.filled(np.nan)
    else:
        dec = np.array(dec_data, dtype=float)
    
    good = np.isfinite(ra) & np.isfinite(dec)
    proj = np.full(len(tab), np.nan)
    
    if good.any() and z is not None:
        try:
            tgt = SkyCoord(ra[good], dec[good], unit="deg")
            theta = tgt.separation(coord).to(u.rad).value
            proj[good] = theta * cosmo.angular_diameter_distance(z).to(u.kpc).value
        except Exception as e:
            print(f"     Warning: Error calculating projections: {e}")
            
    tab["Rproj_kpc"] = proj
    return tab

# ---------------------------- catalogue queries ----------------------------
def q_ned(coord, radius, z):
    try:
        t = _retry(Ned.query_region, coord, radius=radius)
        print(f"     NED returned {len(t)} total objects")
        
        # Filter for galaxies - handle the Type column properly
        if 'Type' in t.colnames:
            # Handle both masked and unmasked arrays
            types = t['Type']
            if hasattr(types, 'filled'):
                types = types.filled("")
            galaxy_mask = [str(x).startswith("G") for x in types]
            t = t[galaxy_mask]
        
        print(f"     NED filtered to {len(t)} galaxies")
        
        if len(t) > 0:
            print(f"     Getting NED redshifts for {len(t)} galaxies (parallel)...")
            t["z_best"] = _get_redshifts_parallel(list(t["Object Name"]))
            
            # Convert z_best to a proper numpy array, replacing None with nan
            z_best_array = np.array([z if z is not None else np.nan for z in t["z_best"]])
            t["z_best"] = z_best_array
            
            # Now count non-nan values
            n_with_z = np.sum(~np.isnan(z_best_array))
            print(f"     Found {n_with_z} galaxies with redshifts")
            
            t = _add_proj(t, coord, z)
        return t
    except Exception as e:
        print(f"   NED failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def q_ps1(coord, radius, z):
    try:
        t = _retry(Catalogs.query_region, coord, radius=radius,
                   catalog="Panstarrs", table="mean", data_release="dr2")
    except Exception as e:
        print(f"   PS1 failed: {e}")
        return None
    
    if t is None or len(t) == 0:
        return None
    
    # Filter for extended sources
    mask_ext = (t['objInfoFlag'] & 0x400000000000) > 0
    def _pk(b):
        psf = f"{b}MeanPSFMag"
        kron = f"{b}MeanKronMag"
        if psf in t.colnames and kron in t.colnames:
            return (t[psf].filled(99) - t[kron].filled(99) > 0.05)
        return np.zeros(len(t), bool)
    
    t = t[mask_ext | _pk("g") | _pk("r") | _pk("i")]
    
    if len(t) > 0:
        print(f"     PS1 returned {len(t)} extended sources")
        
        # Add photometric redshifts
        t = add_photoz_to_ps1(t, coord, z)
        
        # Add projected separation
        t = _add_proj(t, coord, z)
    
    return t

def crossmatch_ps1_sdss(ps1_table, coord, z):
    """Cross-match PS1 objects with SDSS to get photometric redshifts"""
    if ps1_table is None or len(ps1_table) == 0:
        return ps1_table
    
    # Initialize columns
    n = len(ps1_table)
    ps1_table['z_spec'] = np.full(n, np.nan)
    ps1_table['z_phot'] = np.full(n, np.nan)
    ps1_table['z_best'] = np.full(n, np.nan)
    
    # Get PS1 coordinates
    ps1_coords = SkyCoord(ps1_table['raMean'], ps1_table['decMean'], unit='deg')
    
    # Try cone search first
    try:
        radius = theta_proper(z)
        sdss_data = SDSS.query_region(coord, radius=radius, 
                                    spectro=False,
                                    photoobj_fields=['objid', 'ra', 'dec', 'type', 'z', 'zErr'])
        
        if sdss_data is not None and len(sdss_data) > 0:
            # Check for HTML error
            if '<html>' in str(sdss_data.colnames[0]).lower():
                print("       SDSS cross-match returned HTML, skipping")
                return ps1_table
                
            print(f"       Found {len(sdss_data)} SDSS objects for cross-match")
            
            # Filter galaxies
            galaxies = sdss_data[sdss_data['type'] == 3]
            
            if len(galaxies) > 0:
                # Cross-match
                sdss_coords = SkyCoord(galaxies['ra'], galaxies['dec'], unit='deg')
                idx, d2d, _ = ps1_coords.match_to_catalog_sky(sdss_coords)
                
                mask = d2d < 1*u.arcsec
                if 'z' in galaxies.colnames:
                    ps1_table['z_phot'][mask] = galaxies['z'][idx[mask]]
                    ps1_table['z_best'][mask] = galaxies['z'][idx[mask]]
                
                n_matched = np.sum(mask)
                n_with_z = np.sum(~np.isnan(ps1_table['z_best']))
                print(f"       PS1-SDSS cross-match: {n_matched} matches, {n_with_z} with redshifts")
                
    except Exception as e:
        print(f"       PS1-SDSS cross-match failed: {e}")
    
    return ps1_table

def q_sdss(coord, radius, z):
    r = radius.to(u.arcmin).value
    
    # First try simple cone search
    try:
        print(f"     Querying SDSS with radius {r:.2f} arcmin...")
        photoobj = SDSS.query_region(coord, radius=radius, 
                                   spectro=False,  # Get photometric objects
                                   photoobj_fields=['objid', 'ra', 'dec', 'type', 'z', 'zErr', 'petroMag_r'])
        
        if photoobj is not None and len(photoobj) > 0:
            # Check if we got HTML (error page)
            if '<html>' in str(photoobj.colnames[0]).lower():
                print("     SDSS returned HTML error page, trying SQL query...")
                raise Exception("HTML response")
                
            # Filter for galaxies (type = 3)
            galaxies = photoobj[photoobj['type'] == 3]
            
            if len(galaxies) > 0:
                print(f"     SDSS found {len(galaxies)} galaxies via cone search")
                galaxies['z_best'] = galaxies['z'] if 'z' in galaxies.colnames else np.nan
                galaxies = _add_proj(galaxies, coord, z)
                return galaxies
    except Exception as e:
        print(f"     SDSS cone search failed: {e}")
    
    # Fall back to SQL query
    sql = f"""
      SELECT TOP 500 
             p.objid, p.ra, p.dec, p.type, p.petroMag_r,
             p.z AS photo_z, p.zErr AS photo_z_err
      FROM PhotoObjAll AS p
      WHERE 
        p.ra BETWEEN {coord.ra.deg - r/60} AND {coord.ra.deg + r/60}
        AND p.dec BETWEEN {coord.dec.deg - r/60} AND {coord.dec.deg + r/60}
        AND p.type = 3
        AND dbo.fDistanceArcMinEq(p.ra, p.dec, {coord.ra.deg}, {coord.dec.deg}) <= {r}
    """
    
    try:
        t = _retry(SDSS.query_sql, sql, timeout=180)
        
        if t is not None and len(t) > 0:
            # Check for HTML response
            if '<html>' in str(t.colnames[0]).lower():
                print("     SDSS SQL query returned HTML error page")
                return None
                
            print(f"     SDSS SQL query returned {len(t)} galaxies")
            
            # Add z_best column
            if 'photo_z' in t.colnames:
                t['z_best'] = t['photo_z']
            else:
                t['z_best'] = np.nan
                
            t = _add_proj(t, coord, z)
            return t
            
    except Exception as e:
        print(f"   SDSS SQL query failed: {e}")
        
    return None

def q_desi(coord, radius, z):
    Vizier.ROW_LIMIT = -1
    
    # Try multiple DESI catalogs with correct IDs
    catalogs = [
        "VII/292",           # DESI Early Data Release
        "J/AJ/165/144",      # DESI overview
        "J/AJ/164/207",      # DESI Legacy Survey
    ]
    
    for cat in catalogs:
        try:
            print(f"     Trying DESI catalog {cat}...")
            result = Vizier.query_region(coord, radius=radius, catalog=cat)
            
            if result and len(result) > 0:
                # Try each table in the result
                for t in result:
                    print(f"       Found table with columns: {t.colnames[:10]}...")  # First 10 cols
                    
                    # Look for redshift columns
                    z_col = None
                    for col in ['Z', 'z', 'z_spec', 'Z_SPEC', 'ZBEST', 'zphoto', 'zspec']:
                        if col in t.colnames:
                            z_col = col
                            break
                    
                    if z_col:
                        print(f"       Using redshift column: {z_col}")
                        t['z_desi'] = t[z_col]
                        t = _add_proj(t, coord, z)
                        return t
                        
        except Exception as e:
            print(f"       {cat} failed: {e}")
            continue
    
    print("     No DESI data found")
    return None

def get_legacy_survey_photoz(coord, radius):
    """Query Legacy Survey DR10 for photometric redshifts"""
    Vizier.ROW_LIMIT = -1
    
    # Legacy Survey DR10 catalog
    catalogs = [
        "II/371/des-dr2",  # DES DR2 with photo-z
        "II/368/ls-dr9",   # Legacy Survey DR9
    ]
    
    for cat in catalogs:
        try:
            result = Vizier.query_region(coord, radius=radius, catalog=cat)
            if result and len(result) > 0:
                t = result[0]
                # Look for photo-z columns
                for col in ['z_phot_mean', 'z_phot_median', 'photo_z', 'photoz']:
                    if col in t.colnames:
                        return t
        except Exception as e:
            print(f"       Legacy Survey query failed for {cat}: {e}")
            continue
    
    return None

def crossmatch_ps1_legacy(ps1_table, coord, z):
    """Cross-match PS1 with Legacy Survey for photo-z"""
    if ps1_table is None or len(ps1_table) == 0:
        return ps1_table
    
    # Get Legacy Survey data
    radius = theta_proper(z)
    legacy_data = get_legacy_survey_photoz(coord, radius)
    
    if legacy_data is None:
        ps1_table['z_phot'] = np.nan
        return ps1_table
    
    # Cross-match
    ps1_coords = SkyCoord(ps1_table['raMean'], ps1_table['decMean'], unit='deg')
    
    # Get Legacy Survey coordinates (column names vary)
    ra_col = 'RA' if 'RA' in legacy_data.colnames else 'RAJ2000'
    dec_col = 'DEC' if 'DEC' in legacy_data.colnames else 'DEJ2000'
    legacy_coords = SkyCoord(legacy_data[ra_col], legacy_data[dec_col], unit='deg')
    
    idx, d2d, _ = ps1_coords.match_to_catalog_sky(legacy_coords)
    
    # Add photo-z where match is < 0.5 arcsec (tighter for better accuracy)
    ps1_table['z_phot_legacy'] = np.nan
    mask = d2d < 0.5*u.arcsec
    
    # Find photo-z column
    z_col = None
    for col in ['z_phot_mean', 'z_phot_median', 'photo_z', 'photoz']:
        if col in legacy_data.colnames:
            z_col = col
            break
    
    if z_col:
        ps1_table['z_phot_legacy'][mask] = legacy_data[z_col][idx[mask]]
    
    return ps1_table

def add_photoz_to_ps1(ps1_table, coord, z):
    """Add photometric redshifts from multiple catalogs"""
    if ps1_table is None or len(ps1_table) == 0:
        return ps1_table
    
    # Initialize columns as numpy arrays
    n = len(ps1_table)
    ps1_table['z_spec'] = np.full(n, np.nan)
    ps1_table['z_phot_sdss'] = np.full(n, np.nan)
    ps1_table['z_phot_legacy'] = np.full(n, np.nan)
    ps1_table['z_phot_wise'] = np.full(n, np.nan)
    
    ps1_coords = SkyCoord(ps1_table['raMean'], ps1_table['decMean'], unit='deg')
    
    # 1. Try SDSS spectroscopic and photometric
    print("     Cross-matching with SDSS...")
    ps1_table = crossmatch_ps1_sdss(ps1_table, coord, z)
    
    # 2. Try Legacy Survey
    print("     Cross-matching with Legacy Survey...")
    ps1_table = crossmatch_ps1_legacy(ps1_table, coord, z)
    
    # 3. Try WISE photo-z catalog
    print("     Cross-matching with WISE...")
    try:
        Vizier.ROW_LIMIT = -1
        wise_result = Vizier.query_region(coord, radius=theta_proper(z), 
                                         catalog="J/ApJS/234/23/galaxies")
        if wise_result and len(wise_result) > 0:
            wise_data = wise_result[0]
            wise_coords = SkyCoord(wise_data['RAJ2000'], wise_data['DEJ2000'], unit='deg')
            idx, d2d, _ = ps1_coords.match_to_catalog_sky(wise_coords)
            mask = d2d < 1*u.arcsec
            if 'zphot' in wise_data.colnames:
                ps1_table['z_phot_wise'][mask] = wise_data['zphot'][idx[mask]]
    except Exception as e:
        print(f"       WISE cross-match failed: {e}")
    
    # 4. Combine all redshifts with priority: spec > SDSS photo > Legacy > WISE
    ps1_table['z_best'] = np.full(n, np.nan)
    
    # Fill in order of preference (reverse order so higher priority overwrites)
    for z_col in ['z_phot_wise', 'z_phot_legacy', 'z_phot_sdss', 'z_spec']:
        if z_col in ps1_table.colnames:
            z_data = ps1_table[z_col]
            if hasattr(z_data, 'filled'):
                z_data = z_data.filled(np.nan)
            mask = np.isnan(ps1_table['z_best']) & ~np.isnan(z_data)
            ps1_table['z_best'][mask] = z_data[mask]
    
    # Report statistics
    n_spec = np.sum(~np.isnan(ps1_table['z_spec']))
    n_phot = np.sum(~np.isnan(ps1_table['z_best'])) - n_spec
    print(f"     PS1 redshifts: {n_spec} spectroscopic, {n_phot} photometric")
    
    return ps1_table

# -------- standardise to common column set ---------------------------------
def _std(src, mapping, cat):
    if src is None or len(src) == 0:
        return None
    
    n = len(src)
    cols = {}
    
    for new, old in mapping.items():
        if old and old in src.colnames:
            # Handle masked arrays properly
            col_data = src[old]
            if hasattr(col_data, 'filled'):
                cols[new] = col_data.filled(np.nan)
            else:
                # Convert to numpy array to ensure consistent type
                cols[new] = np.array(col_data)
        else:
            cols[new] = np.full(n, np.nan)
    
    cols["catalog"] = np.full(n, cat)
    return Table(cols)

std_ned  = lambda t: _std(t, {"name":"Object Name","ra":"RA","dec":"DEC",
                              "z":"z_best","Mstar":None,"Rproj_kpc":"Rproj_kpc"},
                              "NED")
std_ps1 = lambda t: _std(t, {
    "name": "objName",
    "ra": "raMean",
    "dec": "decMean",
    "z": "z_best",  # Now includes photo-z
    "Mstar": None,
    "Rproj_kpc": "Rproj_kpc"
}, "PS1")

std_sdss = lambda t: _std(t, {
    "name": "objid",
    "ra": "ra", 
    "dec": "dec",
    "z": "z_best",  # Now using our combined redshift
    "Mstar": None,
    "Rproj_kpc": "Rproj_kpc"
}, "SDSS")

std_desi = lambda t: _std(t, {
    "name": "TARGETID" if t is not None and "TARGETID" in t.colnames else ("objID" if t is not None and "objID" in t.colnames else None),
    "ra": "RAJ2000" if t is not None and "RAJ2000" in t.colnames else ("RA" if t is not None and "RA" in t.colnames else None),
    "dec": "DEJ2000" if t is not None and "DEJ2000" in t.colnames else ("DEC" if t is not None and "DEC" in t.colnames else None),
    "z": "z_desi" if t is not None and "z_desi" in t.colnames else None,
    "Mstar": None,
    "Rproj_kpc": "Rproj_kpc" if t is not None else None
}, "DESI") if t is not None else None

QUERY_FUNCS = {"NED":q_ned,"PS1":q_ps1,"SDSS":q_sdss,"DESI":q_desi}
STD_FUNCS   = {"NED":std_ned,"PS1":std_ps1,"SDSS":std_sdss,"DESI":std_desi}

# -------------------------------- MAIN -------------------------------------
def main(target_indices=None):
    """Run catalog queries for specified targets (or all if None)."""
    sheets = {}
    target_list = targets if target_indices is None else [targets[i-1] for i in target_indices]
    
    try:
        for tid, (ra_s, dec_s, z_t) in enumerate(target_list, 1):
            coord = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))
            radius = theta_proper(z_t)
            print(f"[{tid:02d}] {coord.to_string('hmsdms'):>22s}  z={z_t:.3f}  θ={radius:.2f}")
            for cat in ("NED", "PS1", "SDSS", "DESI"):
                raw = QUERY_FUNCS[cat](coord, radius, z_t)
                time.sleep(PAUSE)
                tab = STD_FUNCS[cat](raw)
                debug_table(tab, name=f"{cat} Table")
                if tab is None or len(tab) == 0:
                    continue
                sheets[f"T{tid:02d}_{cat}"] = tab.to_pandas()
                print(f"   ↳ {cat:<4s}: {len(tab):3d} rows")
            print()
        if sheets:
            try:
                with pd.ExcelWriter(OUTFILE, engine="openpyxl") as xl:
                    for name, df in sheets.items():
                        df.to_excel(xl, sheet_name=name[:31], index=False)
                print(f"Results written to '{OUTFILE.name}'")
            except ImportError:
                # Fall back to CSV if openpyxl not installed
                csv_dir = OUTFILE.with_suffix("")
                csv_dir.mkdir(exist_ok=True)
                for name, df in sheets.items():
                    df.to_csv(csv_dir / f"{name}.csv", index=False)
                print(f"Results written to '{csv_dir}/' (CSV fallback)")
        else:
            print("No catalogue returned any galaxy within 100 kpc.")
    except KeyboardInterrupt:
        print("\nInterrupted - writing collected data...")
        if sheets:
            try:
                with pd.ExcelWriter(OUTFILE, engine="openpyxl") as xl:
                    for name, df in sheets.items():
                        df.to_excel(xl, sheet_name=name[:31], index=False)
                print(f"Partial results written to '{OUTFILE.name}'")
            except ImportError:
                csv_dir = OUTFILE.with_suffix("")
                csv_dir.mkdir(exist_ok=True)
                for name, df in sheets.items():
                    df.to_csv(csv_dir / f"{name}.csv", index=False)
                print(f"Partial results written to '{csv_dir}/' (CSV fallback)")
        sys.exit(130)
    
    return sheets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query galaxy catalogs for FRB host candidates")
    parser.add_argument("--targets", type=int, nargs="+", 
                        help="Target indices to query (1-indexed), default: all")
    args = parser.parse_args()
    main(target_indices=args.targets)
