"""Query engines for different galaxy catalogs."""

import os
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier

from .config import COSMO


def _retry(fn, attempts: int = 3, base_delay: float = 2.0):
    """Call ``fn`` with exponential backoff on transient errors.

    CDS Vizier intermittently times out / resets under load; without a retry a
    single transient failure silently degrades a catalog to empty (observed:
    GLADE+/PS1/DESI dropping out mid-run). Retries only re-raise after the last
    attempt, so a genuine error still surfaces to the caller's degrade path.
    """
    for k in range(attempts):
        try:
            return fn()
        except Exception:
            if k == attempts - 1:
                raise
            time.sleep(base_delay * (2**k))


class BaseEngine(ABC):
    """Abstract base class for catalog query engines."""

    @abstractmethod
    def query(self, coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
        pass


class NedEngine(BaseEngine):
    """Engine for querying NASA/IPAC Extragalactic Database (NED)."""

    def __init__(self):
        # NED can hang for minutes during outages/large cones; FLITS_NED_TIMEOUT
        # lets a batch run fail fast (and degrade to empty) when NED is down.
        Ned.TIMEOUT = int(os.environ.get("FLITS_NED_TIMEOUT", "180"))

    def query(self, coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
        try:
            result_table = Ned.query_region(coord, radius=radius)
            if result_table is None or len(result_table) == 0:
                return pd.DataFrame()

            df = result_table.to_pandas()
            # Standardize columns. Keep NED's object Type (e.g. 'GClstr', 'GGroup')
            # as 'classification' so the search can apply the cluster impact threshold.
            df = df.rename(
                columns={
                    "Object Name": "name",
                    "RA": "ra",
                    "DEC": "dec",
                    "Redshift": "z",
                    "Type": "classification",
                }
            )
            df["catalog"] = "NED"
            keep = ["name", "ra", "dec", "z", "catalog"]
            if "classification" in df.columns:
                keep.insert(4, "classification")
            return df[keep]
        except Exception as e:
            print(f"NED query failed: {e}")
            return pd.DataFrame()


class VizierEngine(BaseEngine):
    """Engine for querying Vizier catalogs (GLADE+, DESI, etc.)."""

    def __init__(self, catalog_id: str):
        self.catalog_id = catalog_id
        # A bounded timeout is essential: a stalled Vizier connection with no
        # timeout blocks the whole batch indefinitely (FLITS_VIZIER_TIMEOUT).
        self.vizier = Vizier(
            row_limit=-1, timeout=int(os.environ.get("FLITS_VIZIER_TIMEOUT", "120"))
        )

    def query(self, coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
        try:
            result = _retry(
                lambda: self.vizier.query_region(coord, radius=radius, catalog=self.catalog_id)
            )
            if not result:
                return pd.DataFrame()

            df = result[0].to_pandas()
            if df.empty:
                return df

            # print(f"DEBUG: Vizier {self.catalog_id} columns: {df.columns.tolist()}")

            # Standardize common Vizier column names
            rename_map = {
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "z": "z",
                "zphot": "z",  # DESI DR8 North
                "z_best": "z",  # GLADE+ best redshift
                "z_helio": "z",  # GLADE+ heliocentric redshift
                "z_phot": "z",
                "zph": "z",
                "z_cmb": "z",
                "zph2MPZ": "z",
            }
            # Case-insensitive mapping
            current_cols = {c.lower(): c for c in df.columns}
            final_rename = {}
            for target_low, target_std in rename_map.items():
                if target_low.lower() in current_cols:
                    final_rename[current_cols[target_low.lower()]] = target_std

            df = df.rename(columns=final_rename)
            df["catalog"] = self.catalog_id
            df = _add_desi_stellar_mass(df, self.catalog_id)
            return df
        except Exception as e:
            print(f"Vizier query ({self.catalog_id}) failed: {e}")
            return pd.DataFrame()


PS1_CATALOG = "II/349/ps1"
PS1_MATCH_RADIUS = 3.0 * u.arcsec


def query_ps1_gi_mags(coord: SkyCoord, match_radius: u.Quantity = PS1_MATCH_RADIUS):
    """Nearest Pan-STARRS1 (II/349/ps1) source to ``coord`` within ``match_radius``.

    Returns ``(g_mag, i_mag, sep_arcsec)``. Kron magnitudes are preferred over
    mean-PSF mags: Taylor+2011 (the g-i/M_i estimator these feed) was calibrated
    on Kron-like GAMA total magnitudes, and Kron better captures extended-source
    flux. PS1 AB mags are used directly; the small PS1->SDSS color terms are
    neglected at the order-of-magnitude precision of these halo-mass estimates.

    The DESI VII/292 photo-z catalog carries no photometry, so this PS1 match is
    the only route to a measured stellar mass for DESI-only sightlines. Faint,
    high-z galaxies fall below the shallow PS1 3pi depth and simply will not
    match; callers then fall back to an assumed mass. Either of ``g_mag`` /
    ``i_mag`` may be ``None`` if not finite, and the result is
    ``(None, None, None)`` on no match or query failure.
    """
    try:
        v = Vizier(columns=["RAJ2000", "DEJ2000", "gmag", "imag", "gKmag", "iKmag"], row_limit=50)
        res = _retry(lambda: v.query_region(coord, radius=match_radius, catalog=PS1_CATALOG))
    except Exception as e:
        print(f"PS1 query failed: {e}")
        return None, None, None

    if not res or len(res[0]) == 0:
        return None, None, None

    t = res[0]
    matches = SkyCoord(t["RAJ2000"], t["DEJ2000"], unit="deg")
    seps = coord.separation(matches).arcsec
    j = int(np.argmin(seps))

    def _pick(kron_col: str, psf_col: str):
        for col in (kron_col, psf_col):
            if col in t.colnames:
                val = t[col][j]
                if val is not None and np.isfinite(val):
                    return float(val)
        return None

    return _pick("gKmag", "gmag"), _pick("iKmag", "imag"), float(seps[j])


def _is_desi_dr8_north(catalog_id: str) -> bool:
    return catalog_id.lower() == "vii/292/north"


def _is_glade_plus(catalog_id: str) -> bool:
    return catalog_id.lower() == "vii/291/gladep"


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None


def _standardize_mass_column(df: pd.DataFrame) -> pd.DataFrame:
    mass_col = _find_column(df, ("M_star", "Mstar", "logMstar", "logM_star", "logMass", "logM"))
    if mass_col is not None and mass_col != "M_star":
        df = df.rename(columns={mass_col: "M_star"})
    return df


def _add_desi_stellar_mass(df: pd.DataFrame, catalog_id: str) -> pd.DataFrame:
    df = _standardize_mass_column(df)

    if _is_glade_plus(catalog_id):
        # gladep carries redshift as zcmb/zhelio (no literal 'z') and stellar mass
        # as M* in 1e10 Msun (linear) — convert to the log10(M/Msun) the pipeline
        # expects (the old glade table's M_star was already log).
        if "z" not in df.columns:
            for z_col in ("zcmb", "zhelio"):
                if z_col in df.columns:
                    df["z"] = pd.to_numeric(df[z_col], errors="coerce")
                    break
        if "M_star" not in df.columns and "M*" in df.columns:
            m_lin = pd.to_numeric(df["M*"], errors="coerce")
            df["M_star"] = np.log10(m_lin.where(m_lin > 0.0) * 1.0e10)
        if "M_star" in df.columns:
            df["mass_source"] = "catalog"
        return df

    if not _is_desi_dr8_north(catalog_id):
        return df

    g_col = _find_column(df, ("flux_g", "fluxg", "gflux", "g_flux", "flx_g"))
    r_col = _find_column(df, ("flux_r", "fluxr", "rflux", "r_flux", "flx_r"))
    z_col = _find_column(df, ("flux_z", "fluxz", "zflux", "z_flux", "flx_z"))
    if g_col is None or r_col is None or z_col is None or "z" not in df.columns:
        return df

    g_flux = pd.to_numeric(df[g_col], errors="coerce")
    r_flux = pd.to_numeric(df[r_col], errors="coerce")
    z_flux = pd.to_numeric(df[z_col], errors="coerce")
    redshift = pd.to_numeric(df["z"], errors="coerce")
    valid = (g_flux > 0.0) & (r_flux > 0.0) & (z_flux > 0.0) & (redshift > 0.0)
    if not valid.any():
        return df

    g_mag = 22.5 - 2.5 * np.log10(g_flux[valid])
    r_mag = 22.5 - 2.5 * np.log10(r_flux[valid])
    d_l_pc = COSMO.luminosity_distance(redshift[valid].to_numpy()).to(u.pc).value
    distance_modulus = 5.0 * np.log10(d_l_pc / 10.0)
    absolute_r = r_mag - distance_modulus
    log_mass = -0.68 + 0.70 * (g_mag - r_mag) - 0.4 * absolute_r

    df.loc[valid, "M_star"] = log_mass
    df.loc[valid, "mass_source"] = "photometric"
    return df
