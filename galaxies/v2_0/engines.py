"""Query engines for different galaxy catalogs."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier
from .config import COSMO

class BaseEngine(ABC):
    """Abstract base class for catalog query engines."""
    
    @abstractmethod
    def query(self, coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
        pass

class NedEngine(BaseEngine):
    """Engine for querying NASA/IPAC Extragalactic Database (NED)."""
    
    def __init__(self):
        Ned.TIMEOUT = 180

    def query(self, coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
        try:
            result_table = Ned.query_region(coord, radius=radius)
            if result_table is None or len(result_table) == 0:
                return pd.DataFrame()
            
            df = result_table.to_pandas()
            # Standardize columns
            df = df.rename(columns={
                'Object Name': 'name',
                'RA': 'ra',
                'DEC': 'dec',
                'Redshift': 'z'
            })
            df['catalog'] = 'NED'
            return df[['name', 'ra', 'dec', 'z', 'catalog']]
        except Exception as e:
            print(f"NED query failed: {e}")
            return pd.DataFrame()

class VizierEngine(BaseEngine):
    """Engine for querying Vizier catalogs (GLADE+, DESI, etc.)."""
    
    def __init__(self, catalog_id: str):
        self.catalog_id = catalog_id
        self.vizier = Vizier(row_limit=-1)
        
    def query(self, coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
        try:
            result = self.vizier.query_region(coord, radius=radius, catalog=self.catalog_id)
            if not result:
                return pd.DataFrame()
            
            df = result[0].to_pandas()
            if df.empty:
                return df
            
            # print(f"DEBUG: Vizier {self.catalog_id} columns: {df.columns.tolist()}")
            
            # Standardize common Vizier column names
            rename_map = {
                'RAJ2000': 'ra',
                'DEJ2000': 'dec',
                'z': 'z',
                'zphot': 'z',      # DESI DR8 North
                'z_best': 'z',     # GLADE+ best redshift
                'z_helio': 'z',    # GLADE+ heliocentric redshift
                'z_phot': 'z',
                'zph': 'z',
                'z_cmb': 'z',
                'zph2MPZ': 'z'
            }
            # Case-insensitive mapping
            current_cols = {c.lower(): c for c in df.columns}
            final_rename = {}
            for target_low, target_std in rename_map.items():
                if target_low.lower() in current_cols:
                    final_rename[current_cols[target_low.lower()]] = target_std
            
            df = df.rename(columns=final_rename)
            df['catalog'] = self.catalog_id
            df = _add_desi_stellar_mass(df, self.catalog_id)
            return df
        except Exception as e:
            print(f"Vizier query ({self.catalog_id}) failed: {e}")
            return pd.DataFrame()


def _is_desi_dr8_north(catalog_id: str) -> bool:
    return catalog_id.lower() == "vii/292/north"


def _is_glade_plus(catalog_id: str) -> bool:
    return catalog_id.lower() == "vii/291/glade"


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
