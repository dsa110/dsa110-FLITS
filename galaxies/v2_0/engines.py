"""Query engines for different galaxy catalogs."""

from abc import ABC, abstractmethod
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier
from .config import VIZIER_CATALOGS

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
            
            # Standardize common Vizier column names
            rename_map = {
                'RAJ2000': 'ra',
                'DEJ2000': 'dec',
                'z': 'z',
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
            return df
        except Exception as e:
            print(f"Vizier query ({self.catalog_id}) failed: {e}")
            return pd.DataFrame()
