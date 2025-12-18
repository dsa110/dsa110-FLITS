import sys
import os
from astropy import units as u
from astropy.coordinates import SkyCoord

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from galaxies.v2_0.config import TARGETS, VIZIER_CATALOGS
from galaxies.v2_0.utils import parse_coord, get_angular_radius
from galaxies.v2_0.engines import NedEngine, VizierEngine

def test_engines():
    # Test with M31 (Andromeda) to verify engines are working
    coord = SkyCoord(10.6847, 41.2687, unit='deg', frame='icrs')
    radius = 10.0 * u.arcmin
    
    print(f"Testing M31 with {radius} radius")
    
    ned = NedEngine()
    ned_df = ned.query(coord, radius)
    print(f"NED returned {len(ned_df)} rows")
    
    # Try AllWISE as a fallback/test
    wise = VizierEngine("II/328/allwise")
    wise_df = wise.query(coord, radius)
    print(f"AllWISE returned {len(wise_df)} rows")
    if not wise_df.empty:
        print("AllWISE columns:", wise_df.columns.tolist())

if __name__ == "__main__":
    test_engines()
