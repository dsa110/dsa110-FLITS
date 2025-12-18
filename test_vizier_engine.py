import astropy.units as u
from astropy.coordinates import SkyCoord
from galaxies.v2_0.engines import VizierEngine
from galaxies.v2_0.config import VIZIER_CATALOGS
import pandas as pd

def test_vizier():
    # M31
    coord = SkyCoord.from_name("M31")
    radius = 10 * u.arcmin
    
    from astroquery.vizier import Vizier
    v = Vizier(columns=['*'])
    print("Querying Vizier without catalog ID...")
    result = v.query_region(coord, radius=radius)
    print(f"Found {len(result)} tables.")
    if result:
        for table in result:
            print(f"Table: {table.meta.get('ID')}, Rows: {len(table)}")

if __name__ == "__main__":
    test_vizier()
