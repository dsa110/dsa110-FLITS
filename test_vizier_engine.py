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
    catalog_id = "VII/281/glade2"
    print(f"Querying Vizier {catalog_id} for M31...")
    result = v.query_region(coord, radius=radius, catalog=catalog_id)
    print(f"Found {len(result)} tables.")
    if result:
        for table in result:
            print(f"Table: {table.meta.get('ID')}, Rows: {len(table)}")
            print("Columns:", table.colnames)

if __name__ == "__main__":
    test_vizier()
