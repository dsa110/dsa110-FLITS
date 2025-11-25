from astroquery.utils.tap.core import TapPlus
from astropy.table import Table
import pandas as pd, astropy.units as u

# Tap endpoint
tap = TapPlus(url="https://datalab.noirlab.edu/tap")

# ------------------------ parameters you supply -------------------------
ra0, dec0 = 310.199524, 72.882327   # deg  (20h40m47.886s +72d52m56.38s)
radius   = 0.033333                 # deg  (≈ 2 arcmin)
z0, dz   = 0.025, 0.0007            # example cylinder in redshift space
# -----------------------------------------------------------------------

adql = f"""
SELECT z.targetid, z.ra, z.dec, z.z, s.spec
FROM   desi_dr1.zpix       AS z
JOIN   desi_dr1.spectra_hp AS s ON z.targetid = s.targetid
WHERE  q3c_radial_query(z.ra, z.dec,
                        {ra0}, {dec0}, {radius})
  AND  ABS(z.z - {z0}) < {dz}
"""

# synchronous is fine – result set will be tiny
job = tap.launch_job(adql, dump_to_file=False)
tbl = job.get_results()            # astropy.table.Table
df  = tbl.to_pandas()              # if you prefer pandas
print(df.head())

