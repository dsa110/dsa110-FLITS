import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# List of targets and their relevant FITS files (from your mapping)
targets = [
    ("20h40m47.886s", "+72d52m56.378s", 0.0430, "SOV_31.03_zphot_final.fits"),
    ("08h58m52.92s",  "+73d29m27.0s",   0.4790, "SOV_31.04_zphot_final.fits"),
    ("21h12m10.760s", "+72d49m38.20s",  0.3005, "SOV_31.03_zphot_final.fits"),
    ("04h45m38.64s",  "+70d18m26.6s",   0.2505, "SOV_31.01_zphot_final.fits"),
    ("21h00m31.09s",  "+72d02m15.22s",  0.5100, "SOV_31.03_zphot_final.fits"),
    ("11h51m07.52s",  "+71d41m44.3s",   0.2710, "SOV_31.02_zphot_final.fits"),
    ("05h52m45.12s",  "+74d12m01.7s",   1.0000, "SOV_31.04_zphot_final.fits"),
    ("20h20m08.92s",  "+70d47m33.96s",  0.3024, "SOV_31.01_zphot_final.fits"),
    ("02h39m03.96s",  "+71d01m04.3s",   1.0000, "SOV_31.01_zphot_final.fits"),
    ("20h50m28.59s",  "+73d54m00.0s",   0.0740, "SOV_31.04_zphot_final.fits"),
    ("11h19m56.05s",  "+70d40m34.4s",   0.2870, "SOV_31.01_zphot_final.fits"),
    ("22h23m53.94s",  "+73d01m33.26s",  1.0000, "SOV_31.03_zphot_final.fits"),
]

for i, (ra_str, dec_str, z_limit, fits_file) in enumerate(targets, 1):
    z_limit_eps = z_limit * 1.5
    print(f"Processing Target {i}: {ra_str}, {dec_str} in {fits_file} (z_limit={z_limit}, z_limit_eps={z_limit_eps})")
    sightline = SkyCoord(ra_str, dec_str, unit=("hourangle", "deg"), frame="icrs")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        # Only consider galaxies with positive photometric redshift and z < z_limit_eps
        mask = (data['Z_PHOT'] >= 0.01) & (data['Z_PHOT'] < z_limit_eps)
        ra = data['RA'][mask]
        dec = data['DEC'][mask]
        z_phot = data['Z_PHOT'][mask]
        # Compute angular separation
        galaxies = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
        sep = sightline.separation(galaxies)
        # Compute angular diameter distance for each galaxy
        D_A = cosmo.angular_diameter_distance(z_phot)
        # Impact parameter in kpc
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            impact_kpc = (sep * D_A).to(u.kpc)
        # Select galaxies with impact parameter < 200 kpc
        close_mask = impact_kpc.value < 200
        selected = data[mask][close_mask]
        if len(selected) > 0:
            arr = np.array(selected)
            if arr.dtype.byteorder not in ('=', '|'):
                arr = arr.view(arr.dtype.newbyteorder('='))
            df = pd.DataFrame(arr)
            df['impact_kpc'] = impact_kpc.value[close_mask]
            outname = f"target_{i}_matches.csv"
            df.to_csv(outname, index=False)
            print(f"  Saved {len(df)} matches to {outname}")
        else:
            print("  No galaxies found within 200 kpc and z < z_limit_eps.")

print("Done.")