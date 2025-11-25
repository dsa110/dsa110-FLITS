import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
import numpy as np

# List of target coordinates (RA, DEC in sexagesimal strings)
targets = [
    ("20h40m47.886s", "+72d52m56.378s"),
    ("08h58m52.92s",  "+73d29m27.0s"),
    ("21h12m10.760s", "+72d49m38.20s"),
    ("04h45m38.64s",  "+70d18m26.6s"),
    ("21h00m31.09s",  "+72d02m15.22s"),
    ("11h51m07.52s",  "+71d41m44.3s"),
    ("05h52m45.12s",  "+74d12m01.7s"),
    ("20h20m08.92s",  "+70d47m33.96s"),
    ("02h39m03.96s",  "+71d01m04.3s"),
    ("20h50m28.59s",  "+73d54m00.0s"),
    ("11h19m56.05s",  "+70d40m34.4s"),
    ("22h23m53.94s",  "+73d01m33.26s"),
]

# Convert targets to SkyCoord objects
sky_targets = [SkyCoord(ra, dec, unit=("hourangle", "deg")) for ra, dec in targets]

# Find all FITS files in the directory
fits_files = glob.glob("*.fits")

results = {i: [] for i in range(len(targets))}

for fname in fits_files:
    with fits.open(fname) as hdul:
        ra = hdul[1].data['RA']
        dec = hdul[1].data['DEC']
        ra_min, ra_max = np.min(ra), np.max(ra)
        dec_min, dec_max = np.min(dec), np.max(dec)
        # Check each target
        for i, t in enumerate(sky_targets):
            if (ra_min <= t.ra.deg <= ra_max) and (dec_min <= t.dec.deg <= dec_max):
                results[i].append(fname)

# Print results
for i, files in results.items():
    print(f"Target {i+1} ({targets[i][0]}, {targets[i][1]}):")
    if files:
        for f in files:
            print(f"  - {f}")
    else:
        print("  No file covers this target.")