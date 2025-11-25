from astropy.table import Table, hstack, vstack
import glob, os

# -----------------------------
# helper to pair pz & sweep     
# -----------------------------
def twin(fname):
    return fname.replace('/8.0-photo-z/', '/8.0/').replace('-pz.fits', '.fits')

pz_files = sorted(glob.glob('sweep-*p070-*p075-pz.fits'))
sweep_files = [twin(f) for f in pz_files]

# quick integrity check
for pz, sw in zip(pz_files, sweep_files):
    assert os.path.exists(sw), f"Missing sweep twin for {pz}"

# -----------------------------
# read & stack                  -----------------------------
# -----------------------------
all_tables = []
keep_cols  = ['RA', 'DEC', 'BRICKID', 'OBJID', 'TYPE',  # from sweep
              'z_phot_mean', 'z_phot_std', 'z_phot_l68', 'z_phot_u68']  # from pz

for pz_path, sw_path in zip(pz_files, sweep_files):

    # Read *only* the columns you need from each file
    sweep = Table.read(sw_path, memmap=True)            # RA, DEC, etc.
    pz    = Table.read(pz_path, memmap=True)            # the 10 photo-z cols

    # Defensive sanity check: lengths must match
    if len(sweep) != len(pz):
        raise ValueError(f"Row mismatch: {sw_path}")

    merged = hstack([sweep, pz], join_type='exact')     # constant-time hstack
    all_tables.append( merged[keep_cols] )              # retain chosen cols

# one giant Dec-stripe table
photoz_70_75 = vstack(all_tables, metadata_conflicts='silent')

# final quality cuts
mask = (photoz_70_75['z_phot_mean'] > -90)              # NOBS cut (-99 flag)
photoz_70_75 = photoz_70_75[mask]

photoz_70_75.write('dr8_photoz_dec70_75_full.fits', overwrite=True)
print(len(photoz_70_75), "rows written.")
