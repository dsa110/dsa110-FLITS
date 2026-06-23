import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.constants as const

# Assume these are defined elsewhere in your script
# from baseband_analysis.core.bbdata import BBData
# from baseband_analysis.core.dedispersion import delay_across_the_band

# Dispersion constant in MHz^2 pc^-1 cm^3 s
K_DM = 4.148808e3

def calculate_dm_timing_error(dm_uncertainty, f_obs, f_ref):
    """
    Calculates the timing error due to DM uncertainty.

    Parameters
    ----------
    dm_uncertainty : float
        The uncertainty in the Dispersion Measure (pc/cm^3).
    f_obs : astropy.units.Quantity
        The central observing frequency in MHz.
    f_ref : astropy.units.Quantity
        The reference frequency in MHz.

    Returns
    -------
    astropy.units.Quantity
        The timing error in milliseconds.
    """
    # Calculate the time shift in seconds
    time_shift = K_DM * dm_uncertainty * (1 / f_obs.value**2 - 1 / f_ref.value**2) * u.s
    
    # Return the absolute value in milliseconds
    return np.abs(time_shift.to(u.ms))


# --- Input Parameters for the Single Burst ---
# In your real code, you would load or define these values.
dm_opt = 550.0  # pc/cm^3
dm_uncertainty = 0.2  # pc/cm^3
dsa_mjd = 59000.1
chime_unix_timestamp = 1598882400.0 # Example Unix time
source_coord = "12:00:00 +20:00:00"

print("--- Analyzing Single Burst ---")

# ==================================================================
# This section would contain your CHIME data processing code
# to derive peak_idx_chime, etc.
# For this example, we'll use placeholder values.
# ==================================================================
DM = dm_opt * (u.pc) / (u.cm**3)
# Mocking CHIME results
t0_unix_chime = chime_unix_timestamp * u.s
offset_chime = 0.01 * u.s

# CHIME frequency setup
# Common reference frequency for all TOAs
F_REF = 400.0 * u.MHz
# Representative central frequency for CHIME's band (400.39 - 800.39 MHz)
f_center_chime = 600.39 * u.MHz 

# Your TOA calculation for CHIME
# Note: Your original code used f_dump_chime. Using the band center is also valid,
# as long as the method is consistent.
shift_400_chime = K_DM * DM.value * (1/F_REF.value**2 - 1/f_center_chime.value**2) * u.s
toa_400_unix_chime = t0_unix_chime + offset_chime + shift_400_chime
toa_400_utc_chime = Time(toa_400_unix_chime.value, format='unix', scale='utc')

# ==================================================================
# This section would contain your DSA-110 data processing code
# ==================================================================
# Mocking DSA-110 results
t0_utc_dsa = Time(dsa_mjd, format='mjd', scale='utc')
offset_dsa = 0.005 * u.s

# DSA-110 frequency setup
# Representative central frequency for DSA-110's band (1311.25 - 1498.75 MHz)
f_center_dsa = 1405.0 * u.MHz 

# Your TOA calculation for DSA-110
t_peak_utc_dsa = t0_utc_dsa + offset_dsa
shift_400_dsa = K_DM * DM.value * (1/F_REF.value**2 - 1/f_center_dsa.value**2) * u.s
toa_400_utc_dsa = t_peak_utc_dsa + shift_400_dsa

# --- UNCERTAINTY CALCULATION ---
print(f"Assumed DM Uncertainty: {dm_uncertainty:.2f} pc/cm^3")

# Calculate timing error for each observatory relative to the 400 MHz reference
error_chime = calculate_dm_timing_error(dm_uncertainty, f_center_chime, F_REF)
error_dsa = calculate_dm_timing_error(dm_uncertainty, f_center_dsa, F_REF)

# The total uncertainty on the offset is the sum in quadrature
delta_t_uncertainty = np.sqrt(error_chime**2 + error_dsa**2)

print(f"CHIME TOA Error due to DM uncertainty: {error_chime:.3f}")
print(f"DSA-110 TOA Error due to DM uncertainty: {error_dsa:.3f}")

# --- Final Results ---
dt = toa_400_utc_chime - toa_400_utc_dsa
print(f"\nMeasured TOA Offset (Δt): {dt.to(u.ms):.3f}")
print(f"Combined Uncertainty on Δt from DM: ±{delta_t_uncertainty:.3f}")

# Geometric delay calculation
src = SkyCoord(source_coord, unit=(u.hourangle, u.deg), frame='icrs')
chime_loc = EarthLocation.of_site('DRAO')
dsa_loc = EarthLocation.of_site('OVRO')

def geometric_delay(t):
    p1 = chime_loc.get_gcrs(t).cartesian.xyz
    p2 = dsa_loc.get_gcrs(t).cartesian.xyz
    proj = (p2 - p1).dot(src.cartesian.xyz)
    return (proj / const.c).to(u.ms)

print(f"Geometric Delay: {geometric_delay(toa_400_utc_chime):.3f}")
