import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.constants as const


def main():
    # Inputs for the new co-detected FRB
    # Source position (ICRS)
    ra_dec = "20h31m06.38s +53d50m56.4s"

    # Dispersion Measure (pc cm^-3)
    DM = 405.31751275368407

    # Frequencies
    f_ref = 400.0 * u.MHz   # common reference frequency
    f_center_dsa = 1405.0 * u.MHz  # effective DSA band center

    # Dispersion constant in s MHz^2 pc^-1 cm^3
    K_DM = 4.148808e3

    # DSA peak epoch (UTC MJD)
    dsa_mjd = 60942.172498155225
    t0_dsa = Time(dsa_mjd, format="mjd", scale="utc")

    # CHIME/FRB baseband TOA already referenced to 400 MHz (UTC time-of-day)
    # Use the same calendar date as the DSA MJD
    t_chime_400 = Time(t0_dsa.datetime.strftime("%Y-%m-%d") + " 04:08:33.399", format="iso", scale="utc")

    # Shift the DSA time to 400 MHz using the cold-plasma DM delay
    shift_400_dsa = (K_DM * DM * (1.0 / f_ref.value**2 - 1.0 / f_center_dsa.value**2)) * u.s
    t_dsa_400 = (t0_dsa + shift_400_dsa).utc

    # Measured offset (CHIME − DSA) at 400 MHz in ms
    dt_meas_ms = (t_chime_400 - t_dsa_400).to_value(u.ms)

    # Geometric delay (OVRO − DRAO) at the CHIME time
    src = SkyCoord(ra_dec, unit=(u.hourangle, u.deg), frame="icrs")
    drao = EarthLocation.of_site("DRAO")
    try:
        ovro = EarthLocation.of_site("OVRO")
    except Exception:
        # Fallback OVRO/DSA-110 site coordinates if not present in site registry
        ovro = EarthLocation.from_geodetic(lon=-118.2869 * u.deg, lat=37.2314 * u.deg, height=1200 * u.m)

    def geometric_delay_ms(t):
        p1 = drao.get_gcrs(t).cartesian.xyz
        p2 = ovro.get_gcrs(t).cartesian.xyz
        proj = (p2 - p1).dot(src.cartesian.xyz)
        return (proj / const.c).to_value(u.ms)

    geo_ms = geometric_delay_ms(t_chime_400)
    residual_ms = dt_meas_ms - geo_ms

    # Output
    print("Event inputs:")
    print(f"  RA/Dec (ICRS):         {ra_dec}")
    print(f"  DM (pc cm^-3):         {DM}")
    print(f"  DSA MJD (UTC):         {t0_dsa.mjd:.12f}")
    print()
    print("TOAs at 400 MHz:")
    print(f"  t_CHIME@400 (UTC):     {t_chime_400.iso}")
    print(f"  t_DSA@400   (UTC):     {t_dsa_400.iso}")
    print()
    print("Offsets:")
    print(f"  Δt_meas (CHIME−DSA):   {dt_meas_ms:.3f} ms")
    print(f"  Geometric delay:       {geo_ms:.3f} ms (OVRO−DRAO)")
    print(f"  Residual (Δt−geom):    {residual_ms:.3f} ms")


if __name__ == "__main__":
    main()

