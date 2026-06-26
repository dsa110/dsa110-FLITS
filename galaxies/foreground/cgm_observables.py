"""Pure CGM observable estimates from catalog-derivable galaxy scalars."""

import math

import numpy as np
import astropy.units as u

from .config import COSMO


def _is_bad(x: float | None) -> bool:
    """Return True for missing, NaN, or non-finite scalar inputs."""
    if x is None:
        return True
    try:
        value = float(x)
    except (TypeError, ValueError):
        return True
    return math.isnan(value) or bool(np.isnan(value)) or not math.isfinite(value)


def axis_ratio_from_ellipticity(e1: float, e2: float) -> float | None:
    """Convert LegacySurvey/Tractor ellipticity components to axis ratio."""
    if _is_bad(e1) or _is_bad(e2):
        return None

    # Tractor/Legacy Survey shape docs define reduced ellipticity magnitude
    # |e|=sqrt(e1^2+e2^2) and b/a=(1-|e|)/(1+|e|) for catalog shapes.
    ellipticity = math.hypot(float(e1), float(e2))
    if ellipticity >= 1.0:
        return None
    q = (1.0 - ellipticity) / (1.0 + ellipticity)
    if q <= 0.0 or q > 1.0 or not math.isfinite(q):
        return None
    return float(q)


def position_angle_deg(e1: float, e2: float) -> float | None:
    """Return galaxy position angle in degrees, folded to [0, 180)."""
    if _is_bad(e1) or _is_bad(e2):
        return None

    e1_value = float(e1)
    e2_value = float(e2)
    if math.isclose(e1_value, 0.0, abs_tol=1e-12) and math.isclose(e2_value, 0.0, abs_tol=1e-12):
        return None

    # Tractor/Legacy Survey ellipticity components encode the orientation as
    # (e1,e2)=|e|(cos 2PA, sin 2PA), hence PA=0.5*atan2(e2,e1).
    pa = 0.5 * math.degrees(math.atan2(e2_value, e1_value))
    return float(pa % 180.0)


def inclination_deg(q: float, q0: float = 0.15) -> float | None:
    """Estimate disk inclination from observed and intrinsic axis ratios."""
    if _is_bad(q) or _is_bad(q0):
        return None

    q_value = float(q)
    q0_value = float(q0)
    if q0_value < 0.0 or q0_value >= 1.0:
        return None
    if q_value <= q0_value:
        return None
    if q_value >= 1.0:
        return 0.0

    # Hubble 1926 / Holmberg 1958 oblate disk correction uses intrinsic
    # thickness q0: cos^2 i=(q^2-q0^2)/(1-q0^2).
    cos2_i = (q_value**2 - q0_value**2) / (1.0 - q0_value**2)
    cos2_i = min(1.0, max(0.0, cos2_i))
    return float(math.degrees(math.acos(math.sqrt(cos2_i))))


def azimuthal_angle_phi_deg(
    gal_ra: float,
    gal_dec: float,
    pa_deg: float,
    sight_ra: float,
    sight_dec: float,
) -> float | None:
    """Return sightline azimuthal angle relative to the galaxy major axis."""
    if any(_is_bad(x) for x in (gal_ra, gal_dec, pa_deg, sight_ra, sight_dec)):
        return None

    ra1 = math.radians(float(gal_ra))
    dec1 = math.radians(float(gal_dec))
    ra2 = math.radians(float(sight_ra))
    dec2 = math.radians(float(sight_dec))
    delta_ra = ra2 - ra1

    # Bordoloi+2011, Bouché+2012, and Kacprzak+2012 use CGM azimuthal angle
    # as the projected separation direction relative to the galaxy major axis.
    bearing = math.degrees(
        math.atan2(
            math.sin(delta_ra) * math.cos(dec2),
            math.cos(dec1) * math.sin(dec2)
            - math.sin(dec1) * math.cos(dec2) * math.cos(delta_ra),
        )
    )
    diff = abs((bearing - float(pa_deg)) % 180.0)
    phi = min(diff, 180.0 - diff)
    return float(min(90.0, max(0.0, phi)))


def stellar_mass_desi_gz(g_ab: float, z_ab: float, z_gal: float) -> float | None:
    """Estimate log10 stellar mass from DESI/optical g-z and z-band light."""
    if _is_bad(g_ab) or _is_bad(z_ab) or _is_bad(z_gal) or float(z_gal) <= 0.0:
        return None

    color_gz = float(g_ab) - float(z_ab)
    if not math.isfinite(color_gz):
        return None

    # Zibetti+2009 MNRAS 400,1181 Table B1 gives z-band log10(M/L_z) as a
    # linear function of rest-frame g-z color; k-correction/internal extinction
    # are neglected because this pure helper lacks SED and dust information.
    log_ml_z = -0.171 + 0.322 * color_gz
    abs_mag_z = float(z_ab) - COSMO.distmod(float(z_gal)).value

    # Willmer 2018 ApJS 236,47 Table 3 gives M_sun,z=4.50 AB for SDSS z.
    log_l_z = -0.4 * (abs_mag_z - 4.50)
    log_mstar = log_ml_z + log_l_z
    if not math.isfinite(log_mstar):
        return None
    return float(log_mstar)


def stellar_mass_wise_w1(w1_vega: float, z_gal: float, w1_w2: float | None = None) -> float | None:
    """Estimate log10 stellar mass from WISE W1 luminosity."""
    if _is_bad(w1_vega) or _is_bad(z_gal) or float(z_gal) <= 0.0:
        return None

    if _is_bad(w1_w2):
        color = 0.0
    else:
        color = float(w1_w2)

    # Cluver+2014 ApJ 782,90 eq.2 calibrates W1 log10(M/L) from W1-W2 color;
    # if W1-W2 is missing, using color=0 keeps the published zero-color
    # intercept instead of imposing an external catalog-dependent prior.
    log_ml_w1 = -2.54 * color - 0.17

    # Distance modulus is computed from luminosity distance; W1 k-correction is
    # neglected for the same low-z/no-SED reason as the optical estimator.
    d_l_pc = COSMO.luminosity_distance(float(z_gal)).to(u.pc).value
    if d_l_pc <= 0.0 or not math.isfinite(d_l_pc):
        return None
    abs_mag_w1 = float(w1_vega) - 5.0 * math.log10(d_l_pc / 10.0)

    # Jarrett/WISE convention uses M_sun,W1=3.24 in the Vega system.
    log_l_w1 = -0.4 * (abs_mag_w1 - 3.24)
    log_mstar = log_ml_w1 + log_l_w1
    if not math.isfinite(log_mstar):
        return None
    return float(log_mstar)


def is_star_forming(gr_color: float, logmstar: float | None = None) -> bool:
    """Classify a galaxy as star-forming from rest-frame g-r color."""
    if _is_bad(gr_color):
        # Conservative Lan & Mo 2018 / Schawinski+2014 green-valley usage:
        # missing color cannot confirm blue-cloud membership.
        return False

    if _is_bad(logmstar):
        # Lan & Mo 2018 and Schawinski+2014 motivate a simple green-valley
        # split near g-r=0.65 when mass-dependent color information is absent.
        threshold = 0.65
    else:
        # Lan & Mo 2018 show an approximate weak mass tilt to the color split;
        # this bounded linear form keeps the pure helper deterministic.
        threshold = 0.6 + 0.02 * (float(logmstar) - 10.0)
        threshold = min(0.75, max(0.55, threshold))
    return bool(float(gr_color) < threshold)


def sfr_wise_w3(w3_vega: float, z_gal: float) -> float | None:
    """Estimate star-formation rate from WISE W3 luminosity."""
    if _is_bad(w3_vega) or _is_bad(z_gal) or float(z_gal) <= 0.0:
        return None

    # W3 luminosity uses luminosity-distance distance modulus; k-correction is
    # neglected at low z because no mid-IR SED shape is available here.
    d_l_pc = COSMO.luminosity_distance(float(z_gal)).to(u.pc).value
    if d_l_pc <= 0.0 or not math.isfinite(d_l_pc):
        return None
    abs_mag_w3 = float(w3_vega) - 5.0 * math.log10(d_l_pc / 10.0)

    # Jarrett+2013 gives M_sun,W3=3.27 Vega for converting W3 magnitude to Lsun.
    log_l_12um = -0.4 * (abs_mag_w3 - 3.27)

    # Cluver+2017 ApJ 850,68 calibrates log10(SFR)=0.889 log10(L_12um)-7.76.
    log_sfr = 0.889 * log_l_12um - 7.76
    if not math.isfinite(log_sfr):
        return None
    return float(10.0**log_sfr)


def sfr_uv_nuv(nuv_ab: float, z_gal: float, ebv: float = 0.0) -> float | None:
    """Estimate NUV star-formation rate from an AB magnitude."""
    if _is_bad(nuv_ab) or _is_bad(z_gal) or _is_bad(ebv) or float(z_gal) <= 0.0:
        return None

    # Wyder+2007 supplies the GALEX NUV extinction coefficient A_NUV=8.2 E(B-V).
    a_nuv = 8.2 * float(ebv)
    nuv_corr = float(nuv_ab) - a_nuv

    # AB zeropoint converts corrected NUV magnitude to f_nu in erg/s/cm^2/Hz.
    f_nu = 10.0 ** (-0.4 * (nuv_corr + 48.60))
    d_l_cm = COSMO.luminosity_distance(float(z_gal)).to(u.cm).value
    if f_nu <= 0.0 or d_l_cm <= 0.0 or not math.isfinite(f_nu) or not math.isfinite(d_l_cm):
        return None

    # Kennicutt & Evans 2012 ARA&A 50,531 Table 1 uses L_NUV=nu L_nu at
    # 0.23 micron with log C_NUV=43.17; rest-frame k-correction is neglected.
    l_nu = f_nu * 4.0 * math.pi * d_l_cm**2
    nu_nuv = 2.99792458e18 / 2316.0
    l_nuv = nu_nuv * l_nu
    if l_nuv <= 0.0 or not math.isfinite(l_nuv):
        return None
    log_sfr = math.log10(l_nuv) - 43.17
    return float(10.0**log_sfr)


def metallicity_mzr(logmstar: float, z_gal: float = 0.0) -> float | None:
    """Estimate gas-phase oxygen abundance from the local mass-metallicity relation."""
    if _is_bad(logmstar):
        return None

    # Curti+2020 MNRAS 491,944 eq.5/Table 2 local MZR; z_gal is retained for
    # API compatibility, but redshift evolution is intentionally ignored here.
    z0 = 8.793
    log_m0 = 10.02
    gamma = 0.28
    beta = 1.20
    mass_ratio = 10.0 ** (float(logmstar) - log_m0)
    metallicity = z0 - (gamma / beta) * math.log10(1.0 + mass_ratio ** (-beta))
    if not math.isfinite(metallicity):
        return None
    return float(metallicity)


def wise_agn_stern2012(w1_w2: float) -> bool:
    """Return True when WISE W1-W2 color satisfies the Stern+2012 AGN cut."""
    if _is_bad(w1_w2):
        return False

    # Stern+2012 ApJ 753,30 selects WISE AGN candidates with W1-W2 >= 0.8 Vega.
    return bool(float(w1_w2) >= 0.8)
