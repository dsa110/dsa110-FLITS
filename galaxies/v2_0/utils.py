"""Utility functions for galaxy search v2.0."""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from .config import COSMO

def get_angular_radius(z: float, impact_kpc: float) -> u.Quantity:
    """Calculate angular radius for a given physical impact parameter at redshift z."""
    if z <= 0:
        return 0.0 * u.arcmin
    
    d_a = COSMO.angular_diameter_distance(z)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        theta_rad = (impact_kpc * u.kpc / d_a).to(u.rad)
    return theta_rad.to(u.arcmin)

def calculate_impact_parameter(
    ra_gal: float, 
    dec_gal: float, 
    z_gal: float, 
    ra_sight: float, 
    dec_sight: float
) -> float:
    """Calculate physical impact parameter in kpc."""
    if z_gal <= 0:
        return np.nan
        
    c_sight = SkyCoord(ra_sight, dec_sight, unit='deg', frame='icrs')
    c_gal = SkyCoord(ra_gal, dec_gal, unit='deg', frame='icrs')
    sep = c_sight.separation(c_gal)
    
    d_a = COSMO.angular_diameter_distance(z_gal)
    impact_kpc = (sep.radian * d_a).to(u.kpc).value
    return impact_kpc

def parse_coord(ra_str: str, dec_str: str) -> SkyCoord:
    """Parse RA/Dec strings into a SkyCoord object."""
    return SkyCoord(ra_str, dec_str, frame='icrs')
