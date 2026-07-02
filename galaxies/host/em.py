"""Emission-measure helpers following ``FRBs/FRB`` ``frb/em.py`` (Reynolds 1977)."""

from __future__ import annotations

import astropy.constants as const
import astropy.units as u
import numpy as np

_SR_PER_ARCSEC2 = (np.pi / (180.0 * 3600.0)) ** 2
_HA_E_HA_ERG = (const.c * const.h / (6564.613 * u.Angstrom)).to(u.erg).value


def _sb_value_erg_cm2_s_arcsec2(sb_obs: u.Quantity) -> float:
    """Return surface brightness in erg s^-1 cm^-2 arcsec^-2."""
    try:
        return float(sb_obs.to(u.erg / u.s / u.cm**2 / u.arcsec**2).value)
    except u.UnitConversionError:
        # FRB passes erg/s/arcsec^2 (angular surface brightness label).
        return float(sb_obs.to(u.erg / u.s / u.arcsec**2).value)


def _halpha_sb_to_rayleigh(sb_erg_cm2_s_arcsec2: float) -> float:
    photon_flux_sr = (sb_erg_cm2_s_arcsec2 / _HA_E_HA_ERG) / _SR_PER_ARCSEC2
    return float(photon_flux_sr * 4.0 * np.pi / 1e6)


def em_from_halpha(sb_obs: u.Quantity, z: float, T: u.Quantity = 1e4 * u.K) -> u.Quantity:
    """Estimate EM from observed H-alpha surface brightness."""
    sb_corr = _sb_value_erg_cm2_s_arcsec2(sb_obs) * (1.0 + float(z)) ** 4
    i_rayleigh = _halpha_sb_to_rayleigh(sb_corr)
    return 2.75 * u.pc / u.cm**6 * (T.to(u.K).value / 1e4) ** 0.9 * i_rayleigh


def dm_from_em(
    em: u.Quantity,
    path_length: u.Quantity,
    *,
    filling_factor: float = 1.0,
    eps: float = 1.0,
    cloudcloud: float = 2.0,
) -> u.Quantity:
    """DM at the source from EM (Tendulkar+2017 / Reynolds 1977 / Cordes+2016)."""
    l_kpc = path_length.to(u.kpc).value
    em_pc = em.to(u.pc / u.cm**6).value
    scale = filling_factor / (cloudcloud * (1.0 + eps**2) / 4.0)
    return 387 * u.pc / u.cm**3 * (l_kpc**0.5) * (scale**0.5) * ((em_pc / 600.0) ** 0.5)
