"""Host-galaxy DM models ported from ``FRBs/FRB`` ``frb/dm/host.py``."""

from __future__ import annotations

import math
from typing import Any

import astropy.units as u
import numpy as np
from scipy.optimize import brentq

from galaxies.foreground import scattering_predict as scat
from galaxies.host import em
from galaxies.host.catalog import HostRecord

# Kennicutt 1998 + Chabrier IMF (``frb.galaxies.nebular.Ha_conversion``).
HA_CONVERSION = 0.63 * 7.9e-42 * u.Msun / u.yr

# Host halos use a lower hot-gas fraction than intervening CGM screens (FRB default).
HOST_MNFW_F_HOT = 0.55


def _stellarmass_from_halomass(log_mhalo: float, z: float) -> float:
    """Moster+2013 Eq. 2 with redshift evolution (``frb.halos.models``)."""
    z_factor = float(z) / (1.0 + float(z))
    n = 0.0351 + (-0.0247) * z_factor
    beta = 1.376 + (-0.826) * z_factor
    gamma = 0.608 + 0.329 * z_factor
    log_m1 = 11.59 + 1.195 * z_factor
    m1 = 10.0**log_m1
    m_halo = 10.0**float(log_mhalo)
    return float(log_mhalo) + np.log10(2.0 * n) - np.log10((m_halo / m1) ** (-beta) + (m_halo / m1) ** gamma)


def halomass_from_stellarmass(log_mstar: float, z: float) -> float:
    """Invert Moster+2013 at redshift ``z`` (``frb.halos.models.halomass_from_stellarmass``)."""
    log_mstar = float(log_mstar)
    z_val = float(z)
    lo, hi = 9.5, 15.5
    lo_mstar, hi_mstar = _stellarmass_from_halomass(lo, z_val), _stellarmass_from_halomass(hi, z_val)
    log_mstar = min(max(log_mstar, min(lo_mstar, hi_mstar) + 1e-6), max(lo_mstar, hi_mstar) - 1e-6)
    log_mh = brentq(lambda lmh: _stellarmass_from_halomass(lmh, z_val) - log_mstar, lo, hi)
    return float(10.0**log_mh)


def _finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def dm_host_halo(
    impact_kpc: float,
    log10_mstar: float,
    z: float,
    *,
    f_hot: float = HOST_MNFW_F_HOT,
) -> float | None:
    """Hot-halo DM at projected radius ``impact_kpc`` (observer frame, pc cm^-3).

    Uses the FLITS mNFW kernel with ``f_hot=0.55`` and halves the column, matching
    ``frb.dm.host.dm_host_halo`` (half-path through the host).
    """
    if not all(_finite(x) for x in (impact_kpc, log10_mstar, z)):
        return None
    if float(z) <= 0.0 or float(impact_kpc) < 0.0:
        return None
    try:
        m_halo = halomass_from_stellarmass(float(log10_mstar), float(z))
    except (ArithmeticError, ValueError, RuntimeError, OverflowError):
        return None
    if m_halo <= 0.0:
        return None
    column = scat.dm_mnfw_projected(
        m_halo,
        float(z),
        float(impact_kpc),
        f_hot=float(f_hot),
    )
    if not math.isfinite(column):
        return None
    return float(max(column / 2.0, 0.0))


def dm_host_from_halpha(
    z: float,
    halpha_flux: u.Quantity,
    reff_ang: u.Quantity,
    *,
    av: float | None = None,
    path_length_kpc: float = 1.0,
) -> float | None:
    """ISM DM from total H-alpha flux and effective radius (observer frame)."""
    if not _finite(z) or float(z) <= 0.0:
        return None
    if reff_ang <= 0 * u.arcsec:
        return None

    al = 0.0
    if av is not None and _finite(av):
        try:
            from dust_extinction.parameter_averages import G23

            extmod = G23(Rv=3.1)
            al_av = float(extmod(np.atleast_1d(6564.0))[0])
            al = al_av * float(av)
        except ImportError:
            al = 0.0

    halpha_sb = halpha_flux * 10 ** (al / 2.5) / (np.pi * reff_ang**2)
    em_r1 = em.em_from_halpha(halpha_sb, float(z))
    dm_obs = em.dm_from_em(em_r1, path_length_kpc * u.kpc) / (1.0 + float(z))
    return float(max(dm_obs.to(u.pc / u.cm**3).value, 0.0))


def dm_host_from_ssfr(
    z: float,
    ssfr: u.Quantity,
    *,
    path_length_pc: float = 100.0,
) -> float | None:
    """ISM DM from specific SFR surface density (observer frame)."""
    if not _finite(z) or float(z) <= 0.0:
        return None
    from galaxies.foreground.config import COSMO

    halpha_kpc2 = ssfr / HA_CONVERSION * u.erg / u.s
    kpc_arcmin = COSMO.kpc_proper_per_arcmin(float(z))
    halpha_sqarcsec = halpha_kpc2 * kpc_arcmin**2
    d_l = COSMO.luminosity_distance(float(z))
    halpha_sb = halpha_sqarcsec / (4.0 * np.pi * d_l**2)
    em_burst = em.em_from_halpha(halpha_sb, float(z))
    dm_obs = em.dm_from_em(em_burst, path_length_pc * u.pc) / (1.0 + float(z))
    return float(max(dm_obs.to(u.pc / u.cm**3).value, 0.0))


def predict_host_dm(record: HostRecord) -> dict[str, float | None | str]:
    """Return predicted host halo + ISM DM components for one burst."""
    out: dict[str, float | None | str] = {
        "dm_host_halo_pred": None,
        "dm_host_ism_pred": None,
        "dm_host_pred": None,
        "host_pred_method": "none",
    }
    if record.z_is_placeholder or not _finite(record.z):
        out["host_pred_method"] = "z_placeholder"
        return out

    z = float(record.z)
    if _finite(record.log10_mstar):
        out["dm_host_halo_pred"] = dm_host_halo(
            float(record.offset_kpc) if _finite(record.offset_kpc) else 0.0,
            float(record.log10_mstar),
            z,
        )
        out["host_pred_method"] = "halo_mnfw"

    ism = None
    ism_method = None
    if _finite(record.halpha_flux_erg_s) and _finite(record.reff_arcsec):
        ism = dm_host_from_halpha(
            z,
            float(record.halpha_flux_erg_s) * u.erg / u.s,
            float(record.reff_arcsec) * u.arcsec,
            av=record.av if _finite(record.av) else None,
            path_length_kpc=float(record.ism_path_kpc)
            if _finite(record.ism_path_kpc)
            else 1.0,
        )
        ism_method = "halpha"
    elif _finite(record.ssfr_msun_yr_kpc2):
        ism = dm_host_from_ssfr(
            z,
            float(record.ssfr_msun_yr_kpc2) * u.Msun / u.yr / u.kpc**2,
            path_length_pc=float(record.ism_path_pc) if _finite(record.ism_path_pc) else 100.0,
        )
        ism_method = "ssfr"

    if ism is not None:
        out["dm_host_ism_pred"] = ism
        if out["host_pred_method"] == "halo_mnfw":
            out["host_pred_method"] = "halo_mnfw+halpha" if ism_method == "halpha" else "halo_mnfw+ssfr"
        else:
            out["host_pred_method"] = ism_method or "none"

    parts = [out["dm_host_halo_pred"], out["dm_host_ism_pred"]]
    finite_parts = [p for p in parts if p is not None and _finite(p)]
    if finite_parts:
        out["dm_host_pred"] = float(sum(finite_parts))
    return out
