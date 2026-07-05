"""Rank normalized candidates by impact parameter and halo scale."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Cosmology


def angular_separation_arcmin(ra1_deg, dec1_deg, ra2_deg: float, dec2_deg: float):
    """Angular separation in arcmin; accepts scalars or arrays for the first position."""
    a = SkyCoord(ra1_deg * u.deg, dec1_deg * u.deg, frame="icrs")
    b = SkyCoord(ra2_deg * u.deg, dec2_deg * u.deg, frame="icrs")
    return a.separation(b).to(u.arcmin).value


def impact_parameter_kpc(theta_arcmin, z, cosmo: Cosmology):
    """Proper impact parameter b (kpc) at redshift z for angular offset theta.

    b = theta * D_A(z); accepts scalars or arrays.
    """
    theta_rad = (theta_arcmin * u.arcmin).to(u.rad).value
    da = cosmo.angular_diameter_distance(z).to(u.kpc).value
    return theta_rad * da


def compute_rdelta_from_mdelta(m_delta_msun: float, delta: float, z: float, cosmo: Cosmology) -> float:
    """R_delta (kpc) from M_delta = (4/3) pi delta rho_c(z) R_delta^3."""
    rho_c = cosmo.critical_density(z).to(u.Msun / (u.kpc**3)).value
    return ((3.0 * m_delta_msun) / (4.0 * math.pi * delta * rho_c)) ** (1.0 / 3.0)


def merge_and_rank(
    df: pd.DataFrame,
    frb_ra_deg: float,
    frb_dec_deg: float,
    cosmo: Cosmology,
) -> pd.DataFrame:
    """Compute theta (arcmin), b (kpc), optional R_delta, and rank by b/R_delta when available.

    Assumes df has columns: ra, dec, z, and optionally delta_def, m_delta, r_delta.
    """
    out = df.copy()

    out["theta_arcmin"] = [
        angular_separation_arcmin(ra, dec, frb_ra_deg, frb_dec_deg) if pd.notna(ra) and pd.notna(dec) else np.nan
        for ra, dec in zip(out["ra"], out["dec"], strict=True)
    ]

    out["b_kpc"] = [
        impact_parameter_kpc(theta, z, cosmo) if pd.notna(theta) and pd.notna(z) else np.nan
        for theta, z in zip(out["theta_arcmin"], out["z"], strict=True)
    ]

    def _compute_r(row):
        if pd.notna(row.get("r_delta")):
            return row["r_delta"]
        if pd.notna(row.get("m_delta")) and pd.notna(row.get("delta_def")) and pd.notna(row.get("z")):
            try:
                return compute_rdelta_from_mdelta(float(row["m_delta"]), float(row["delta_def"]), float(row["z"]), cosmo)
            except Exception:
                return np.nan
        return np.nan

    out["r_delta_computed"] = out.apply(_compute_r, axis=1)

    # Ranking key: prefer smaller b/R when R available; otherwise smaller b
    def _rank_key(row):
        r = row["r_delta_computed"]
        b = row["b_kpc"]
        if pd.notna(r) and r > 0 and pd.notna(b):
            return b / r
        return b if pd.notna(b) else np.inf

    out["rank_key"] = out.apply(_rank_key, axis=1)
    out.sort_values(["rank_key", "service", "table", "id", "name"], inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out
