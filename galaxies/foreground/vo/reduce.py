"""Reduction and ranking for foreground candidates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Cosmology


def angular_separation_arcmin(
    frb_ra_deg: float,
    frb_dec_deg: float,
    cand_ra_deg: float,
    cand_dec_deg: float,
) -> float:
    """Compute angular separation in arcminutes."""
    frb = SkyCoord(frb_ra_deg * u.deg, frb_dec_deg * u.deg, frame="icrs")
    candidate = SkyCoord(cand_ra_deg * u.deg, cand_dec_deg * u.deg, frame="icrs")
    return frb.separation(candidate).to(u.arcmin).value


def impact_parameter_kpc(theta_arcmin: float, z: float, cosmo: Cosmology) -> float:
    """Compute projected impact parameter in kpc."""
    theta_rad = (theta_arcmin * u.arcmin).to(u.rad).value
    da_kpc = cosmo.angular_diameter_distance(z).to(u.kpc).value
    return theta_rad * da_kpc


def rdelta_from_mdelta(
    m_delta_msun: float,
    delta: float,
    z: float,
    cosmo: Cosmology,
) -> float:
    """Compute R_delta in kpc from M_delta without changing delta definition."""
    rho_c = cosmo.critical_density(z).to(u.Msun / (u.kpc**3)).value
    r_cubed = (3.0 * m_delta_msun) / (4.0 * math.pi * delta * rho_c)
    return r_cubed ** (1.0 / 3.0)


def merge_and_rank(
    df: pd.DataFrame,
    *,
    frb_ra_deg: float,
    frb_dec_deg: float,
    cosmo: Cosmology | None = None,
) -> pd.DataFrame:
    """Add geometry columns and rank candidates deterministically."""
    if cosmo is None:
        from astropy.cosmology import Planck18

        cosmo = Planck18
    out = df.copy()

    out["theta_arcmin"] = [
        angular_separation_arcmin(frb_ra_deg, frb_dec_deg, ra, dec)
        if pd.notna(ra) and pd.notna(dec)
        else np.nan
        for ra, dec in zip(out["ra"], out["dec"], strict=True)
    ]
    out["b_kpc"] = [
        impact_parameter_kpc(theta, z, cosmo) if pd.notna(theta) and pd.notna(z) else np.nan
        for theta, z in zip(out["theta_arcmin"], out["z"], strict=True)
    ]

    def _r_delta(row: pd.Series) -> float:
        if pd.notna(row.get("r_delta")):
            return float(row["r_delta"])
        if (
            pd.notna(row.get("m_delta"))
            and pd.notna(row.get("delta_def"))
            and pd.notna(row.get("z"))
        ):
            try:
                return rdelta_from_mdelta(
                    float(row["m_delta"]),
                    float(row["delta_def"]),
                    float(row["z"]),
                    cosmo,
                )
            except Exception:
                return np.nan
        return np.nan

    out["r_delta_computed"] = out.apply(_r_delta, axis=1)

    def _rank_key(row: pd.Series) -> float:
        b_kpc = row.get("b_kpc")
        r_delta = row.get("r_delta_computed")
        if pd.notna(b_kpc) and pd.notna(r_delta) and r_delta > 0:
            return float(b_kpc) / float(r_delta)
        if pd.notna(b_kpc):
            return float(b_kpc)
        return np.inf

    out["rank_key"] = out.apply(_rank_key, axis=1)
    out.sort_values(
        ["rank_key", "service", "table", "id", "name"],
        inplace=True,
        kind="mergesort",
        na_position="last",
    )
    out.reset_index(drop=True, inplace=True)
    return out
