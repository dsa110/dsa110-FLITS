"""Pure matching and physics helpers for the expanded foreground catalog.

These functions do not perform network I/O and do not alter the adjudicated
foreground verdicts.  Numerical conventions follow the papers named in each
function and use the pipeline Planck cosmology for ``R200c``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18


@dataclass(frozen=True)
class CatalogMatch:
    status: str
    selected_row: Mapping[str, Any] | None
    selected_index: int | None
    separation_arcsec: float | None
    candidate_count: int
    second_separation_arcsec: float | None
    query_error: str | None = None

    @property
    def selected_id(self) -> str | None:
        return _identifier(self.selected_row) if self.selected_row is not None else None


@dataclass(frozen=True)
class DerivedValue:
    value: float | None
    error: float | None
    status: str
    method: str
    units: str
    authority: str

    @property
    def uncertainty(self) -> float | None:
        return self.error


def _separation(row: Mapping[str, Any], target: Any) -> float:
    for name in ("separation_arcsec", "_r"):
        value = row.get(name)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    if target is None:
        raise ValueError("catalog row lacks a finite angular separation")
    target_coord = (
        target
        if isinstance(target, SkyCoord)
        else SkyCoord(float(target[0]) * u.deg, float(target[1]) * u.deg)
    )
    ra = row.get("ra_deg", row.get("match_ra_deg"))
    dec = row.get("dec_deg", row.get("match_dec_deg"))
    if ra is None or dec is None:
        raise ValueError("catalog row lacks coordinates and angular separation")
    row_coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
    return float(target_coord.separation(row_coord).arcsec)


def _identifier(row: Mapping[str, Any]) -> str:
    for name in ("id", "catalog_id", "GSC2", "AllWISE", "Name", "objID"):
        value = row.get(name)
        if value is not None:
            return str(value)
    return ""


def select_match(
    rows: Sequence[Mapping[str, Any]] | None,
    target: Any,
    radius_arcsec: float,
    ambiguity_arcsec: float,
    *,
    query_error: str | None = None,
) -> CatalogMatch:
    """Select the nearest row with a deterministic, fail-closed ambiguity rule.

    Rows are ordered by exact spherical separation and then stable catalog ID.
    A second candidate within ``ambiguity_arcsec`` of the nearest separation
    makes the result ambiguous; the deterministic first row is retained only
    for audit, not treated as a secure match. ``target`` is accepted for the
    public contract; normalized snapshots already carry exact separations.
    """
    if query_error is not None:
        return CatalogMatch("query_error", None, None, None, 0, None, query_error)
    if not rows:
        return CatalogMatch("unmatched", None, None, None, 0, None)

    indexed = [(idx, row, _separation(row, target)) for idx, row in enumerate(rows)]
    inside = [item for item in indexed if item[2] <= float(radius_arcsec)]
    if not inside:
        return CatalogMatch("unmatched", None, None, None, 0, None)
    ordered = sorted(inside, key=lambda item: (item[2], _identifier(item[1])))
    idx, row, nearest = ordered[0]
    second = ordered[1][2] if len(ordered) > 1 else None
    ambiguous = second is not None and (second - nearest) <= float(ambiguity_arcsec)
    return CatalogMatch(
        "ambiguous" if ambiguous else "matched",
        row,
        idx,
        nearest,
        len(ordered),
        second,
    )


def cluver14_log_mstar(
    w1: float,
    w2: float,
    distance_modulus: float,
    *,
    rest_frame: bool,
    valid_photometry: bool,
    w1_error: float | None = None,
    w2_error: float | None = None,
    distance_modulus_error: float | None = None,
) -> DerivedValue:
    """Cluver et al. (2014) Equation 2 diagnostic stellar mass.

    The returned uncertainty propagates the supplied independent measurement
    errors. The paper-relation intrinsic scatter is not silently invented; it
    remains outside this measurement-only error and is named in ``method``.
    """
    method = "Cluver14_Eq2_measurement_error_only"
    base = dict(method=method, units="dex(log10_Msun)", authority="diagnostic")
    values = (w1, w2, distance_modulus)
    if not rest_frame:
        return DerivedValue(None, None, "not_rest_frame", **base)
    if not valid_photometry or not all(np.isfinite(float(v)) for v in values):
        return DerivedValue(None, None, "invalid_photometry", **base)
    errors = (w1_error, w2_error, distance_modulus_error)
    if any(v is None or not np.isfinite(float(v)) or float(v) < 0 for v in errors):
        return DerivedValue(None, None, "missing_uncertainty", **base)

    absolute_w1 = float(w1) - float(distance_modulus)
    log_l_w1 = -0.4 * (absolute_w1 - 3.24)
    value = log_l_w1 - 2.54 * (float(w1) - float(w2)) - 0.17
    error = np.sqrt(
        (2.94 * float(w1_error)) ** 2
        + (2.54 * float(w2_error)) ** 2
        + (0.4 * float(distance_modulus_error)) ** 2
    )
    return DerivedValue(float(value), float(error), "pass", **base)


def stern12_status(w1_minus_w2: float, w2: float, *, color_valid: bool) -> str:
    """Classify the Stern et al. (2012) WISE AGN selection domain."""
    if not color_valid or not np.isfinite(w1_minus_w2) or not np.isfinite(w2):
        return "insufficient_color"
    if float(w2) > 15.05:
        return "outside_validated_depth"
    if float(w1_minus_w2) >= 0.8:
        return "selected_by_stern12"
    return "not_selected_within_depth"


def m200c_to_r200c(m200c_msun: float, z: float) -> float:
    """Return ``R200c`` in proper kpc from the critical-density definition."""
    if not np.isfinite(m200c_msun) or float(m200c_msun) <= 0:
        raise ValueError("m200c_msun must be positive and finite")
    if not np.isfinite(z) or float(z) < 0:
        raise ValueError("z must be non-negative and finite")
    rho_c = rho_critical_msun_kpc3(float(z))
    return float((3.0 * float(m200c_msun) / (4.0 * np.pi * 200.0 * rho_c)) ** (1.0 / 3.0))


def rho_critical_msun_kpc3(z: float) -> float:
    """Planck18 critical density in solar masses per proper cubic kpc."""
    if not np.isfinite(z) or float(z) < 0:
        raise ValueError("z must be non-negative and finite")
    return float(Planck18.critical_density(float(z)).to_value(u.Msun / u.kpc**3))


def dutton_maccio14_c200c(m200c_msun: float, z: float) -> float:
    """Dutton & Macciò (2014) Planck ``c200c`` redshift evolution.

    Their published Planck calibration fixes ``h=0.671``. This is a fit
    parameter, not the cosmology used to evaluate ``rho_critical(z)``.
    """
    if not np.isfinite(m200c_msun) or float(m200c_msun) <= 0:
        raise ValueError("mass must be positive and finite")
    if not np.isfinite(z) or float(z) < 0:
        raise ValueError("z must be non-negative and finite")
    zf = float(z)
    b = -0.101 + 0.026 * zf
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * zf**1.21)
    log_c = a + b * np.log10(float(m200c_msun) * 0.671 / 1.0e12)
    return float(10.0**log_c)
