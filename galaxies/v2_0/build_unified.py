"""Build unified per-galaxy CGM/scattering records from matched catalogs."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from galaxies.v2_0 import cgm_observables as cgm
from galaxies.v2_0 import config
from galaxies.v2_0 import enrichers
from galaxies.v2_0 import scattering_predict as scat
from galaxies.v2_0.generate_galaxy_plots import (
    estimate_halo_mass,
    estimate_logmstar_from_photometry,
    get_rvir_and_rs,
    nfw_enclosed_mass,
)


MASS_PRIORITY = (
    "glade_catalog",
    "xsc_kband",
    "desi_ls_sed",
    "wise_w1",
    "ps1_taylor",
    "assumed",
)


def _is_bad(value: Any) -> bool:
    if value is None:
        return True
    try:
        if np.ma.is_masked(value):
            return True
    except TypeError:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(numeric) or numeric <= -9990.0


def _has(row: pd.Series, name: str) -> bool:
    return name in row.index


def _num(row: pd.Series, *names: str) -> float:
    for name in names:
        if not _has(row, name):
            continue
        value = row[name]
        if not _is_bad(value):
            return float(value)
    return math.nan


def _text(row: pd.Series, name: str) -> str | None:
    if not _has(row, name):
        return None
    value = row[name]
    if value is None:
        return None
    try:
        if np.ma.is_masked(value):
            return None
    except TypeError:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    return text if text else None


def _finite(value: Any) -> bool:
    return not _is_bad(value)


def _or_nan(value: Any) -> float:
    return math.nan if _is_bad(value) else float(value)


def _boolish(value: Any) -> bool:
    if value is None:
        return False
    try:
        if np.ma.is_masked(value):
            return False
    except TypeError:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "1", "yes", "y"}
    if _is_bad(value):
        return False
    return bool(value)


def _distmod(z_gal: float) -> float:
    if _is_bad(z_gal) or float(z_gal) <= 0.0:
        return math.nan
    return float(config.COSMO.distmod(float(z_gal)).value)


def select_stellar_mass(row, z_gal) -> tuple[float, str, str]:
    """Return the best available log10 stellar mass and provenance."""
    catalog = (_text(row, "catalog") or "").lower()
    glade_mass = _num(row, "M_star")
    if "glade" in catalog and _finite(glade_mass):
        return glade_mass, "glade_catalog", "glade_catalog"

    if _finite(z_gal) and float(z_gal) > 0.0:
        xsc_kmag = _num(row, "xsc_kmag")
        if _finite(xsc_kmag):
            # Bell & de Jong 2001 place near-IR M/L_K near unity for normal
            # galaxies; with Willmer 2018 M_sun,K=3.28 Vega this gives an
            # explicit order-of-magnitude 2MASS prior when no SED fit exists.
            abs_k = xsc_kmag - _distmod(float(z_gal))
            if _finite(abs_k):
                return -0.4 * (abs_k - 3.28), "xsc_kband", "xsc_kband"

        desi_mass = cgm.stellar_mass_desi_gz(
            _num(row, "desi_ls_gmag"),
            _num(row, "desi_ls_zmag"),
            float(z_gal),
        )
        if _finite(desi_mass):
            return float(desi_mass), "desi_ls_sed", "desi_ls_gz_zibetti2009"

        wise_mass = cgm.stellar_mass_wise_w1(
            _num(row, "W1mag"),
            float(z_gal),
            w1_w2=_num(row, "wise_W1_W2"),
        )
        if _finite(wise_mass):
            return float(wise_mass), "wise_w1", "wise_w1_cluver2014"

        ps1_mass = estimate_logmstar_from_photometry(row, float(z_gal))
        if _finite(ps1_mass):
            return float(ps1_mass), "ps1_taylor", "ps1_taylor2011"

    return 10.0, "assumed", "assumed_default"


def _empty_with_unified_columns(matches: pd.DataFrame) -> pd.DataFrame:
    out = matches.copy()
    for column in _UNIFIED_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype=object if column in _OBJECT_COLUMNS else "float64")
    return out


_OBJECT_COLUMNS = {
    "mass_source",
    "mass_method",
    "morph_type",
    "is_star_forming",
    "wise_agn_flag",
    "desi_is_agn",
    "bpt_class",
    "group_member",
    "group_n",
    "intersects_rvir",
    "sfr_is_limit",
    "cgm_extractable_flags",
    "z_source",
}

_UNIFIED_COLUMNS = (
    "z_source",
    "z_err",
    "logM_best",
    "mass_source",
    "mass_method",
    "M_halo",
    "logM_halo",
    "R_vir_kpc",
    "r_s",
    "c",
    "b_over_rvir",
    "intersects_rvir",
    "gi_color",
    "gr_color",
    "morph_type",
    "sersic",
    "shape_r_kpc",
    "axis_ratio",
    "position_angle_deg",
    "inclination_deg",
    "azimuthal_phi_deg",
    "is_star_forming",
    "W1mag",
    "W2mag",
    "W3mag",
    "W4mag",
    "wise_W1_W2",
    "wise_agn_flag",
    "galex_fuv",
    "galex_nuv",
    "galex_ebv",
    "sfr_w3",
    "sfr_uv",
    "sfr_best",
    "sfr_is_limit",
    "metallicity_12logOH",
    "desi_oii_flux",
    "desi_halpha_flux",
    "desi_is_agn",
    "bpt_class",
    "group_member",
    "group_n",
    "pred_mgii_wr",
    "cool_fc",
    "cool_fc_lo",
    "cool_fc_hi",
    "dm_halo",
    "dm_cool",
    "f_tilde",
    "f_tilde_lo",
    "f_tilde_hi",
    "g_scatt",
    "pred_tau_scat_ms_1GHz",
    "pred_tau_scat_ms_1GHz_lo",
    "pred_tau_scat_ms_1GHz_hi",
    "pred_scint_bw_khz",
    "scattering_rank",
    "cgm_extractable_flags",
)


def build_unified_records(matches, z_frb, sight_ra, sight_dec, enrich=True, enrich_fn=None) -> pd.DataFrame:
    """Build one unified per-galaxy DataFrame from foreground matches."""
    if matches is None:
        matches = pd.DataFrame()
    if len(matches) == 0:
        return _empty_with_unified_columns(matches)

    if enrich:
        if enrich_fn is None:
            enrich_fn = enrichers.enrich_all_catalogs
        out = enrich_fn(matches)
    else:
        out = matches.copy()
    out = out.copy()

    records: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        z_gal = _num(row, "z")
        impact_kpc = _num(row, "impact_kpc")
        logm_best, mass_source, mass_method = select_stellar_mass(row, z_gal)

        m_halo = logm_halo = r_vir = r_s = concentration = math.nan
        if _finite(logm_best):
            try:
                m_halo = float(estimate_halo_mass(logm_best))
                logm_halo = math.log10(m_halo)
                r_vir, r_s, concentration = get_rvir_and_rs(m_halo, z_gal)
            except (ArithmeticError, ValueError, RuntimeError, OverflowError):
                m_halo = logm_halo = r_vir = r_s = concentration = math.nan

        b_over_rvir = impact_kpc / r_vir if _finite(impact_kpc) and _finite(r_vir) and r_vir > 0.0 else math.nan
        intersects_rvir = bool(_finite(impact_kpc) and _finite(r_vir) and impact_kpc <= r_vir)

        gmag = _num(row, "gmag")
        imag = _num(row, "imag")
        gi_color = gmag - imag if _finite(gmag) and _finite(imag) else math.nan
        desi_g = _num(row, "desi_ls_gmag")
        desi_r = _num(row, "desi_ls_rmag")
        gr_color = desi_g - desi_r if _finite(desi_g) and _finite(desi_r) else math.nan

        shape_r_arcsec = _num(row, "desi_ls_shape_r")
        shape_r_kpc = math.nan
        if _finite(shape_r_arcsec) and _finite(z_gal) and z_gal > 0.0:
            shape_r_kpc = (
                shape_r_arcsec
                * config.COSMO.kpc_proper_per_arcmin(z_gal).to(u.kpc / u.arcmin).value
                / 60.0
            )

        e1 = _num(row, "shape_e1", "desi_ls_shape_e1")
        e2 = _num(row, "shape_e2", "desi_ls_shape_e2")
        axis_ratio = math.nan
        position_angle = math.nan
        if _finite(e1) and _finite(e2):
            axis_ratio = _or_nan(cgm.axis_ratio_from_ellipticity(e1, e2))
            position_angle = _or_nan(cgm.position_angle_deg(e1, e2))
        if not _finite(axis_ratio):
            axis_ratio = _num(row, "xsc_axis_ratio")
        if not _finite(position_angle):
            position_angle = _num(row, "xsc_pa")

        inclination = _or_nan(cgm.inclination_deg(axis_ratio)) if _finite(axis_ratio) else math.nan
        gal_ra = _num(row, "ra")
        gal_dec = _num(row, "dec")
        phi = (
            _or_nan(cgm.azimuthal_angle_phi_deg(gal_ra, gal_dec, position_angle, sight_ra, sight_dec))
            if _finite(position_angle)
            else math.nan
        )

        is_star_forming = bool(cgm.is_star_forming(gr_color, logm_best))
        w1mag = _num(row, "W1mag")
        w2mag = _num(row, "W2mag")
        w3mag = _num(row, "W3mag")
        w4mag = _num(row, "W4mag")
        wise_w1_w2 = _num(row, "wise_W1_W2")
        wise_agn_flag = bool(cgm.wise_agn_stern2012(wise_w1_w2))
        galex_fuv = _num(row, "galex_fuv")
        galex_nuv = _num(row, "galex_nuv")
        galex_ebv = _num(row, "galex_ebv")

        sfr_w3 = _or_nan(cgm.sfr_wise_w3(w3mag, z_gal))
        sfr_uv = _or_nan(cgm.sfr_uv_nuv(galex_nuv, z_gal, ebv=galex_ebv if _finite(galex_ebv) else 0.0))
        sfr_best = sfr_w3 if _finite(sfr_w3) else sfr_uv
        sfr_is_limit = not (_finite(w3mag) or _finite(galex_nuv))
        metallicity = _or_nan(cgm.metallicity_mzr(logm_best, z_gal))

        desi_is_agn = _text(row, "desi_is_agn")
        if desi_is_agn is None and _has(row, "desi_is_agn"):
            desi_is_agn = row["desi_is_agn"]
        agn_for_scattering = wise_agn_flag or _boolish(desi_is_agn)

        pred_mgii_wr = _or_nan(scat.predict_mgii_wr(impact_kpc, logm_best))
        cool_fc = cool_fc_lo = cool_fc_hi = math.nan
        try:
            cool_fc, cool_fc_lo, cool_fc_hi = scat.cool_covering_fraction(
                b_over_rvir,
                logm_best,
                is_star_forming,
                phi_deg=phi,
            )
        except (ArithmeticError, ValueError, RuntimeError, OverflowError):
            pass

        dm_halo = _or_nan(scat.dm_halo_mnfw(m_halo, z_gal, impact_kpc))
        dm_cool = _or_nan(scat.dm_cool(dm_halo, cool_fc, mgii_wr=pred_mgii_wr))
        f_tilde, f_tilde_lo, f_tilde_hi = scat.f_tilde_prior(
            sfr_best,
            metallicity_12logOH=metallicity,
            agn=agn_for_scattering,
        )
        g_scatt = scat.g_scatt(z_gal, z_frb)
        # g_scatt == 0 means the intervening-screen geometry has no leverage:
        # the galaxy is at/behind the FRB (z_gal >= z_frb) or its redshift is
        # invalid. In that case the screen model cannot predict burst scattering,
        # so tau is reported as "not predictable" (NaN) rather than a literal 0
        # ("no scattering"). Foreground galaxies (g_scatt > 0) keep a finite
        # prediction even when the stellar mass is an assumed default.
        screen_predictable = _finite(g_scatt) and float(g_scatt) > 0.0
        if screen_predictable:
            tau = _or_nan(scat.tau_scat_ms(f_tilde, g_scatt, dm_halo, z_gal, nu_ghz=1.0))
            tau_lo = _or_nan(scat.tau_scat_ms(f_tilde_lo, g_scatt, dm_halo, z_gal, nu_ghz=1.0))
            tau_hi = _or_nan(scat.tau_scat_ms(f_tilde_hi, g_scatt, dm_halo, z_gal, nu_ghz=1.0))
            scint_bw = _or_nan(scat.scint_bandwidth_khz(tau))
        else:
            tau = tau_lo = tau_hi = scint_bw = math.nan

        desi_emission_measured = _has(row, "desi_emission_matched") and _boolish(row["desi_emission_matched"])
        flags = {
            "stellar_mass": "MEASURED" if mass_source in set(MASS_PRIORITY[:-1]) else "PREDICTED",
            "morphology": "MEASURED" if _finite(axis_ratio) else "NOT_AVAILABLE",
            "sfr": "MEASURED" if (_finite(w3mag) or _finite(galex_nuv)) else "NOT_AVAILABLE",
            "wise": "MEASURED" if _finite(w1mag) else "NOT_AVAILABLE",
            "desi_spectro": "MEASURED" if desi_emission_measured else "NOT_AVAILABLE",
            "dm_halo": "PREDICTED",
            "dm_cool": "PREDICTED",
            "tau_scat": "PREDICTED" if screen_predictable else "NOT_PREDICTABLE",
            "cool_covering": "PREDICTED",
        }

        records.append(
            {
                "z_source": _text(row, "catalog"),
                "z_err": _num(row, "z_best_err", "e_zphot", "z_phot_err"),
                "logM_best": logm_best,
                "mass_source": mass_source,
                "mass_method": mass_method,
                "M_halo": m_halo,
                "logM_halo": logm_halo,
                "R_vir_kpc": r_vir,
                "r_s": r_s,
                "c": concentration,
                "b_over_rvir": b_over_rvir,
                "intersects_rvir": intersects_rvir,
                "gi_color": gi_color,
                "gr_color": gr_color,
                "morph_type": _text(row, "desi_ls_type") or _text(row, "type"),
                "sersic": _num(row, "desi_ls_sersic"),
                "shape_r_kpc": shape_r_kpc,
                "axis_ratio": axis_ratio,
                "position_angle_deg": position_angle,
                "inclination_deg": inclination,
                "azimuthal_phi_deg": phi,
                "is_star_forming": is_star_forming,
                "W1mag": w1mag,
                "W2mag": w2mag,
                "W3mag": w3mag,
                "W4mag": w4mag,
                "wise_W1_W2": wise_w1_w2,
                "wise_agn_flag": wise_agn_flag,
                "galex_fuv": galex_fuv,
                "galex_nuv": galex_nuv,
                "galex_ebv": galex_ebv,
                "sfr_w3": sfr_w3,
                "sfr_uv": sfr_uv,
                "sfr_best": sfr_best,
                "sfr_is_limit": sfr_is_limit,
                "metallicity_12logOH": metallicity,
                "desi_oii_flux": _num(row, "desi_oii_flux"),
                "desi_halpha_flux": _num(row, "desi_halpha_flux"),
                "desi_is_agn": desi_is_agn,
                "bpt_class": None,
                "group_member": False,
                "group_n": 0,
                "pred_mgii_wr": pred_mgii_wr,
                "cool_fc": cool_fc,
                "cool_fc_lo": cool_fc_lo,
                "cool_fc_hi": cool_fc_hi,
                "dm_halo": dm_halo,
                "dm_cool": dm_cool,
                "f_tilde": f_tilde,
                "f_tilde_lo": f_tilde_lo,
                "f_tilde_hi": f_tilde_hi,
                "g_scatt": g_scatt,
                "pred_tau_scat_ms_1GHz": tau,
                "pred_tau_scat_ms_1GHz_lo": tau_lo,
                "pred_tau_scat_ms_1GHz_hi": tau_hi,
                "pred_scint_bw_khz": scint_bw,
                "cgm_extractable_flags": flags,
            }
        )

    unified = out.reset_index(drop=True).copy()
    derived = pd.DataFrame(records)
    for column in derived.columns:
        unified[column] = derived[column]
    unified["scattering_rank"] = _scattering_ranks(unified["pred_tau_scat_ms_1GHz"])
    return unified


def _scattering_ranks(values: pd.Series) -> list[int]:
    numeric = pd.to_numeric(values, errors="coerce")
    order = sorted(range(len(numeric)), key=lambda i: (not np.isfinite(numeric.iloc[i]), -numeric.iloc[i] if np.isfinite(numeric.iloc[i]) else 0.0, i))
    ranks = [0] * len(numeric)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def build_for_target(name, ra_str, dec_str, z_frb, results_dir="results", enrich=True) -> pd.DataFrame:
    """Read a target match CSV, write its lower-case unified CSV, and return it."""
    base = name.lower()
    input_path = os.path.join(results_dir, f"{base}_galaxies.csv")
    if not os.path.exists(input_path):
        print(f"Warning: missing input CSV {input_path}")
        return pd.DataFrame()

    matches = pd.read_csv(input_path)
    sight = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    unified = build_unified_records(
        matches,
        z_frb=z_frb,
        sight_ra=sight.ra.deg,
        sight_dec=sight.dec.deg,
        enrich=enrich,
    )
    unified.to_csv(os.path.join(results_dir, f"{base}_unified.csv"), index=False)
    return unified


def build_all(results_dir="results", enrich=True) -> dict[str, pd.DataFrame]:
    """Build unified records for every configured target."""
    return {
        name: build_for_target(name, ra_str, dec_str, z_frb, results_dir=results_dir, enrich=enrich)
        for name, ra_str, dec_str, z_frb in config.TARGETS
    }


__all__ = [
    "MASS_PRIORITY",
    "build_all",
    "build_for_target",
    "build_unified_records",
    "nfw_enclosed_mass",
    "select_stellar_mass",
]
