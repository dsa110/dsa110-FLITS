#!/usr/bin/env python3
"""Per-sightline DM and scattering *budget* attribution for the FRB sample.

For each FRB sightline this assembles the observed burst features (dispersion
measure DM and, where fit, the scattering time tau) and decomposes them into the
media most likely responsible:

DM budget (pc/cm^3, observer frame):
  DM_obs        observed burst DM (from the burst id encoded in its filename)
  DM_MW_ISM     Milky-Way disk ISM, NE2001 (pygedm); YMW16 reported alongside
  DM_MW_halo    Milky-Way hot halo, constant prior (Yamasaki & Totani 2020)
  DM_cosmic     <DM_cosmic>(z) Macquart relation mean (pure astropy)
  DM_interv     sum over foreground galaxies of mNFW hot + cool CGM columns
  DM_host       residual = DM_obs - the above (host galaxy, observer frame)

Scattering budget (ms at 1 GHz):
  tau_obs       measured burst scattering (scat_analysis fit, where available)
  tau_MW        Galactic scattering, NE2001 (pygedm) -- negligible at |b| here
  tau_interv    sum over foreground galaxies of the predicted two-phase screen
  tau_host      not directly predictable; the likely residual

The attribution verdict compares the predicted intervening contribution to the
measurement: a closely-intersecting, midpoint, cool-gas-bearing foreground
galaxy whose predicted tau approaches tau_obs is a coherent case that an
intervening galaxy scatters the burst. Where tau_interv << tau_obs the
scattering must be host/Milky-Way.

DM_cosmic, parse_dm_obs, and read_measured_tau_ms are pure. The Galactic model
(galactic_dm_tau) is offline but uses the compiled pygedm NE2001/YMW16 models;
it is injectable so the test suite stays network- and pygedm-free.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from . import config
    from . import scattering_predict as scat
    from .build_unified import MASS_PRIORITY, build_unified_records
except ImportError:  # pragma: no cover - supports direct script execution.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from galaxies.v2_0 import config
    from galaxies.v2_0 import scattering_predict as scat
    from galaxies.v2_0.build_unified import MASS_PRIORITY, build_unified_records


# Every MASS_PRIORITY source except the trailing "assumed" default is a real
# measured/derived stellar mass.
_MEASURED_MASS_SOURCES = frozenset(MASS_PRIORITY[:-1])


# Below this scaled impact the smooth mNFW hot-halo column is extrapolated
# through the galaxy interior (not the CGM it was calibrated for) and the DM
# blows up; the "capped" intervening DM evaluates the column at this floor so a
# physically bounded value can be reported alongside the raw one.
INTERIOR_B_OVER_RVIR = 0.1

# In this sample z_frb == 1.0 exactly is a placeholder for an unmeasured host
# redshift (Freya/Mahi/Johndoeii). Without a real z the Macquart <DM_cosmic> and
# the foreground/background split are meaningless, so those budgets are flagged
# and the cosmic/host terms are withheld rather than presented as if real.
PLACEHOLDER_Z = 1.0


def _is_placeholder_z(z: float) -> bool:
    return math.isfinite(_f(z)) and abs(float(z) - PLACEHOLDER_Z) < 1.0e-6


# Yamasaki & Totani 2020 ApJ 888,105 give a Galactic hot-halo DM ~43 pc/cm^3
# along typical sightlines; 40 is a round, deliberately conservative prior.
DM_MW_HALO = 40.0

# Macquart+2020 Nature 581,391 / Deng & Zhang 2014: f_IGM ~ 0.84 of cosmic
# baryons are in the diffuse ionized IGM, with electron fraction chi_e ~ 7/8
# (H + singly-ionized He).
F_IGM = 0.84
CHI_E = 0.875


def _f(value: Any) -> float:
    """Coerce to float, returning NaN for missing/non-finite/sentinel values."""
    if value is None:
        return math.nan
    try:
        if np.ma.is_masked(value):
            return math.nan
    except TypeError:
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(out) or out <= -9990.0:
        return math.nan
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "1", "yes", "y"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def dm_cosmic_macquart(z: float, f_igm: float = F_IGM, chi_e: float = CHI_E) -> float:
    """Return the Macquart-relation mean cosmic/IGM DM at redshift z (pc/cm^3)."""
    z_value = float(z)
    if not math.isfinite(z_value) or z_value <= 0.0:
        return 0.0

    from scipy.integrate import quad

    # Deng & Zhang 2014 ApJ 783,L35 / Macquart+2020 Nature 581,391:
    # <DM_cosmic> = n_e,0 * (c/H0) * int_0^z (1+z')/E(z') dz', with the present
    # diffuse-IGM electron density n_e,0 = f_IGM chi_e Omega_b rho_crit,0 / m_p.
    n_e0 = (f_igm * chi_e * config.COSMO.Ob0 * config.COSMO.critical_density0 / const.m_p).to(u.cm**-3)
    hubble_dist = (const.c / config.COSMO.H0).to(u.pc)
    integral, _ = quad(
        lambda zp: (1.0 + zp) / math.sqrt(config.COSMO.Om0 * (1.0 + zp) ** 3 + config.COSMO.Ode0),
        0.0,
        z_value,
    )
    dm = (n_e0 * hubble_dist * integral).to(u.pc / u.cm**3).value
    return float(dm)


def parse_dm_obs(path_or_name: str | None) -> float | None:
    """Extract the observed DM from a burst filename/id.

    The codetection bursts encode DM as the integer token after the Stokes flag,
    e.g. ``casey_chime_I_491_2085_32000b_cntr_bpc.npy`` -> 491 (DSA paths used a
    lowercase 'l' where Stokes 'I' was intended).
    """
    if not path_or_name:
        return None
    import re

    stem = os.path.basename(str(path_or_name))
    m = re.search(r"_[IlL]_(\d+)_\d+_\d+b", stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def read_measured_tau_ms(fit_json_path: str | None) -> float | None:
    """Return tau_1ghz (ms) from a scat_analysis fit_results.json, else None.

    This is a raw parser: it returns the best-fit tau regardless of fit quality.
    Use read_tau_fit for the quality-gated value with uncertainties.
    """
    if not fit_json_path or not os.path.exists(fit_json_path):
        return None
    try:
        with open(fit_json_path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    params = data.get("best_params", {}) if isinstance(data, dict) else {}
    tau = params.get("tau_1ghz")
    tau_f = _f(tau)
    if not math.isfinite(tau_f) or tau_f <= 0.0:
        return None
    return float(tau_f)


def read_tau_fit(fit_json_path: str | None) -> dict | None:
    """Read tau_1ghz with uncertainty and quality flag from a fit_results.json.

    Prefers the posterior median + 16/84 from best_params_percentiles (the
    uncertainty-bearing summary); falls back to the best_params point estimate.
    Returns {tau, err_minus, err_plus, quality_flag, chi2_reduced} or None when no
    usable tau is present. The quality_flag is the pipeline's recalibrated
    chi2-driven gate (PASS/MARGINAL/FAIL); callers decide whether to ingest.
    """
    if not fit_json_path or not os.path.exists(fit_json_path):
        return None
    try:
        with open(fit_json_path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    err_minus = err_plus = math.nan
    tau_f = math.nan
    pct = data.get("best_params_percentiles")
    if isinstance(pct, dict) and isinstance(pct.get("tau_1ghz"), dict):
        tau_pct = pct["tau_1ghz"]
        tau_f = _f(tau_pct.get("median"))
        err_minus = _f(tau_pct.get("err_minus"))
        err_plus = _f(tau_pct.get("err_plus"))
    if not math.isfinite(tau_f):
        params = data.get("best_params", {}) or {}
        tau_f = _f(params.get("tau_1ghz"))
    if not math.isfinite(tau_f) or tau_f <= 0.0:
        return None

    gof = data.get("goodness_of_fit", {}) or {}
    return {
        "tau": float(tau_f),
        "err_minus": err_minus,
        "err_plus": err_plus,
        "quality_flag": gof.get("quality_flag"),
        "chi2_reduced": _f(gof.get("chi2_reduced")),
    }


_PYGEDM: Any = None


def _load_pygedm() -> Any:
    """Import pygedm once, shimming the SciPy>=1.14 removal of integrate.simps."""
    global _PYGEDM
    if _PYGEDM is not None:
        return _PYGEDM
    try:
        import scipy.integrate as _si

        if not hasattr(_si, "simps"):
            # pygedm.yt2020 still calls the removed scipy.integrate.simps; the
            # successor simpson made x keyword-only, so wrap rather than alias.
            _si.simps = lambda y, x=None, *a, **k: _si.simpson(y, x=x, *a, **k)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pygedm
        _PYGEDM = pygedm
    except Exception:
        _PYGEDM = False
    return _PYGEDM


def galactic_dm_tau(l_deg: float, b_deg: float, method: str = "ne2001", dist_pc: float = 30000.0):
    """Return (DM_NE2001, DM_YMW16, tau_MW_ms) toward Galactic (l, b).

    Uses the compiled pygedm NE2001/YMW16 models (offline). Returns NaNs if
    pygedm is unavailable so callers degrade gracefully.
    """
    pg = _load_pygedm()
    if not pg:
        return (math.nan, math.nan, math.nan)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dm_ne, tau_ne = pg.dist_to_dm(l_deg, b_deg, dist_pc, method="ne2001")
        dm_yw, tau_yw = pg.dist_to_dm(l_deg, b_deg, dist_pc, method="ymw16")
    tau = tau_ne if method == "ne2001" else tau_yw
    try:
        tau_ms = float(tau.to(u.ms).value)
    except (AttributeError, u.UnitConversionError):
        tau_ms = float(tau) * 1.0e3
    return (float(dm_ne.value), float(dm_yw.value), tau_ms)


def _find_burst_config_path(name: str, configs_dir: str) -> str | None:
    """Return the data path string from a target's CHIME burst config yaml."""
    if not configs_dir or not os.path.isdir(configs_dir):
        return None
    target = name.lower()
    for fn in os.listdir(configs_dir):
        if not fn.lower().endswith((".yaml", ".yml")):
            continue
        if not fn.lower().startswith(target):
            continue
        try:
            with open(os.path.join(configs_dir, fn)) as fh:
                for line in fh:
                    if line.strip().startswith("path:"):
                        return line.split("path:", 1)[1].strip()
        except OSError:
            continue
    return None


def _lookup_dm_obs(name: str, configs_dir: str | None) -> float | None:
    if configs_dir is None:
        configs_dir = os.path.join("scattering", "configs", "bursts", "chime")
    return parse_dm_obs(_find_burst_config_path(name, configs_dir))


def _lookup_tau_fit(name: str, bursts_dir: str | None) -> dict | None:
    """Find the best CHIME scattering fit for a target across result locations.

    Returns the read_tau_fit dict, preferring a quality-PASS fit when several are
    present (so a re-run that supersedes an old FAIL is picked up). Returns None
    if no fit_results.json with a usable tau exists.
    """
    target = name.lower()
    candidates: list[str] = []
    search_dirs = []
    if bursts_dir:
        search_dirs.append(os.path.join(bursts_dir, target))
        search_dirs.append(bursts_dir)
    search_dirs.append(os.path.join("scattering", "scat_process"))
    seen = set()
    for d in search_dirs:
        for pat in (f"{target}*chime*fit_results.json", f"{target}*fit_results.json"):
            for path in sorted(glob.glob(os.path.join(d, pat))):
                if path not in seen:
                    seen.add(path)
                    candidates.append(path)

    best = None
    for path in candidates:
        fit = read_tau_fit(path)
        if fit is None:
            continue
        if fit.get("quality_flag") == "PASS":
            return fit
        if best is None:
            best = fit
    return best


def _scattering_verdict(tau_obs: float, tau_interv: float, tau_interv_hi: float, n_fg: int) -> str:
    if not math.isfinite(tau_obs) or tau_obs <= 0.0:
        return f"no scattering measurement (predicted intervening tau={tau_interv:.2g} ms)"
    if n_fg == 0:
        return "no intervening screen; scattering is host / Milky-Way"
    ratio = tau_interv / tau_obs
    ratio_hi = (tau_interv_hi / tau_obs) if math.isfinite(tau_interv_hi) else ratio
    if ratio >= 0.5:
        return f"intervening galaxy plausibly dominates (pred/obs={ratio:.2f})"
    if ratio_hi >= 0.5:
        return f"intervening galaxy may contribute (prior-upper pred/obs={ratio_hi:.2f})"
    if ratio >= 0.05:
        return f"intervening subdominant; host / Milky-Way dominates (pred/obs={ratio:.2g})"
    return f"intervening negligible; host / Milky-Way dominated (pred/obs={ratio:.2g})"


def _dm_verdict(dm_obs, dm_mw_ism, dm_mw_halo, dm_cosmic, dm_interv, dm_host) -> str:
    if not math.isfinite(dm_obs):
        return "no observed DM"
    if dm_host < -0.05 * max(dm_obs, 1.0):
        return "over-budget (MW+cosmic mean exceed observed; sightline below cosmic mean)"
    comps = {
        "Milky-Way": _nz(dm_mw_ism) + dm_mw_halo,
        "cosmic/IGM": dm_cosmic,
        "intervening galaxy": dm_interv,
        "host": max(dm_host, 0.0),
    }
    dominant = max(comps, key=lambda k: comps[k])
    return f"{dominant}-dominated"


def _nz(x: float) -> float:
    """Treat NaN as 0 for additive budget closure (model-unavailable component)."""
    return x if math.isfinite(x) else 0.0


def build_sightline_budget(
    name: str,
    ra_str: str,
    dec_str: str,
    z_frb: float,
    *,
    results_dir: str = "results",
    configs_dir: str | None = None,
    bursts_dir: str | None = None,
    enrich: bool = False,
    dm_mw_fn: Callable | None = None,
    dm_obs: float | None = None,
    tau_obs: float | None = None,
    dm_mw_halo: float = DM_MW_HALO,
) -> dict:
    """Assemble the full DM + scattering budget for one FRB sightline."""
    sight = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    gal = sight.galactic
    l_deg, b_deg = float(gal.l.deg), float(gal.b.deg)

    if dm_mw_fn is None:
        dm_mw_fn = galactic_dm_tau
    dm_mw_ism, dm_mw_ism_ymw16, tau_mw_ms = dm_mw_fn(l_deg, b_deg, "ne2001")

    if dm_obs is None:
        dm_obs = _lookup_dm_obs(name, configs_dir)

    # Scattering measurement: gate ingestion on the recalibrated quality flag.
    # An injected tau_obs (tests / manual override) is trusted as-is; otherwise a
    # fit_results.json is ingested ONLY when quality_flag == "PASS". A present but
    # non-PASS fit is recorded (so the verdict can say "fit present but not
    # locked in") but its tau is withheld from the budget.
    tau_obs_err_minus = tau_obs_err_plus = math.nan
    tau_obs_chi2 = math.nan
    if tau_obs is not None:
        tau_obs_quality = "INJECTED"
    else:
        tau_obs_quality = None
        fit = _lookup_tau_fit(name, bursts_dir)
        if fit is not None:
            tau_obs_quality = fit.get("quality_flag") or "UNKNOWN"
            tau_obs_chi2 = _f(fit.get("chi2_reduced"))
            if fit.get("quality_flag") == "PASS":
                tau_obs = fit.get("tau")
                tau_obs_err_minus = _f(fit.get("err_minus"))
                tau_obs_err_plus = _f(fit.get("err_plus"))
            # non-PASS: tau_obs stays None (withheld), quality recorded above

    dm_hot = dm_cool = 0.0
    dm_hot_cap = dm_cool_cap = 0.0
    min_b_over_rvir = math.inf
    tau_int = tau_int_lo = tau_int_hi = 0.0
    n_fg = n_isect = 0
    dom: dict[str, Any] = {
        "best_z": math.nan, "best_b_over_rvir": math.nan, "best_g_scatt": math.nan,
        "best_cool_fc": math.nan, "best_logM": math.nan, "best_mass_source": None,
        "best_impact_kpc": math.nan,
    }
    best_tau = -1.0

    csv_path = os.path.join(results_dir, f"{name.lower()}_galaxies.csv")
    if os.path.exists(csv_path):
        matches = pd.read_csv(csv_path)
        unified = build_unified_records(
            matches, z_frb=z_frb, sight_ra=sight.ra.deg, sight_dec=sight.dec.deg, enrich=enrich
        )
        for _, row in unified.iterrows():
            z = _f(row.get("z"))
            if not (math.isfinite(z) and z < float(z_frb)):  # foreground galaxies only
                continue
            n_fg += 1
            if _truthy(row.get("intersects_rvir")):
                n_isect += 1
            dh_raw = _nz(_f(row.get("dm_halo")))
            dc_raw = _nz(_f(row.get("dm_cool")))
            dm_hot += dh_raw
            dm_cool += dc_raw

            # Capped intervening DM: re-evaluate the hot-halo column at the
            # CGM-regime floor max(b, 0.1 R_vir), then rescale the cool phase by
            # the same hot-column ratio (the cool term is sub-dominant).
            bor = _f(row.get("b_over_rvir"))
            if math.isfinite(bor):
                min_b_over_rvir = min(min_b_over_rvir, bor)
            m_halo = _f(row.get("M_halo"))
            rvir = _f(row.get("R_vir_kpc"))
            impact = _f(row.get("impact_kpc"))
            if math.isfinite(bor) and bor < INTERIOR_B_OVER_RVIR and all(
                math.isfinite(v) for v in (m_halo, rvir, impact)
            ) and rvir > 0.0 and dh_raw > 0.0:
                b_cap = max(impact, INTERIOR_B_OVER_RVIR * rvir)
                dh_cap = _nz(_f(scat.dm_halo_mnfw(m_halo, z, b_cap)))
                dc_cap = dc_raw * (dh_cap / dh_raw) if dh_raw > 0.0 else dc_raw
            else:
                dh_cap, dc_cap = dh_raw, dc_raw
            dm_hot_cap += dh_cap
            dm_cool_cap += dc_cap

            t = _f(row.get("pred_tau_scat_ms_1GHz"))
            if math.isfinite(t):
                tau_int += t
                tl = _f(row.get("pred_tau_scat_ms_1GHz_lo"))
                th = _f(row.get("pred_tau_scat_ms_1GHz_hi"))
                tau_int_lo += tl if math.isfinite(tl) else t
                tau_int_hi += th if math.isfinite(th) else t
                if t > best_tau:
                    best_tau = t
                    dom = {
                        "best_z": z,
                        "best_b_over_rvir": _f(row.get("b_over_rvir")),
                        "best_g_scatt": _f(row.get("g_scatt")),
                        "best_cool_fc": _f(row.get("cool_fc")),
                        "best_logM": _f(row.get("logM_best")),
                        "best_mass_source": (
                            str(row.get("mass_source")) if row.get("mass_source") is not None else None
                        ),
                        "best_impact_kpc": _f(row.get("impact_kpc")),
                    }

    dm_intervening = dm_hot + dm_cool
    dm_intervening_capped = dm_hot_cap + dm_cool_cap
    if n_fg == 0:
        dm_regime = "none"
    elif math.isfinite(min_b_over_rvir) and min_b_over_rvir < INTERIOR_B_OVER_RVIR:
        dm_regime = "GALAXY_INTERIOR"
    else:
        dm_regime = "CGM"

    # Without a real host redshift the cosmic/host decomposition is undefined.
    z_is_placeholder = _is_placeholder_z(z_frb)
    dm_cosmic = math.nan if z_is_placeholder else dm_cosmic_macquart(float(z_frb))
    dm_obs_f = _f(dm_obs)
    if math.isfinite(dm_obs_f) and not z_is_placeholder:
        base = dm_obs_f - _nz(dm_mw_ism) - dm_mw_halo - dm_cosmic
        dm_host = base - dm_intervening
        dm_host_capped = base - dm_intervening_capped
        dm_host_rest = dm_host * (1.0 + float(z_frb))
        dm_host_rest_capped = dm_host_capped * (1.0 + float(z_frb))
    else:
        dm_host = dm_host_capped = dm_host_rest = dm_host_rest_capped = math.nan

    # Reliability of the intervening DM/tau: which mass the dominant screen used.
    intervening_mass_source = dom.get("best_mass_source")
    if intervening_mass_source is None:
        intervening_mass_confidence = "none"
    elif intervening_mass_source in _MEASURED_MASS_SOURCES:
        intervening_mass_confidence = "measured"
    else:
        intervening_mass_confidence = "assumed"

    tau_obs_f = _f(tau_obs)
    if not math.isfinite(tau_obs_f) and tau_obs_quality not in (None, "PASS", "INJECTED"):
        # A fit exists but failed the quality gate -> its tau is withheld.
        verdict_scat = (
            f"scattering fit present but quality_flag={tau_obs_quality} "
            f"(chi2_red={tau_obs_chi2:.1f}); tau not locked in"
            if math.isfinite(tau_obs_chi2)
            else f"scattering fit present but quality_flag={tau_obs_quality}; tau not locked in"
        )
    else:
        verdict_scat = _scattering_verdict(tau_obs_f, tau_int, tau_int_hi, n_fg)
    if z_is_placeholder:
        verdict_dm = "z_frb is a placeholder (unknown host z); cosmic & host DM budget unavailable"
    else:
        # The capped intervening DM is the physically bounded one; base the
        # dominant-component verdict on it, and note when the raw core-
        # extrapolated value over-shoots the observed budget.
        verdict_dm = _dm_verdict(
            dm_obs_f, dm_mw_ism, dm_mw_halo, dm_cosmic, dm_intervening_capped, dm_host_capped
        )
        if dm_regime == "GALAXY_INTERIOR" and math.isfinite(dm_host) and dm_host < -0.05 * max(dm_obs_f, 1.0):
            verdict_dm += "; raw intervening DM is core-extrapolated and exceeds the observed budget"

    flags = {
        "z_frb": "PLACEHOLDER" if z_is_placeholder else "MEASURED",
        "dm_obs": "MEASURED" if math.isfinite(dm_obs_f) else "NOT_AVAILABLE",
        "dm_mw_ism": "MODEL" if math.isfinite(_f(dm_mw_ism)) else "NOT_AVAILABLE",
        "dm_mw_halo": "PRIOR",
        "dm_cosmic": "NOT_AVAILABLE (placeholder z)" if z_is_placeholder else "PREDICTED_MEAN",
        "dm_intervening": f"PREDICTED ({intervening_mass_confidence} mass)",
        "dm_intervening_capped": f"PREDICTED ({dm_regime})",
        "dm_host": "RESIDUAL" if math.isfinite(dm_host) else "NOT_AVAILABLE",
        "tau_obs": (
            f"MEASURED ({tau_obs_quality})" if math.isfinite(tau_obs_f)
            else (f"WITHHELD ({tau_obs_quality})" if tau_obs_quality not in (None, "PASS", "INJECTED")
                  else "NOT_AVAILABLE")
        ),
        "tau_mw": "MODEL" if math.isfinite(_f(tau_mw_ms)) else "NOT_AVAILABLE",
        "tau_intervening": "PREDICTED",
    }

    record = {
        "name": name,
        "z_frb": float(z_frb),
        "l_deg": l_deg,
        "b_deg": b_deg,
        "dm_obs": dm_obs_f,
        "dm_mw_ism": _f(dm_mw_ism),
        "dm_mw_ism_ymw16": _f(dm_mw_ism_ymw16),
        "dm_mw_halo": float(dm_mw_halo),
        "dm_cosmic": dm_cosmic,
        "dm_intervening": dm_intervening,
        "dm_intervening_hot": dm_hot,
        "dm_intervening_cool": dm_cool,
        "dm_intervening_capped": dm_intervening_capped,
        "dm_intervening_hot_capped": dm_hot_cap,
        "dm_intervening_cool_capped": dm_cool_cap,
        "dm_intervening_regime": dm_regime,
        "intervening_mass_source": intervening_mass_source,
        "intervening_mass_confidence": intervening_mass_confidence,
        "z_is_placeholder": z_is_placeholder,
        "min_b_over_rvir": min_b_over_rvir if math.isfinite(min_b_over_rvir) else math.nan,
        "dm_host": dm_host,
        "dm_host_capped": dm_host_capped,
        "dm_host_rest": dm_host_rest,
        "dm_host_rest_capped": dm_host_rest_capped,
        "tau_obs_ms": tau_obs_f if math.isfinite(tau_obs_f) else math.nan,
        "tau_obs_err_minus": tau_obs_err_minus,
        "tau_obs_err_plus": tau_obs_err_plus,
        "tau_obs_quality": tau_obs_quality,
        "tau_obs_chi2_reduced": tau_obs_chi2,
        "tau_mw_ms": _f(tau_mw_ms),
        "tau_intervening_ms": tau_int,
        "tau_intervening_lo": tau_int_lo,
        "tau_intervening_hi": tau_int_hi,
        "n_foreground": n_fg,
        "n_intersecting": n_isect,
        "verdict_scattering": verdict_scat,
        "verdict_dm": verdict_dm,
        "cgm_budget_flags": flags,
    }
    record.update(dom)
    return record


def build_all_budgets(
    targets=None,
    results_dir: str = "results",
    configs_dir: str | None = None,
    bursts_dir: str | None = None,
    enrich: bool = False,
    dm_mw_fn: Callable | None = None,
    dm_obs_map: Mapping[str, float] | None = None,
    tau_obs_map: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build the DM + scattering budget for every configured sightline."""
    if targets is None:
        targets = config.TARGETS
    rows = []
    for name, ra_str, dec_str, z_frb in targets:
        dm_obs = dm_obs_map.get(name) if dm_obs_map is not None else None
        tau_obs = tau_obs_map.get(name) if tau_obs_map is not None else None
        rows.append(
            build_sightline_budget(
                name, ra_str, dec_str, z_frb,
                results_dir=results_dir, configs_dir=configs_dir, bursts_dir=bursts_dir,
                enrich=enrich, dm_mw_fn=dm_mw_fn, dm_obs=dm_obs, tau_obs=tau_obs,
            )
        )
    return pd.DataFrame(rows)


def format_budget_table(df: pd.DataFrame) -> str:
    """Render the DM + scattering budget as a GitHub-flavored markdown table."""
    headers = [
        "Sightline", "z", "DM_obs", "DM_MW", "DM_cosmic", "DM_interv_raw",
        "DM_interv_cap", "regime", "interv_mass", "DM_host_cap", "tau_obs(ms)",
        "tau_interv(ms)", "scattering attribution",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    def fmt(x, spec):
        v = _f(x)
        return format(v, spec) if math.isfinite(v) else "-"

    for _, r in df.iterrows():
        cells = [
            str(r["name"]),
            fmt(r.get("z_frb"), ".3f"),
            fmt(r.get("dm_obs"), ".0f"),
            fmt(_nz(_f(r.get("dm_mw_ism"))) + _f(r.get("dm_mw_halo")), ".0f"),
            fmt(r.get("dm_cosmic"), ".0f"),
            fmt(r.get("dm_intervening"), ".1f"),
            fmt(r.get("dm_intervening_capped"), ".1f"),
            str(r.get("dm_intervening_regime", "")),
            str(r.get("intervening_mass_confidence", "")),
            fmt(r.get("dm_host_capped"), ".0f"),
            fmt(r.get("tau_obs_ms"), ".3g"),
            fmt(r.get("tau_intervening_ms"), ".2g"),
            str(r.get("verdict_scattering", "")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Styling consistent with the rest of the module.
DARK_BLUE = "#1B365D"
MW_COLOR = "#4A90E2"
HALO_COLOR = "#7FB3E8"
COSMIC_COLOR = "#9B59B6"
INTERV_COLOR = "#F5A623"
HOST_COLOR = "#D0021B"
TEXT_DARK = "#333333"
GRID_COLOR = "#E5E5E5"
BG_LIGHT = "#FAFBFC"


def make_budget_figure(df: pd.DataFrame):
    """Two-panel figure: stacked DM budget + predicted-vs-measured scattering."""
    d = df.copy()
    names = [str(n) for n in d["name"]]
    y = np.arange(len(d))[::-1]

    fig, (ax_dm, ax_tau) = plt.subplots(
        1, 2, figsize=(14, max(3.5, 0.55 * len(d) + 1.6)), dpi=150,
        facecolor=BG_LIGHT, gridspec_kw={"width_ratios": [1.7, 1.0]},
    )
    for ax in (ax_dm, ax_tau):
        ax.set_facecolor(BG_LIGHT)

    def col(name):
        return pd.to_numeric(d.get(name, pd.Series(np.nan, index=d.index)), errors="coerce").fillna(0.0).to_numpy(float)

    mw = col("dm_mw_ism")
    halo = col("dm_mw_halo")
    cosmic = col("dm_cosmic")
    interv = col("dm_intervening_capped")
    interv_raw = col("dm_intervening")
    host = np.maximum(col("dm_host_capped"), 0.0)
    regime = [str(v) for v in d.get("dm_intervening_regime", pd.Series([""] * len(d)))]

    left = np.zeros(len(d))
    for vals, color, label in (
        (mw, MW_COLOR, "MW ISM (NE2001)"),
        (halo, HALO_COLOR, "MW halo"),
        (cosmic, COSMIC_COLOR, "cosmic/IGM <DM>"),
        (interv, INTERV_COLOR, "intervening CGM (b>=0.1Rvir cap)"),
        (host, HOST_COLOR, "host (residual)"),
    ):
        ax_dm.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.6, label=label, zorder=3)
        left = left + vals

    dm_obs = col("dm_obs")
    for yi, xo in zip(y, dm_obs):
        if xo > 0:
            ax_dm.plot([xo, xo], [yi - 0.4, yi + 0.4], color=TEXT_DARK, lw=2.0, zorder=5)

    # Flag core-extrapolated sightlines and show the raw (uncapped) intervening DM.
    for yi, reg, raw, cap in zip(y, regime, interv_raw, interv):
        if reg == "GALAXY_INTERIOR" and raw > cap:
            ax_dm.annotate(
                f"raw interv DM={raw:.0f} (core-extrap.)",
                xy=(0, yi), xytext=(4, -2), textcoords="offset points",
                va="top", ha="left", fontsize=7, color=HOST_COLOR, zorder=6,
            )

    # Mark sightlines whose host redshift is a placeholder (no cosmic/host budget).
    placeholder = [_truthy(v) for v in d.get("z_is_placeholder", pd.Series([False] * len(d)))]
    for yi, ph, xo in zip(y, placeholder, dm_obs):
        if ph:
            ax_dm.annotate(
                "z placeholder — no cosmic/host budget",
                xy=(max(xo, 1.0), yi), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=7, color=COSMIC_COLOR, zorder=6,
            )
    ax_dm.set_yticks(y)
    ax_dm.set_yticklabels(names, fontsize=10)
    ax_dm.set_xlabel("DM (pc cm$^{-3}$)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax_dm.set_title("DM budget per sightline  (black bar = observed DM)",
                    fontsize=12, fontweight="bold", color=DARK_BLUE, pad=10)
    ax_dm.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white", edgecolor=GRID_COLOR)
    ax_dm.grid(True, axis="x", linestyle=":", color=GRID_COLOR, alpha=0.8, zorder=0)

    tau_obs = col("tau_obs_ms")
    tau_int = col("tau_intervening_ms")
    tau_lo = col("tau_intervening_lo")
    tau_hi = col("tau_intervening_hi")
    for yi, to, ti, tl, th in zip(y, tau_obs, tau_int, tau_lo, tau_hi):
        if ti > 0:
            ax_tau.plot([max(tl, 1e-7), max(th, 1e-7)], [yi, yi], color=INTERV_COLOR, lw=2, zorder=2)
            ax_tau.scatter([ti], [yi], color=INTERV_COLOR, s=45, zorder=3,
                           label="predicted intervening" if yi == y[0] else None)
        if to > 0:
            tlo = _f(d["tau_obs_err_minus"].iloc[list(y).index(yi)]) if "tau_obs_err_minus" in d.columns else math.nan
            thi = _f(d["tau_obs_err_plus"].iloc[list(y).index(yi)]) if "tau_obs_err_plus" in d.columns else math.nan
            if math.isfinite(tlo) and math.isfinite(thi) and (tlo > 0 or thi > 0):
                ax_tau.errorbar([to], [yi], xerr=[[max(tlo, 0)], [max(thi, 0)]], fmt="none",
                                ecolor=TEXT_DARK, elinewidth=1.2, capsize=3, zorder=4)
            ax_tau.scatter([to], [yi], marker="D", color=TEXT_DARK, s=55, zorder=5,
                           label="measured burst (PASS)" if yi == y[0] else None)

    # Mark sightlines with a fit present but withheld by the quality gate.
    quality = [str(v) for v in d.get("tau_obs_quality", pd.Series([""] * len(d)))]
    withheld_labeled = False
    for yi, q, ti in zip(y, quality, tau_int):
        if q in ("FAIL", "MARGINAL", "UNKNOWN"):
            x_at = max(ti, 1e-6)
            ax_tau.scatter([x_at], [yi], marker="x", color=HOST_COLOR, s=50, zorder=4,
                           label="fit withheld (failed gate)" if not withheld_labeled else None)
            withheld_labeled = True

    ax_tau.set_yticks(y)
    ax_tau.set_yticklabels([])
    positive = np.concatenate([tau_obs[tau_obs > 0], tau_int[tau_int > 0]]) if len(d) else np.array([])
    if positive.size:
        ax_tau.set_xscale("log")
    ax_tau.set_xlabel(r"$\tau_{\rm scat}$ at 1 GHz (ms)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax_tau.set_title("Scattering: predicted intervening vs measured",
                     fontsize=12, fontweight="bold", color=DARK_BLUE, pad=10)
    handles, labels = ax_tau.get_legend_handles_labels()
    if handles:
        ax_tau.legend(loc="best", fontsize=8, frameon=True, facecolor="white", edgecolor=GRID_COLOR)
    ax_tau.grid(True, axis="x", linestyle=":", color=GRID_COLOR, alpha=0.8)

    fig.suptitle("FRB sightline DM & scattering budgets (priors + measurements)",
                 fontsize=13, fontweight="bold", color=DARK_BLUE)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results")
    configs_dir = os.path.join(base_dir, "scattering", "configs", "bursts", "chime")
    bursts_dir = os.path.join(results_dir, "bursts")

    df = build_all_budgets(
        results_dir=results_dir, configs_dir=configs_dir, bursts_dir=bursts_dir, enrich=False
    )

    # Emit TNS designations (not internal nicknames) in the CSV, table, and figure.
    from scattering.scat_analysis.burst_metadata import load_tns_name
    df["name"] = [load_tns_name(n) for n in df["name"]]

    csv_path = os.path.join(results_dir, "sightline_dm_scattering_budget.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    md_path = os.path.join(results_dir, "sightline_dm_scattering_budget.md")
    with open(md_path, "w") as fh:
        fh.write("# FRB sightline DM & scattering budgets\n\n")
        fh.write(format_budget_table(df))
        fh.write("\n")
    print(f"Wrote {md_path}")

    fig = make_budget_figure(df)
    png_path = os.path.join(results_dir, "sightline_dm_scattering_budget.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
