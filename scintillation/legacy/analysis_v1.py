# ==============================================================================
# File: scint_analysis/scint_analysis/analysis.py
# ==============================================================================
import numpy as np
import logging
from .core import ACF
from lmfit import Model, Parameters
from lmfit.models import ConstantModel
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.odr import RealData, ODR, Model as ModelODR
from typing import Dict, Callable, Optional, Tuple

log = logging.getLogger(__name__)

# -------------------------
# --- Model Definitions ---
# -------------------------

def lorentzian_component(x, gamma, m):
    """A single Lorentzian component without a baseline constant."""
    return (m**2) / (1 + (x / gamma)**2)

def gaussian_component(x, sigma, m):
    """A single Gaussian component without a baseline constant."""
    return (m**2) * np.exp(-0.5 * (x / sigma)**2)

def lorentzian_generalised(x: np.ndarray | float,
                           gamma: float,
                           alpha: float,
                           m: float) -> np.ndarray:
    """Generalised (power‑law) Lorentzian.

    C(x) = m1² / [1 + |x/γ₁|^{α+2}]
    *alpha = 0* reproduces the standard Lorentzian; *alpha = 5/3* is the
    Kolmogorov diffractive prediction.
    """
    z = np.abs(x / gamma)
    return (m ** 2.0) / (1.0 + z ** (alpha + 2.0))

def power_law_model(x: np.ndarray | float,
                    p_c: float,
                    p_n: float) -> np.ndarray:
    """Pure power‑law tail *without* a core component."""
    eps = 1.0e-12  # guards against |x|**n when n < 0 and x == 0
    return p_c * (np.abs(x) + eps) ** p_n

# -----------------------------------------------------------------------------
# Fixed‑width self‑noise Gaussian
# -----------------------------------------------------------------------------

def gauss_fixed_width(x, sigma_self, m_self):
    """Pure Gaussian that models the pulse-width self‑noise component."""
    return m_self**2 * np.exp(-0.5 * (x / sigma_self) ** 2)

def _self_noise_model(sigma_self_mhz: float):
    sn = Model(gauss_fixed_width, prefix="sn_")
    p  = sn.make_params(
        sigma_self=sigma_self_mhz,  
        vary=True,
        min=sigma_self_mhz*0.25,   
        max=sigma_self_mhz*1.75,   # ±25 %
        m_self=0.3,
    )
    return sn, p

# -----------------------------------------------------------------------------
# Baseline Model Registry
# -----------------------------------------------------------------------------

def _baseline_registry(cfg_init: dict | None = None):
    """Return a list describing **all** baseline scattering models."""
    if cfg_init is None:
        cfg_init = {}

    def merge(seed: dict, tag: str):
        merged = seed.copy()
        merged.update(cfg_init.get(tag, {}))
        return merged

    return [
        ('lor', lorentzian_component, 'l_',
         merge(dict(l_gamma=0.05, l_m=0.8), tag='lor'),
         lambda p: (
             p['l_gamma'].set(min=1e-6),
             p['l_m'].set(min=0))),

        ('gauss', gaussian_component, 'g_',
         merge(dict(g_sigma=0.05, g_m=0.8), tag='gauss'),
         lambda p: (
             p['g_sigma'].set(min=1e-6),
             p['g_m'].set(min=0))),

        ('lor_gen', lorentzian_generalised, 'lg_',
         merge(dict(lg_gamma=0.05, lg_alpha=5/3, lg_m=0.8), tag='lor_gen'),
         lambda p: (
             p['lg_gamma'].set(min=1e-6),
             p['lg_alpha'].set(min=0.1, max=4.0),
             p['lg_m'].set(min=0))),

        ('power', power_law_model, 'p_',
         merge(dict(p_c=0.01, p_n=-2.0), tag='power'),
         lambda p: (
             p['p_c'].set(min=1e-6))),
    ]

# ----------------------------------------------
# --- Core Calculation and Fitting Functions ---
# ----------------------------------------------

def calculate_acf(spectrum_1d, channel_width_mhz, off_burst_spectrum_mean=None, max_lag_bins=None):
    """
    Calculates the ACF and its diagonal errors, including statistical and
    finite scintle contributions.
    """
    log.debug(f"Calculating ACF with robust errors for spectrum of length {len(spectrum_1d)}.")
    n_unmasked = spectrum_1d.count()
    if n_unmasked < 20:
        log.warning(f"Not enough data ({n_unmasked} points) to calculate a reliable ACF. Skipping.")
        return None
    if max_lag_bins is None:
        max_lag_bins = n_unmasked // 4
    if max_lag_bins < 2:
        log.warning("max_lag_bins is too small. Skipping ACF calculation.")
        return None
    mean_on = np.ma.mean(spectrum_1d)
    denom = (mean_on - off_burst_spectrum_mean)**2 if off_burst_spectrum_mean is not None else mean_on**2
    if denom == 0: denom = 1.0
    x = spectrum_1d.filled(np.nan) - mean_on
    lags = np.arange(1, max_lag_bins)
    acf_vals = np.zeros(len(lags))
    stat_errs = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        prod = x[:-lag] * x[lag:]
        valid_products = prod[~np.isnan(prod)]
        num_valid = len(valid_products)
        if num_valid > 1:
            acf_vals[i] = np.mean(valid_products) / denom
            var_of_products = np.var(valid_products, ddof=1)
            std_err_of_mean = np.sqrt(var_of_products / num_valid)
            stat_errs[i] = std_err_of_mean / denom
        else:
            acf_vals[i] = np.nan
            stat_errs[i] = np.nan
    positive_lags_mhz = lags * channel_width_mhz
    clean_mask = ~np.isnan(acf_vals)
    if not np.any(clean_mask): return None
    clean_acf = acf_vals[clean_mask]
    clean_lags = positive_lags_mhz[clean_mask]
    half_max = 0.5 * np.max(clean_acf)
    try:
        hwhm_mhz = np.interp(half_max, clean_acf[::-1], clean_lags[::-1])
        delta_nu_dc = hwhm_mhz
    except Exception:
        delta_nu_dc = channel_width_mhz * 10
    total_bandwidth = n_unmasked * channel_width_mhz
    n_scintles = max(1.0, total_bandwidth / delta_nu_dc)
    finite_scintle_frac_err = 1.0 / np.sqrt(n_scintles)
    finite_scintle_errs = np.abs(acf_vals) * finite_scintle_frac_err
    full_acf = np.concatenate((acf_vals[clean_mask][::-1], [1.0], acf_vals[clean_mask]))
    full_lags = np.concatenate((-positive_lags_mhz[clean_mask][::-1], [0.0], positive_lags_mhz[clean_mask]))
    full_stat_err = np.concatenate((stat_errs[clean_mask][::-1], [1e-9], stat_errs[clean_mask]))
    full_finite_err = np.concatenate((finite_scintle_errs[clean_mask][::-1], [0.0], finite_scintle_errs[clean_mask]))
    total_diag_err = np.sqrt(full_stat_err**2 + full_finite_err**2)
    return ACF(full_acf, full_lags, acf_err=total_diag_err)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _estimate_sigma_self(ds, burst_lims):
    """Return σ_self (MHz) – the Gaussian width of the self‑noise ACF component."""
    t_series = ds.power[:, burst_lims[0]:burst_lims[1]].sum(axis=0).filled(0.0)
    if t_series.sum() == 0:
        return None
    cdf = np.cumsum(t_series)
    cdf /= cdf[-1]
    t_bins = ds.times[: len(t_series)]
    t16, t84 = np.interp([0.16, 0.84], cdf, t_bins)
    sigma_t = 0.5 * (t84 - t16)
    sigma_self_hz = 1.0 / (2.0 * np.pi * sigma_t)
    return sigma_self_hz / 1e6

_noise_acf_cache: Dict[Tuple[int, int, float, int], np.ndarray] = {}
def _mean_noise_acf(
        noise_desc,
        n_rep,
        spec_len,
        channel_width_mhz,
        *,
        mask_hash,
        acf_fn=calculate_acf):
    """Monte‑Carlo average spectral ACF of pure noise rows."""
    key = (id(noise_desc), spec_len, channel_width_mhz, mask_hash)
    if key in _noise_acf_cache:
        return _noise_acf_cache[key]
    acfs = []
    for _ in range(n_rep):
        noise_row = noise_desc.sample()[0]
        acf_obj = acf_fn(
            np.ma.masked_invalid(noise_row),
            channel_width_mhz,
            off_burst_spectrum_mean=0.0,
            max_lag_bins=(spec_len + 1) // 2,
        )
        if acf_obj is not None:
            acfs.append(acf_obj.acf)
    if not acfs:
        return None
    mean_acf = np.mean(acfs, axis=0)
    _noise_acf_cache[key] = mean_acf
    return mean_acf

def calculate_acf_noerrs(spectrum_1d, channel_width_mhz, off_burst_spectrum_mean=None, max_lag_bins=None):
    """Calculates the one-sided autocorrelation function of a spectrum."""
    log.debug(f"Calculating ACF for a spectrum of length {len(spectrum_1d)}.")
    valid_spec = spectrum_1d.compressed()
    if valid_spec.size < 10: return None
    mean_on = np.mean(valid_spec)
    denom = (mean_on - off_burst_spectrum_mean)**2 if off_burst_spectrum_mean is not None else mean_on**2
    if denom == 0: denom = 1.0
    x = spectrum_1d.filled(np.nan) - mean_on
    n_chan = len(x)
    if max_lag_bins is None: max_lag_bins = n_chan
    lags = np.arange(1, max_lag_bins)
    acf_vals = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        v1, v2 = x[:-lag], x[lag:]
        prod = v1 * v2
        num_valid = np.sum(~np.isnan(prod))
        if num_valid > 1:
            acf_vals[i] = np.nansum(prod) / (num_valid * denom)
    pos_lags_mhz = lags * channel_width_mhz
    full_acf = np.concatenate((acf_vals[::-1], acf_vals))
    full_lags = np.concatenate((-pos_lags_mhz[::-1], pos_lags_mhz))
    return ACF(full_acf, full_lags)

# ... (rest of the code remains unchanged, as it is actively referenced and used) ...

