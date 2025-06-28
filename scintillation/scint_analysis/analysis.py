# ==============================================================================
# File: scint_analysis/scint_analysis/analysis.py
# ==============================================================================
import numpy as np
import logging
from .core import ACF
from lmfit import Model, Parameters
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

def lorentzian_model_1_comp(x, gamma1, m1, c1):
    return (m1**2 / (1 + (x / gamma1)**2)) + c1

def lorentzian_model_2_comp(x, gamma1, m1, gamma2, m2, c2):
    lor1 = m1**2 / (1 + (x / gamma1)**2)
    lor2 = m2**2 / (1 + (x / gamma2)**2)
    return lor1 + lor2 + c2

def lorentzian_model_3_comp(x, gamma1, m1, gamma2, m2, gamma3, m3, c3):
    lor1 = m1**2 / (1 + (x / gamma1)**2)
    lor2 = m2**2 / (1 + (x / gamma2)**2)
    lor3 = m3**2 / (1 + (x / gamma3)**2)
    return lor1 + lor2 + lor3 + c3

def gaussian_model_1_comp(x, sigma1, m1, c1):
    return (m1**2 * np.exp(-0.5 * (x / sigma1)**2)) + c1

def gaussian_model_2_comp(x, sigma1, m1, sigma2, m2, c2):
    gauss1 = m1**2 * np.exp(-0.5 * (x / sigma1)**2)
    gauss2 = m2**2 * np.exp(-0.5 * (x / sigma2)**2)
    return gauss1 + gauss2 + c2

def gaussian_model_3_comp(x, sigma1, m1, sigma2, m2, sigma3, m3, c3):
    gauss1 = m1**2 * np.exp(-0.5 * (x / sigma1)**2)
    gauss2 = m2**2 * np.exp(-0.5 * (x / sigma2)**2)
    gauss3 = m3**2 * np.exp(-0.5 * (x / sigma3)**2)
    return gauss1 + gauss2 + gauss3 + c3

def gauss_plus_lor_model(x, sigma1, m1, gamma2, m2, c):
    gauss = m1**2 * np.exp(-0.5 * (x / sigma1)**2)
    lor = m2**2 / (1 + (x / gamma2)**2)
    return gauss + lor + c

def two_screen_unresolved_model(x, gamma1, m1, gamma2, m2, c):
    """ A physically-motivated model for two unresolved screens. """
    lor1 = (m1**2) / (1 + (x / gamma1)**2)
    lor2 = (m2**2) / (1 + (x / gamma2)**2)
    return lor1 + lor2 + (lor1 * lor2) + c

def power_law_model(x, c, n):
    """A simple power-law model: y = c * x^n."""
    return c * (x**n)

# -----------------------------------------------------------------------------
# Fixed‑width self‑noise Gaussian
# -----------------------------------------------------------------------------

def gauss_fixed_width(x, sigma_self, m_self, c):
    """Pure Gaussian that models the pulse-width self-noise component."""
    return m_self**2 * np.exp(-0.5 * (x / sigma_self) ** 2) + c

def _self_noise_model(sigma_self_mhz: float):
    sn = Model(gauss_fixed_width, prefix="sn_")
    p  = sn.make_params(
        sigma_self=sigma_self_mhz,  
        vary=True,
        min=sigma_self_mhz*0.25,   
        max=sigma_self_mhz*1.75,   # ±25 %
        m_self=0.3,
        c=0.0,
    )
    return sn, p

    
def _baseline_registry(cfg_init=None):
    """
    Return list describing all baseline scattering models.
    cfg_init : dict  (values from YAML → analysis.fitting.init_guess)
    """
    if cfg_init is None:
        cfg_init = {}

    def merge(seed: dict, tag: str):
        """override hard-coded seed with YAML values"""
        merged = seed.copy()
        merged.update(cfg_init.get(tag, {}))
        return merged

    return [
        # ---------- 1-component Lorentzian ---------------------------------
        ('1c_lor', lorentzian_model_1_comp, 'l1_',
         merge(dict(l1_gamma1=0.05, l1_m1=0.8, l1_c1=0), tag='1c_lor'),
         lambda p: (
             p['l1_gamma1'].set(min=1e-6),
             p['l1_m1'].set(min=0))),

        # ---------- 2-component Lorentzian ---------------------------------
        ('2c_lor', lorentzian_model_2_comp, 'l2_',
         merge(dict(l2_gamma1=0.01, l2_gamma2=0.05,
                    l2_m1=0.5,  l2_m2=0.5, l2_c2=0), tag='2c_lor'),
         lambda p: (
             p['l2_gamma1'].set(min=1e-6),
             p['l2_gamma2'].set(min=1e-6),
             p['l2_m1'].set(min=0),
             p['l2_m2'].set(min=0))),

        # ---------- 3-component Lorentzian ---------------------------------
        ('3c_lor', lorentzian_model_3_comp, 'l3_',
         merge(dict(l3_gamma1=0.01, l3_gamma2=0.10, l3_gamma3=0.30,
                    l3_m1=0.3,   l3_m2=0.3,   l3_m3=0.3, l3_c3=0), tag='3c_lor'),
         lambda p: (
             p['l3_gamma1'].set(min=1e-6),
             p['l3_gamma2'].set(min=1e-6),
             p['l3_gamma3'].set(min=1e-6),
             p['l3_m1'].set(min=0),
             p['l3_m2'].set(min=0),
             p['l3_m3'].set(min=0))),
    ]

def _baseline_registry_v0():
    """Return list of tuples describing all baseline scattering models."""
    return [
        # --- 1‑component models ------------------------------------------------
        ('1c_lor',  lorentzian_model_1_comp,  'l1_',
         dict(l1_gamma1=0.05, l1_m1=0.8, l1_c1=0),
         lambda p: (p['l1_gamma1'].set(min=1e-6), p['l1_m1'].set(min=0))),

        #('1c_gauss', gaussian_model_1_comp,  'g1_',
        # dict(g1_sigma1=0.05, g1_m1=0.8, g1_c1=0),
        # lambda p: (p['g1_sigma1'].set(min=1e-6), p['g1_m1'].set(min=0))),

        # --- 2‑component models ----------------------------------------------
        #('2c_lor',  lorentzian_model_2_comp, 'l2_',
        # dict(l2_gamma1=0.01, l2_m1=0.5, l2_c2=0),
        # lambda p: (
        #     p['l2_gamma1'].set(min=1e-6), p['l2_m1'].set(min=0),
        #     p['l2_m2'].set(value=0.5, min=0),
        #     p.add('l2_gamma_factor', value=5, min=1.01),
        #     p['l2_gamma2'].set(expr='l2_gamma1 * l2_gamma_factor'))),
        
        ('2c_lor', lorentzian_model_2_comp, 'l2_',
        dict(l2_gamma1=0.01, l2_m1=0.5, l2_c2=0),
        lambda p: (
             p['l2_gamma1'].set(min=1e-6),
             p['l2_m1'].set(min=0),
             # NEW – free gamma2, just require ≥ gamma1
             p.add('l2_gamma2', value=0.05, min=1e-6),
             p['l2_m2'].set(value=0.5, min=0))),
        
        #('2c_gauss', gaussian_model_2_comp, 'g2_',
        # dict(g2_sigma1=0.01, g2_m1=0.5, g2_c2=0),
        # lambda p: (
        #     p['g2_sigma1'].set(min=1e-6), 
        #     p['g2_m1'].set(min=0),
        #     p.add('g2_sigma2', value=5, min=1e-6),
        #     p['g2_m2'].set(value=0.5, min=0))),
        
        #('2c_gauss', gaussian_model_2_comp, 'g2_',
        # dict(g2_sigma1=0.01, g2_m1=0.5, g2_c2=0),
        # lambda p: (
        #     p['g2_sigma1'].set(min=1e-6), p['g2_m1'].set(min=0),
        #     p['g2_m2'].set(value=0.5, min=0),
        #     p.add('g2_sigma_factor', value=5, min=1.01),
        #     p['g2_sigma2'].set(expr='g2_sigma1 * g2_sigma_factor'))),

        #('2c_mixed', gauss_plus_lor_model, 'gl_',
        # dict(gl_sigma1=0.01, gl_m1=0.5, gl_gamma2=0.1, gl_m2=0.5, gl_c=0),
        # lambda p: (
        #     p['gl_sigma1'].set(min=1e-6), p['gl_m1'].set(min=0),
        #     p['gl_gamma2'].set(min=1e-6), p['gl_m2'].set(min=0))),

        #('2c_unresolved', two_screen_unresolved_model, 'tsu_',
        # dict(tsu_gamma1=0.01, tsu_m1=0.5, tsu_c=0),
        # lambda p: (
        #     p['tsu_gamma1'].set(min=1e-6), 
        #     p['tsu_m1'].set(value=0.5, min=0),
        #     p.add('tsu_gamma2', value=5, min=1e-6),
        #     p['tsu_m2'].set(value=0.5, min=0))),
        
        #('2c_unresolved', two_screen_unresolved_model, 'tsu_',
        # dict(tsu_gamma1=0.01, tsu_m1=0.5, tsu_c=0),
        # lambda p: (
        #     p['tsu_gamma1'].set(min=1e-6), p['tsu_m1'].set(min=0),
        #     p['tsu_m2'].set(value=0.5, min=0),
        #     p.add('tsu_gamma_factor', value=5, min=1.01),
        #     p['tsu_gamma2'].set(expr='tsu_gamma1 * tsu_gamma_factor'))),

        # --- 3‑component models ----------------------------------------------
        
        
        ('3c_lor',  lorentzian_model_3_comp, 'l3_',
         dict(l3_gamma1=0.01, l3_m1=0.3, l3_c3=0),
         lambda p: (
             p['l3_gamma1'].set(min=1e-6), 
             p['l3_m1'].set(min=0),
             p['l3_m2'].set(value=0.3, min=0),
             p.add('l3_gamma2', value=1, min=1e-6),
             p.add('l3_gamma3', value=3, min=1e-6),
             p['l3_m3'].set(value=0.3, min=0))),
        
        #('3c_gauss', gaussian_model_3_comp, 'g3_',
        # dict(g3_sigma1=0.01, g3_m1=0.3, g3_c3=0),
        # lambda p: (
        #     p['g3_sigma1'].set(min=1e-6), p['g3_m1'].set(min=0),
        #     p['g3_m2'].set(value=0.3, min=0),
        #     p.add('g3_sigma2', value=1, min=1e-6),
        #     p.add('g3_sigma3', value=3, min=1e-6),
        #     p['g3_m3'].set(value=0.3, min=0))),
        
        #('3c_lor',  lorentzian_model_3_comp, 'l3_',
        # dict(l3_gamma1=0.01, l3_m1=0.3, l3_c3=0),
        # lambda p: (
        #     p['l3_gamma1'].set(min=1e-6), p['l3_m1'].set(min=0),
        #     p['l3_m2'].set(value=0.3, min=0), p['l3_m3'].set(value=0.3, min=0),
        #     p.add('l3_gamma_factor2', value=5, min=1.01),
        #     p.add('l3_gamma_factor3', value=5, min=1.01),
        #     p['l3_gamma2'].set(expr='l3_gamma1 * l3_gamma_factor2'),
        #     p['l3_gamma3'].set(expr='l3_gamma2 * l3_gamma_factor3'))),

        #('3c_gauss', gaussian_model_3_comp, 'g3_',
        # dict(g3_sigma1=0.01, g3_m1=0.3, g3_c3=0),
        # lambda p: (
        #     p['g3_sigma1'].set(min=1e-6), p['g3_m1'].set(min=0),
        #     p['g3_m2'].set(value=0.3, min=0), p['g3_m3'].set(value=0.3, min=0),
        #     p.add('g3_sigma_factor2', value=5, min=1.01),
        #     p.add('g3_sigma_factor3', value=5, min=1.01),
        #     p['g3_sigma2'].set(expr='g3_sigma1 * g3_sigma_factor2'),
        #     p['g3_sigma3'].set(expr='g3_sigma2 * g3_sigma_factor3'))),
    ]


# ----------------------------------------------
# --- Core Calculation and Fitting Functions ---
# ----------------------------------------------

def calculate_acf(spectrum_1d, channel_width_mhz, off_burst_spectrum_mean=None, max_lag_bins=None):
    """
    Calculates the ACF and its diagonal errors, including statistical and
    finite scintle contributions.

    This method calculates the standard error of the mean for the products at each
    lag and combines it in quadrature with an estimate of the finite scintle noise.

    Parameters
    ----------
    spectrum_1d : np.ma.MaskedArray
        The 1D spectrum to autocorrelate.
    channel_width_mhz : float
        The channel width in MHz.
    off_burst_spectrum_mean : float, optional
        The mean of the off-burst spectrum for normalization.
    max_lag_bins : int, optional
        The maximum number of bins for the ACF.

    Returns
    -------
    ACF object or None
        An ACF object containing the ACF, lags, and combined errors, or None if
        the calculation fails.
    """
    log.debug(f"Calculating ACF with robust errors for spectrum of length {len(spectrum_1d)}.")
    
    n_unmasked = spectrum_1d.count()
    if n_unmasked < 20:
        log.warning(f"Not enough data ({n_unmasked} points) to calculate a reliable ACF. Skipping.")
        return None

    if max_lag_bins is None:
        max_lag_bins = n_unmasked // 4 # Default to 1/4 of the unmasked channels
    if max_lag_bins < 2:
        log.warning("max_lag_bins is too small. Skipping ACF calculation.")
        return None

    # --- 1. Basic ACF Calculation ---
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
            # Calculate the ACF value (mean of the products)
            acf_vals[i] = np.mean(valid_products) / denom
            
            # Calculate the statistical error (standard error of the mean of the products)
            # Use ddof=1 for sample standard deviation
            var_of_products = np.var(valid_products, ddof=1)
            std_err_of_mean = np.sqrt(var_of_products / num_valid)
            stat_errs[i] = std_err_of_mean / denom
        else:
            acf_vals[i] = np.nan
            stat_errs[i] = np.nan

    # --- 2. Finite Scintle Error Calculation ---
    # Use the calculated ACF to estimate the decorrelation bandwidth (Δν_DC)
    positive_lags_mhz = lags * channel_width_mhz
    
    # Clean out any NaNs from failed lag calculations before finding HWHM
    clean_mask = ~np.isnan(acf_vals)
    if not np.any(clean_mask): return None # Return if all lags failed
    
    clean_acf = acf_vals[clean_mask]
    clean_lags = positive_lags_mhz[clean_mask]
    
    half_max = 0.5 * np.max(clean_acf)
    try:
        # Interpolate to find the HWHM accurately
        # Note: interp needs monotonically increasing x-values (clean_acf is decreasing)
        hwhm_mhz = np.interp(half_max, clean_acf[::-1], clean_lags[::-1])
        delta_nu_dc = hwhm_mhz
    except Exception:
        delta_nu_dc = channel_width_mhz * 10 # Fallback if interpolation fails

    # Number of scintles = Total Bandwidth / Decorrelation Bandwidth
    total_bandwidth = n_unmasked * channel_width_mhz
    n_scintles = max(1.0, total_bandwidth / delta_nu_dc)
    
    # Fractional error due to finite scintles
    finite_scintle_frac_err = 1.0 / np.sqrt(n_scintles)
    
    # Convert fractional error to error in ACF units for each lag
    finite_scintle_errs = np.abs(acf_vals) * finite_scintle_frac_err

    # --- 3. Symmetrize and Combine ---
    # Create the full, two-sided arrays for ACF, lags, and errors
    full_acf = np.concatenate((acf_vals[clean_mask][::-1], [1.0], acf_vals[clean_mask]))
    full_lags = np.concatenate((-positive_lags_mhz[clean_mask][::-1], [0.0], positive_lags_mhz[clean_mask]))
    
    full_stat_err = np.concatenate((stat_errs[clean_mask][::-1], [1e-9], stat_errs[clean_mask]))
    full_finite_err = np.concatenate((finite_scintle_errs[clean_mask][::-1], [0.0], finite_scintle_errs[clean_mask]))
    
    # Combine the two error sources in quadrature to get the total diagonal error
    total_diag_err = np.sqrt(full_stat_err**2 + full_finite_err**2)

    return ACF(full_acf, full_lags, acf_err=total_diag_err)

# -----------------------------------------------------------------------------
# Helper utilities (place these anywhere in analysis.py above the main function)
# -----------------------------------------------------------------------------

def _estimate_sigma_self(ds, burst_lims):
    """Return σ_self (MHz) – the Gaussian width of the self‑noise ACF component.

    Uses the 16–84 % cumulative‑energy interval of the frequency‑summed burst
    profile so that it is robust to multi‑peaked or asymmetric bursts.
    Implements Eq. (7) of Pradeep et al. (2025).
    """
    # Collapse dynamic spectrum to one time series (mask ignored → filled with 0)
    t_series = ds.power[:, burst_lims[0]:burst_lims[1]].sum(axis=0).filled(0.0)
    if t_series.sum() == 0:
        return None  # no useful signal

    cdf = np.cumsum(t_series)
    cdf /= cdf[-1]
    t_bins = ds.times[: len(t_series)]
    t16, t84 = np.interp([0.16, 0.84], cdf, t_bins)
    sigma_t = 0.5 * (t84 - t16)  # ≈1 σ for a Gaussian pulse
    sigma_self_hz = 1.0 / (2.0 * np.pi * sigma_t)
    return sigma_self_hz / 1e6  # MHz

_noise_acf_cache: Dict[Tuple[int, int, float, int], np.ndarray] = {}
def _mean_noise_acf(
        noise_desc,
        n_rep,
        spec_len,
        channel_width_mhz,
        *,
        mask_hash,
        acf_fn=calculate_acf):
    """Monte‑Carlo average spectral ACF of pure noise rows.

    Parameters
    ----------
    noise_desc : NoiseDescriptor
        Object capable of `.sample()` → (time, freq) array(s); statistics match data.
    n_rep : int
        Number of synthetic rows to average. ≥100 is recommended for smoothness.
    spec_len : int
        Number of frequency bins (≥ 1 + 2·max_lag_bins) for which the ACF will be
        evaluated so that shapes match the real ACF from `calculate_acf`.
    channel_width_mhz : float
        Frequency bin width in MHz for unit conversion.
    """
    key = (id(noise_desc), spec_len, channel_width_mhz, mask_hash)
    if key in _noise_acf_cache:
        return _noise_acf_cache[key]

    acfs = []
    for _ in range(n_rep):
        noise_row = noise_desc.sample()[0]      # (nchan,) synthetic row
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
    """
    Calculates the one-sided autocorrelation function of a spectrum using
    efficient NumPy operations.

    Parameters
    ----------
    spectrum_1d : np.ma.MaskedArray
        The 1D spectrum to autocorrelate. Must be a masked array.
    channel_width_mhz : float
        The channel width in MHz.
    off_burst_spectrum_mean : float, optional
        The mean of the off-burst spectrum, used for normalization.
    max_lag_bins : int, optional
        The maximum number of bins to compute the ACF out to.

    Returns
    -------
    ACF: object
    """
    log.debug(f"Calculating ACF for a spectrum of length {len(spectrum_1d)}.")
    valid_spec = spectrum_1d.compressed()
    if valid_spec.size < 10: return None
    
    mean_on = np.mean(valid_spec)
    
    # Define the normalization denominator for measuring the modulation index
    denom = (mean_on - off_burst_spectrum_mean)**2 if off_burst_spectrum_mean is not None else mean_on**2
    if denom == 0: denom = 1.0

    # Prepare the mean-subtracted spectrum, using NaN for masked values
    x = spectrum_1d.filled(np.nan) - mean_on
    n_chan = len(x)
    if max_lag_bins is None: max_lag_bins = n_chan
    
    lags = np.arange(1, max_lag_bins)
    acf_vals = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        # Create two shifted versions of the array
        v1, v2 = x[:-lag], x[lag:]
        # The product will be NaN if either element was originally masked
        prod = v1 * v2
        # Count only the pairs where both elements were valid
        num_valid = np.sum(~np.isnan(prod))
        if num_valid > 1:
            # Sum only the valid (non-NaN) products in numerator
            acf_vals[i] = np.nansum(prod) / (num_valid * denom)
            
    pos_lags_mhz = lags * channel_width_mhz
    full_acf = np.concatenate((acf_vals[::-1], acf_vals))
    full_lags = np.concatenate((-pos_lags_mhz[::-1], pos_lags_mhz))
    
    return ACF(full_acf, full_lags)

def calculate_acfs_for_subbands(masked_spectrum, config, burst_lims, noise_desc=None):
    """Calculate spectral ACFs for each frequency sub‑band of a burst.

    This upgraded version (a) removes the mean radiometer‑noise contribution via
    Monte‑Carlo synthetic spectra and (b) records σ_self so that downstream
    model fits can add a fixed‑width Gaussian self‑noise term.
    """
    log.info("Starting sub‑band ACF calculations (self‑noise + synthetic‑noise aware).")

    analysis_cfg = config.get("analysis", {})
    acf_cfg = analysis_cfg.get("acf", {})
    
    n_rep = analysis_cfg.get('noise', {}).get('template_n_draws', 15)
    use_template = not analysis_cfg.get("noise", {}).get("disable_template", False)

    n_sub = acf_cfg.get("num_subbands", 8)
    use_snr = acf_cfg.get("use_snr_subbanding", False)
    max_lag_mhz_global = acf_cfg.get("max_lag_mhz", 45.0)

    # Self‑noise width and optional off‑burst reference
    if config.get("analysis", {}).get("self_noise", {}).get("disable", False):
        sigma_self_mhz = None      # ← skip Gaussian in every sub-band
    else:
        sigma_self_mhz = _estimate_sigma_self(masked_spectrum, burst_lims)
        if sigma_self_mhz is None:
            log.warning("Could not estimate σ_self; Gaussian self‑noise term will be skipped.")

    if noise_desc is None:
        # Legacy off‑burst mean estimate for downward compatibility
        rfi_cfg = analysis_cfg.get("rfi_masking", {})
        if rfi_cfg.get("use_symmetric_noise_window", False):
            on_dur = burst_lims[1] - burst_lims[0]
            off_end = max(burst_lims[0] - 1, 0)
            off_start = max(off_end - on_dur, 0)
        else:
            off_end = max(burst_lims[0] - rfi_cfg.get("off_burst_buffer", 100), 0)
            off_start = 0
        off_burst_spec = masked_spectrum.get_spectrum((off_start, off_end))
    else:
        off_burst_spec = None  # not used when we have a descriptor

    # Prepare results container
    results = {
        "subband_acfs": [],
        "subband_lags_mhz": [],
        "subband_center_freqs_mhz": [],
        "subband_channel_widths_mhz": [],
        "subband_num_channels": [],
        "noise_template": [],
        "sigma_self_mhz": sigma_self_mhz,
    }

    # Split burst‑integrated spectrum into sub‑bands (uniform or equal‑S/N)
    burst_spec_full = masked_spectrum.get_spectrum(burst_lims)
    start_idx = 0
    total_signal = np.sum(burst_spec_full.compressed())

    for i in tqdm(range(n_sub), desc="ACF per sub‑band"):
        # Decide indices [start_idx:end_idx)
        if not use_snr:
            sub_len = masked_spectrum.num_channels // n_sub
            end_idx = start_idx + sub_len if i < n_sub - 1 else masked_spectrum.num_channels
        else:
            target_signal = total_signal / n_sub
            cum_sig = 0.0
            end_idx = start_idx
            while cum_sig < target_signal and end_idx < masked_spectrum.num_channels:
                if not burst_spec_full.mask[end_idx]:
                    cum_sig += burst_spec_full.data[end_idx]
                end_idx += 1
            if i == n_sub - 1:
                end_idx = masked_spectrum.num_channels  # ensure coverage

        sub_spec = burst_spec_full[start_idx:end_idx]
        sub_freqs = masked_spectrum.frequencies[start_idx:end_idx]

        # Off‑burst mean for normalisation (noise‑aware if descriptor is present)
        if noise_desc is not None:
            sub_off_mean = noise_desc.mu if noise_desc.kind == "intensity" else 0.0
        else:
            sub_off_mean = np.ma.mean(off_burst_spec[start_idx:end_idx])

        # Basic dimensions
        if len(sub_freqs) < 2:
            log.warning("Sub‑band %d too narrow; skipped.", i)
            start_idx = end_idx
            continue
        chan_width = float(np.abs(np.mean(np.diff(sub_freqs))))
        available_bw = sub_spec.count() * chan_width
        max_lag_mhz = min(max_lag_mhz_global, available_bw)
        max_lag_bins_sub = int(max_lag_mhz / chan_width)

        # ACF calculation – base object
        acf_obj = calculate_acf(
            sub_spec,
            chan_width,
            off_burst_spectrum_mean=sub_off_mean,
            max_lag_bins=max_lag_bins_sub,
        )
        if not acf_obj:
            start_idx = end_idx
            continue

        #  Synthetic-noise template handling
        mean_noise_acf = None
        if noise_desc is not None and use_template:
            real_mask_hash = hash(sub_spec.mask.tobytes()) # ← mask-aware key
            mean_noise_acf = _mean_noise_acf(
                        noise_desc,
                        n_rep=n_rep,
                        spec_len=len(acf_obj.acf),
                        channel_width_mhz=chan_width,
                        mask_hash=real_mask_hash)
            if mean_noise_acf is not None:
                # normalise so fitted 'amp' really is the radiometer m-value
                centre = len(mean_noise_acf) // 2
                if mean_noise_acf[centre] != 0:
                    mean_noise_acf /= mean_noise_acf[centre]

        # Store results
        results["noise_template"].append(mean_noise_acf)
        results["subband_acfs"].append(acf_obj.acf)
        results["subband_lags_mhz"].append(acf_obj.lags)
        results["subband_center_freqs_mhz"].append(float(np.mean(sub_freqs)))
        results["subband_channel_widths_mhz"].append(chan_width)
        results["subband_num_channels"].append(sub_spec.count())

        start_idx = end_idx  # next sub‑band
        log.debug(f"Cache now holds {len(_noise_acf_cache)} noise ACF template(s)")

    return results

def _make_noise_model(template, lags):
    """Return (Model, Parameters) with one free amp parameter."""
    shape = template / template[len(template)//2]        # unity at Δν=0
    f = interp1d(lags, shape, kind="linear",
                 bounds_error=False, fill_value=0.0)

    def noise_tpl(x, amp):
        return amp * f(x)

    nmod = Model(noise_tpl, prefix="n_")
    p    = nmod.make_params(amp=0.2, min=0, max=2.0)     # free amplitude
    return nmod, p

def _fit_acf_models(acf_object,
                    fit_lagrange_mhz: float,
                    *,
                    sigma_self_mhz: Optional[float] = None,
                    noise_template: Optional[np.ndarray] = None,
                    config=None):
    """
    Fit every scattering candidate to one ACF.

    Parameters
    ----------
    acf_object : ACF
    fit_lagrange_mhz : float
    sigma_self_mhz : float or None
        Fixed pulse-width self-noise HWHM.  If supplied, skip baseline-only fits.
    noise_template : 1-D ndarray or None
        Pre-computed radiometer-noise ACF shape for this sub-band.
        If None, the fitter omits that component.
    """
    fit_results: Dict[str, Optional["lmfit.ModelResult"]] = {}

    # --- data slice & weights ---------------------------------------------
    m = (np.abs(acf_object.lags) <= fit_lagrange_mhz) & (acf_object.lags != 0)
    x, y = acf_object.lags[m], acf_object.acf[m]
    w = None if acf_object.err is None else 1.0 / np.maximum(acf_object.err[m], 1e-9)

    # --- optional components ----------------------------------------------
    has_sn   = sigma_self_mhz is not None
    has_tpl  = noise_template is not None

    if has_sn:
        sn_model, sn_params = _self_noise_model(sigma_self_mhz)
    if has_tpl:
        tpl_model, tpl_params = _make_noise_model(noise_template, acf_object.lags)

    # --- iterate over baseline registry -----------------------------------
    init_cfg = config.get('analysis', {}).get('fitting', {}).get('init_guess', {})
    for key, mfn, prefix, seed, hook in _baseline_registry(init_cfg):

        base = Model(mfn, prefix=prefix)
        p0   = base.make_params(**seed)   # create once
        if hook:
            hook(p0)                      # inject bounds / ties

        # decide which composite we will fit -------------------------------
        if has_sn and has_tpl:
            model  = sn_model + tpl_model + base
            params = sn_params.copy() + tpl_params.copy() + p0.copy()

        elif has_sn:                      # self-noise only
            model  = sn_model + base
            params = sn_params.copy() + p0.copy()

        elif has_tpl:                     # template only
            model  = tpl_model + base
            params = tpl_params.copy() + p0.copy()

        else:                             # baseline only
            model, params = base, p0.copy()

        # run the fit -------------------------------------------------------
        label = f"fit_{'sn_tpl_' if has_sn and has_tpl else 'sn_' if has_sn else 'tpl_' if has_tpl else ''}{key}"
        try:
            
            # Print the model being tested for the current sub-band
            print(f"\n--- Initial Guesses for: {label} ---")
            # Print a detailed list of the parameters, their values, and bounds
            print(params)
            
            fit_results[label] = model.fit(y, params, x=x, weights=w,
                                           method='nelder', max_nfev=4000)
        except Exception as e:
            log.debug(f"{label} failed ({e})")
            fit_results[label] = None

        # keep legacy key when baseline-only was skipped
        if has_sn and not has_tpl:
            fit_results.setdefault(f"fit_{key}", None)

    return fit_results

def _fit_acf_models_v0(acf_object, fit_lagrange_mhz: float, sigma_self_mhz: Optional[float] = None):
    """Fit every scattering candidate to one ACF.

    If *sigma_self_mhz* is given we **always** include the fixed‑width
    self‑noise Gaussian and *skip* the baseline‐only fit to save time.
    The function still returns a dictionary with all keys; baseline keys that
    were not evaluated are set to ``None`` so downstream code remains stable.
    """

    fit_results: Dict[str, Optional["lmfit.ModelResult"]] = {}

    # 0.  Prepare data & weights ------------------------------------------------
    mask = (np.abs(acf_object.lags) <= fit_lagrange_mhz) & (acf_object.lags != 0)
    x = acf_object.lags[mask]
    y = acf_object.acf[mask]
    w = None if acf_object.err is None else 1.0 / np.maximum(acf_object.err[mask], 1e-9)

    # 1.  Self‑noise component (optional) --------------------------------------
    has_sn = sigma_self_mhz is not None
    if has_sn:
        sn_model, sn_params = _self_noise_model(sigma_self_mhz)

    # 2.  Iterate through baseline candidate registry --------------------------
    for key, model_fn, prefix, init_vals, hook in _baseline_registry():
        # Build baseline model (needed by combo regardless)
        base_model  = Model(model_fn, prefix=prefix)
        base_params = base_model.make_params(**init_vals)
        if hook is not None:
            hook(base_params)

        # --- (a) baseline‑only fit -------------------------------------------
        if not has_sn:
            try:
                fit_results[f"fit_{key}"] = base_model.fit(y, base_params, x=x, weights=w)
            except Exception as e:
                log.debug(f"Baseline model '{key}' failed: {e}")
                fit_results[f"fit_{key}"] = None
        else:
            # keep dict key for consistency
            fit_results[f"fit_{key}"] = None

        # --- (b) self‑noise + baseline ---------------------------------------
        if has_sn:
            try:
                combo_model  = sn_model + base_model
                combo_params = sn_params.copy() + base_model.make_params(**init_vals)
                if hook is not None:
                    hook(combo_params)
                fit_results[f"fit_sn_{key}"] = combo_model.fit(y, combo_params, x=x, weights=w)
            except Exception as e:
                log.debug(f"Self‑noise+{key} fit failed: {e}")
                fit_results[f"fit_sn_{key}"] = None

    return fit_results


def _select_overall_best_model_new(all_subband_fits):
    totals = defaultdict(float)
    counts = defaultdict(int)

    for band_fits in all_subband_fits:
        for name, fit in band_fits.items():
            if fit and fit.success:
                totals[name] += fit.bic
                counts[name] += 1

    # Require completeness: same number of successes as sub-bands
    n_bands = len(all_subband_fits)
    complete = {k: v for k, v in totals.items() if counts[k] == n_bands}

    if not complete:
        raise RuntimeError("No model fit succeeded on all sub-bands.")

    best = min(complete, key=complete.get)          # minimum total BIC
    return best

def _select_overall_best_model(all_subband_fits):
    """
    Determines the best overall model by summing the BIC across all sub-bands
    for each model type and selecting the one with the lowest total BIC.
    
    Keep the pretty log ordering (optional):
    ----------------------------------------
    
    for model_name in sorted(model_bics):
    bic_entry = model_bics[model_name]
    if bic_entry['count'] > 0:
        avg_bic = bic_entry['total_bic'] / bic_entry['count']
        log.info(f"{model_name:>20s}:  Total BIC = {avg_bic:7.1f}  "
                 f"(from {bic_entry['count']:2d} fits)")
    """
    # Use a dictionary to store total BICs and fit counts for each model
    model_bics = defaultdict(lambda: {'total_bic': 0.0, 'count': 0})
    
    for fits in all_subband_fits:
        for model_name, fit_result in fits.items():
            if fit_result and fit_result.success:
                model_bics[model_name]['total_bic'] += fit_result.bic
                model_bics[model_name]['count'] += 1

    log.info("--- Model Comparison (Lowest Total BIC is Best) ---")
    
    best_model = None
    min_bic = float('inf')

    for model_name, results in model_bics.items():
        if results['count'] > 0:
            log.info(f"Model '{model_name}': Total BIC = {results['total_bic']:.2f} (from {results['count']} fits)")
            if results['total_bic'] < min_bic:
                min_bic = results['total_bic']
                best_model = model_name
        else:
            log.info(f"Model '{model_name}': No successful fits.")
    
    if best_model is None:
        log.warning("No successful fits for any model. Defaulting to 'fit_1c_lor'.")
        return 'fit_1c_lor'

    log.info(f"==> Best overall model selected: {best_model}")
    return best_model

def analyze_scintillation_from_acfs(acf_results, config):
    """
    Main analysis orchestrator. Fits multiple ACF models, selects the best one,
    and derives scintillation parameters, including goodness-of-fit checks.
    """
    fit_config = config.get('analysis', {}).get('fitting', {})
    fit_lagrange_mhz = fit_config.get('fit_lagrange_mhz', 45.0)
    ref_freq = fit_config.get('reference_frequency_mhz', 600.0)

    log.info("Fitting all ACF models to all sub-band ACFs...")
    all_fits = []
    noise_templates = acf_results.get("noise_template", None)
    sigma_self_mhz = acf_results.get('sigma_self_mhz', None)
    for i in tqdm(range(len(acf_results['subband_acfs'])), desc="Fitting Sub-band ACFs"):
        acf_data = acf_results['subband_acfs'][i]
        lags = acf_results['subband_lags_mhz'][i]
        sub_bandwidth = (acf_results['subband_num_channels'][i] * acf_results['subband_channel_widths_mhz'][i])
        current_fit_lagrange = min(fit_lagrange_mhz, sub_bandwidth / 2.0)
        tpl = noise_templates[i] if noise_templates else None
        fit_result = _fit_acf_models(
                ACF(acf_data, lags),
                current_fit_lagrange,
                sigma_self_mhz=sigma_self_mhz,
                noise_template=tpl,
                config=config)  
        all_fits.append(fit_result)

    # 1. Get the automatically selected best model via BIC as a default.
    auto_best_model = _select_overall_best_model(all_fits)
    
    # 2. Check the config for a user-forced model.
    forced_model = fit_config.get('force_model')
    
    if forced_model:
        # Check if the forced model is a valid option
        valid_models = all_fits[0].keys() if all_fits else []
        if forced_model in valid_models:
            log.warning(f"OVERRIDE: User has forced the model to '{forced_model}'. Bypassing BIC selection.")
            best_model_name = forced_model
        else:
            log.error(f"Invalid model '{forced_model}' specified in config. Falling back to automatic BIC selection.")
            log.info(f"Valid model names are: {list(valid_models)}")
            best_model_name = auto_best_model
    else:
        # If no model is forced, use the automatic selection.
        best_model_name = auto_best_model
    
    # Logic for determining the number of components was not robust.
    if '3c' in best_model_name:
        num_comps = 3
    elif '2c' in best_model_name or 'unresolved' in best_model_name:
        num_comps = 2
    else:
        num_comps = 1
    
    params_per_comp = [[] for _ in range(num_comps)]
    
    for i, fits in enumerate(all_fits):
        fit_obj = fits.get(best_model_name)
        
        if not (fit_obj and fit_obj.success):
            for comp_list in params_per_comp: comp_list.append({})
            continue

        p = fit_obj.params
        sub_bw = acf_results['subband_num_channels'][i] * acf_results['subband_channel_widths_mhz'][i]
        gof_metrics = {'bic': fit_obj.bic, 'redchi': fit_obj.redchi}

        def get_bw_params(param_name, is_gauss):
            val = p[param_name].value
            err = p[param_name].stderr if p[param_name].stderr is not None else np.nan
            if is_gauss:
                hwhm_factor = np.sqrt(2 * np.log(2))
                return val * hwhm_factor, err * hwhm_factor
            return val, err
        
        def get_mod_err(param_name):
            param = p.get(param_name)
            return param.stderr if param is not None and param.stderr is not None else np.nan

        component_params = []
        if '1c' in best_model_name:
            is_gauss = 'gauss' in best_model_name
            prefix = 'g1_' if is_gauss else 'l1_'
            p_root = 'sigma' if is_gauss else 'gamma'
            bw, bw_err = get_bw_params(f'{prefix}{p_root}1', is_gauss)
            mod = p[f'{prefix}m1'].value
            mod_err = get_mod_err(f'{prefix}m1')
            component_params.append((bw, mod, bw_err, mod_err))

        elif '2c' in best_model_name and 'mixed' not in best_model_name and 'unresolved' not in best_model_name:
            is_gauss = 'gauss' in best_model_name
            prefix = 'g2_' if is_gauss else 'l2_'
            p_root = 'sigma' if is_gauss else 'gamma'
            for j in range(1, 3):
                bw, bw_err = get_bw_params(f'{prefix}{p_root}{j}', is_gauss)
                mod = p[f'{prefix}m{j}'].value
                mod_err = get_mod_err(f'{prefix}m{j}')
                component_params.append((bw, mod, bw_err, mod_err))
        
        # Added case for the two_screen_unresolved_model
        elif 'unresolved' in best_model_name:
            prefix = 'tsu_'
            p_root = 'gamma'
            for j in range(1, 3):
                bw, bw_err = get_bw_params(f'{prefix}{p_root}{j}', is_gauss=False)
                mod = p[f'{prefix}m{j}'].value
                mod_err = get_mod_err(f'{prefix}m{j}')
                component_params.append((bw, mod, bw_err, mod_err))

        elif '3c' in best_model_name:
            is_gauss = 'gauss' in best_model_name
            prefix = 'g3_' if is_gauss else 'l3_'
            p_root = 'sigma' if is_gauss else 'gamma'
            for j in range(1, 4):
                bw, bw_err = get_bw_params(f'{prefix}{p_root}{j}', is_gauss)
                mod = p[f'{prefix}m{j}'].value
                mod_err = get_mod_err(f'{prefix}m{j}')
                component_params.append((bw, mod, bw_err, mod_err))

        elif best_model_name == 'fit_2c_mixed':
            bw_g, bw_err_g = get_bw_params('gl_sigma1', is_gauss=True)
            bw_l, bw_err_l = get_bw_params('gl_gamma2', is_gauss=False)
            mod_g, mod_l = p['gl_m1'].value, p['gl_m2'].value
            mod_err_g, mod_err_l = get_mod_err('gl_m1'), get_mod_err('gl_m2')
            component_params.extend([(bw_g, mod_g, bw_err_g, mod_err_g), (bw_l, mod_l, bw_err_l, mod_err_l)])
        
        # Sort components by bandwidth (narrowest first) before assigning
        for comp_idx, (bw, mod, bw_err, mod_err) in enumerate(sorted(component_params)):
            num_scintles = max(1, sub_bw / bw) if bw > 0 else 1
            finite_err = bw / (2 * np.sqrt(num_scintles))
            param_dict = {'bw': bw, 'mod': mod, 'bw_err': bw_err, 'mod_err': mod_err, 'finite_err': finite_err}
            if comp_idx < len(params_per_comp):
                if comp_idx == 0: param_dict['gof'] = gof_metrics
                params_per_comp[comp_idx].append(param_dict)

    final_results = {'best_model': best_model_name, 'components': {}}
    all_powerlaw_fits = {}
    
    for i, params_list in enumerate(params_per_comp):
        name = f'component_{i+1}' if num_comps > 1 else 'scint_scale'
        measurements = [p for p in params_list if 'bw' in p]
        
        # Check for non-positive values before taking log
        if not all(p.get('bw', -1) > 0 for p in measurements):
            log.warning(f"Skipping power-law fit for {name}: contains non-positive bandwidths.")
            final_results['components'][name] = {'power_law_fit_report': 'Fit failed: Non-positive BWs'}
            continue

        freqs = np.array([acf_results['subband_center_freqs_mhz'][j] for j, p in enumerate(params_list) if 'bw' in p])
        bws = np.array([p.get('bw') for p in measurements])
        bw_errs = np.array([p.get('bw_err') for p in measurements])
        finite_errs = np.array([p.get('finite_err') for p in measurements])
        total_errs = np.sqrt(np.nan_to_num(bw_errs)**2 + np.nan_to_num(finite_errs)**2)

        ### FIX: Perform the fit in log-space for numerical stability ###

        # Log-transform the data and errors
        log_freqs = np.log10(freqs)
        log_bws = np.log10(bws)
        # Error propagation: err(log10(y)) = err(y) / (y * ln(10))
        log_bw_errs = total_errs / (bws * np.log(10))

        # Define a linear model: f(x) = slope*x + intercept
        linear_model = ModelODR(lambda B, x: B[0]*x + B[1])
        data = RealData(log_freqs, log_bws, sy=log_bw_errs)
        
        # Initial guess: slope (alpha) = 4, intercept can be 0
        odr = ODR(data, linear_model, beta0=[4.0, 0.0])
        out = odr.run()

        # Extract results. B[0] is the slope alpha, B[1] is log10(c)
        alpha_fit, log_c_fit = out.beta
        alpha_err, log_c_err = out.sd_beta
        c_fit = 10**log_c_fit
        
        # Propagate error for bandwidth at reference frequency
        log_ref_freq = np.log10(ref_freq)
        log_b_ref = alpha_fit * log_ref_freq + log_c_fit
        b_ref = 10**log_b_ref
        
        # Gradient for error propagation in log space
        grad = np.array([log_ref_freq, 1.0])
        var_log_b_ref = grad @ out.cov_beta @ grad
        # Convert error from log-space back to linear space
        b_ref_err = b_ref * np.sqrt(var_log_b_ref) * np.log(10)

        all_powerlaw_fits[name] = out
        
        # ================================================================= #
        # Use the fitted alpha and its error to suggest a 
        # physical scenario based on the findings from Pradeep et al. (2025)
        # and Nimmo et al. (2025).

        interpretation = "Undetermined"
        # Check for consistency within 3-sigma of the theoretical values
        if alpha_err is not None and not np.isnan(alpha_err):
            if abs(alpha_fit - 4.0) < 3 * alpha_err:
                # α ≈ 4 is expected for an unresolved point source 
                interpretation = "Consistent with unresolved point source (α ≈ 4)"
            elif abs(alpha_fit - 3.0) < 3 * alpha_err:
                # α ≈ 3 is expected for a screen resolving an incoherent emission region 
                interpretation = "Consistent with resolved emission region (α ≈ 3)"
            elif abs(alpha_fit - 1.0) < 3 * alpha_err:
                # α ≈ 1 is the flattened slope for two fully resolving screens 
                interpretation = "Consistent with resolved screens (α ≈ 1)"
        # ================================================================= #

        subband_measurements = []
        for j, p_dict in enumerate(measurements):
            measurement = {
                'freq_mhz': freqs[j], 'bw': p_dict.get('bw'), 'mod': p_dict.get('mod'),
                'bw_err': p_dict.get('bw_err'), 'mod_err': p_dict.get('mod_err'),
                'finite_err': p_dict.get('finite_err'), 'gof': p_dict.get('gof', {})
            }
            subband_measurements.append(measurement)
            
        final_results['components'][name] = {
            'power_law_fit_report': [c_fit, alpha_fit], # Store linear-space c and slope alpha
            'scaling_index': alpha_fit, 
            'scaling_index_err': alpha_err,
            'bw_at_ref_mhz': b_ref, 
            'bw_at_ref_mhz_err': b_ref_err,
            'subband_measurements': subband_measurements,
            'scaling_interpretation': interpretation
        }
    
    return final_results, all_fits, all_powerlaw_fits

def analyze_intra_pulse_scintillation(masked_spectrum, burst_lims, config, noise_desc):
    """
    Analyzes the evolution of scintillation parameters across the burst profile.

    This function divides the on-pulse data into time slices, calculates the ACF
    for each, and fits a model to track the evolution of the decorrelation
    bandwidth and modulation index.

    Args:
        masked_spectrum (DynamicSpectrum): The processed dynamic spectrum.
        burst_lims (tuple): The (start, end) time bins of the on-pulse region.
        config (dict): The analysis configuration dictionary.
        noise_desc (NoiseDescriptor): A pre-calculated noise descriptor for ACF normalization.

    Returns:
        list: A list of dictionaries, where each dictionary contains the fitted
              parameters ('time_s', 'bw', 'bw_err', 'mod', 'mod_err') for one time slice.
              Returns an empty list if the analysis cannot be run.
    """
    log.info("Starting intra-pulse scintillation analysis...")
    acf_config = config.get('analysis', {}).get('acf', {})
    fit_config = config.get('analysis', {}).get('fitting', {})
    analysis_cfg = config.get('analysis', {})  
    
    if analysis_cfg.get('self_noise', {}).get('disable', False):
        sigma_self_mhz = None
    else:
        sigma_self_mhz = _estimate_sigma_self(masked_spectrum, burst_lims)
    
    noise_template = None          # always None for temporal analysis

    num_time_bins  = acf_config.get('intra_pulse_time_bins', 10)
    model_to_fit   = fit_config.get('intra_pulse_fit_model', 'fit_1c_lor')
    max_lag_mhz    = acf_config.get('max_lag_mhz', 45.0)
    
    if '1c' not in model_to_fit:
        log.error(f"Model '{model_to_fit}' is not a 1-component model. Intra-pulse analysis requires a simple model to track evolution. Aborting.")
        return []

    results = []
    
    on_pulse_start, on_pulse_end = burst_lims
    total_duration_bins = on_pulse_end - on_pulse_start
    slice_width_bins = total_duration_bins // num_time_bins

    if slice_width_bins < 2:
        log.warning("Burst duration is too short for the number of requested time slices. Skipping intra-pulse analysis.")
        return []

    for i in tqdm(range(num_time_bins), desc="Analyzing ACF vs. Time"):
        start_bin = on_pulse_start + (i * slice_width_bins)
        end_bin = start_bin + slice_width_bins

        sub_spectrum = masked_spectrum.power[:, start_bin:end_bin].mean(axis=1)
        if sub_spectrum.count() < 10:
            continue

        if noise_desc and noise_desc.kind == "intensity":
            sub_off_mean = noise_desc.mu
        else:
            sub_off_mean = 0.0

        # Calculate max_lag_bins before calling the function 
        channel_width = masked_spectrum.channel_width_mhz
        if channel_width > 0:
            max_lag_bins_sub = int(max_lag_mhz / channel_width)
        else:
            continue # Cannot proceed without a valid channel width
            
        # Calculate ACF
        acf_obj = calculate_acf(
            sub_spectrum,
            channel_width,
            off_burst_spectrum_mean=sub_off_mean,
            max_lag_bins=max_lag_bins_sub  
        )
        if not acf_obj:
            continue

        # Fit models to the ACF
        fit_results = _fit_acf_models(
            acf_obj,
            fit_lagrange_mhz = fit_config.get('fit_lagrange_mhz', 45.0),
            sigma_self_mhz   = sigma_self_mhz,
            noise_template   = noise_template,    # always None here
            config=config
        )
        fit_obj = fit_results.get(model_to_fit)
        if not (fit_obj and fit_obj.success):
            continue
            
        # Determine the exact lags used for the fit
        fit_lagrange = fit_config.get('fit_lagrange_mhz', 45.0)
        fit_mask = np.abs(acf_obj.lags) <= fit_lagrange
        fit_lags = acf_obj.lags[fit_mask] # These are the lags matching the best_fit array

        # Extract parameters from the 1-component fit
        p = fit_obj.params
        is_gauss = 'gauss' in model_to_fit
        prefix = 'g1_' if is_gauss else 'l1_'
        p_root = 'sigma' if is_gauss else 'gamma'
        
        bw_val = p[f'{prefix}{p_root}1'].value
        bw_err = p[f'{prefix}{p_root}1'].stderr if p[f'{prefix}{p_root}1'].stderr is not None else np.nan
        
        # Convert Gaussian sigma to HWHM if necessary
        if is_gauss:
            hwhm_factor = np.sqrt(2 * np.log(2))
            bw_val *= hwhm_factor
            if bw_err: bw_err *= hwhm_factor
            
        mod_val = p[f'{prefix}m1'].value
        mod_err = p[f'{prefix}m1'].stderr if p[f'{prefix}m1'].stderr is not None else np.nan

        # Calculate the central time of the bin
        center_time = np.mean(masked_spectrum.times[start_bin:end_bin])

        results.append({
            'time_s': center_time,
            'bw': bw_val,
            'bw_err': bw_err,
            'mod': mod_val,
            'mod_err': mod_err,
            'acf_lags': acf_obj.lags,      # Full lags for the raw ACF data
            'acf_data': acf_obj.acf,      # Raw ACF data
            'acf_fit_lags': fit_lags,     # Lags corresponding to the fit
            'acf_fit_best': fit_obj.best_fit, # The best-fit line
            'fit_success': fit_obj.success
        })


    log.info(f"Intra-pulse analysis complete. Found results for {len(results)} time slices.")
    return results

