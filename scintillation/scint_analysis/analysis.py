# ==============================================================================
# File: scint_analysis/scint_analysis/analysis.py
# ==============================================================================
import numpy as np
import logging

log = logging.getLogger(__name__)

try:
    import numba as nb
    _NUMBA = True
    log.info("Numba detected. Using JIT-accelerated ACF computations.")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _NUMBA = False
    log.info("Numba not found. Using pure Python ACF computations.")
from .core import ACF
from lmfit import Model, Parameters
from lmfit.models import ConstantModel
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.odr import RealData, ODR, Model as ModelODR
from typing import Dict, Callable, Optional, Tuple

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
    """Generalised (power‑law) Lorentzian without a baseline constant.

    C(x) = m1² / [1 + |x/γ₁|^{α+2}]

    *alpha = 0* reproduces the standard Lorentzian; *alpha = 5/3* is the
    Kolmogorov diffractive prediction.
    """
    z = np.abs(x / gamma)
    return (m ** 2.0) / (1.0 + z ** (alpha + 2.0))


def power_law_model(x: np.ndarray | float, c: float, n: float) -> np.ndarray:
    """Pure power-law tail without a baseline constant."""
    eps = 1.0e-12
    return c * (np.abs(x) + eps) ** n

# -----------------------------------------------------------------------------
# Fixed‑width self‑noise Gaussian
# -----------------------------------------------------------------------------

def gauss_fixed_width(x, sigma_self, m_self):
    """Pure Gaussian that models the pulse-width self‑noise component without a baseline constant."""
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

def _baseline_registry(cfg_init: dict | None = None):
    """Return a list describing **all** baseline scattering models.

    Parameters
    ----------
    cfg_init : dict | None
        Overrides for the initial‑guess dictionaries below, typically taken
        from YAML → ``analysis → fitting → init_guess``.
    """

    if cfg_init is None:
        cfg_init = {}

    def merge(seed: dict, tag: str):
        """Override hard‑coded seed with YAML values for the given *tag*."""
        merged = seed.copy()
        merged.update(cfg_init.get(tag, {}))
        return merged

    return [
        # ------------------------------------------------------------------
        #  SINGLE‑COMPONENT MODELS
        # ------------------------------------------------------------------

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
         merge(dict(lg_gamma=0.05, lg_alpha=5/3, lg_m=0.8),
               tag='lor_gen'),
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

if _NUMBA:
    @nb.njit(cache=True)
    def _acf_with_errs(x, lags, denom):
        nlag = lags.size
        acf_vals = np.empty(nlag, dtype=np.float64)
        stat_errs = np.empty(nlag, dtype=np.float64)
        for i in range(nlag):
            lag = lags[i]
            prod = x[:-lag] * x[lag:]
            count = 0
            sum_prod = 0.0
            sum_sq = 0.0
            for p in prod:
                if not np.isnan(p):
                    count += 1
                    sum_prod += p
                    sum_sq += p * p
            if count > 1:
                mean = sum_prod / count
                acf_vals[i] = mean / denom
                var = (sum_sq - count * mean * mean) / (count - 1)
                stat_errs[i] = np.sqrt(var / count) / denom
            else:
                acf_vals[i] = np.nan
                stat_errs[i] = np.nan
        return acf_vals, stat_errs

    @nb.njit(cache=True)
    def _acf_noerrs(x, lags, denom):
        nlag = lags.size
        acf_vals = np.empty(nlag, dtype=np.float64)
        for i in range(nlag):
            lag = lags[i]
            prod = x[:-lag] * x[lag:]
            count = 0
            sum_prod = 0.0
            for p in prod:
                if not np.isnan(p):
                    count += 1
                    sum_prod += p
            if count > 1:
                acf_vals[i] = sum_prod / (count * denom)
            else:
                acf_vals[i] = np.nan
        return acf_vals
else:
    def _acf_with_errs(x, lags, denom):
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
        return acf_vals, stat_errs

    def _acf_noerrs(x, lags, denom):
        acf_vals = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            v1, v2 = x[:-lag], x[lag:]
            prod = v1 * v2
            num_valid = np.sum(~np.isnan(prod))
            if num_valid > 1:
                acf_vals[i] = np.nansum(prod) / (num_valid * denom)
            else:
                acf_vals[i] = np.nan
        return acf_vals

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

    acf_vals, stat_errs = _acf_with_errs(x, lags, denom)

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

_noise_acf_cache: Dict[Tuple, np.ndarray] = {}
_noise_acf_cache_max_size: int = 100  # Limit cache size to prevent memory issues


def _noise_descriptor_hash(noise_desc) -> int:
    """Generate a stable hash for a NoiseDescriptor based on its content.
    
    This replaces the fragile id()-based caching with a content-based hash
    that remains valid even if the same descriptor object is modified.
    """
    if noise_desc is None:
        return 0
    # Hash based on the key statistical properties
    hash_components = (
        noise_desc.kind,
        noise_desc.nt,
        noise_desc.nchan,
        round(noise_desc.mu, 8),
        round(noise_desc.sigma, 8),
        round(noise_desc.gamma_k, 8),
        round(noise_desc.gamma_theta, 8),
        round(noise_desc.phi_t, 8),
        round(noise_desc.phi_f, 8),
    )
    return hash(hash_components)


def clear_noise_acf_cache():
    """Clear the noise ACF template cache.
    
    Call this between pipeline runs or when noise characteristics change.
    """
    global _noise_acf_cache
    _noise_acf_cache.clear()
    log.info("Noise ACF template cache cleared.")


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
        Number of synthetic rows to average. ≥100 is recommended for smoothness.
    spec_len : int
        Number of frequency bins (≥ 1 + 2·max_lag_bins) for which the ACF will be
        evaluated so that shapes match the real ACF from `calculate_acf`.
    channel_width_mhz : float
        Frequency bin width in MHz for unit conversion.
    mask_hash : int
        Hash of the mask array for cache keying.
    """
    global _noise_acf_cache
    
    # Use content-based hash instead of id()
    desc_hash = _noise_descriptor_hash(noise_desc)
    key = (desc_hash, spec_len, round(channel_width_mhz, 6), mask_hash)
    
    if key in _noise_acf_cache:
        log.debug(f"Using cached noise ACF template (key hash: {hash(key)})")
        return _noise_acf_cache[key]
    
    # Limit cache size to prevent memory issues
    if len(_noise_acf_cache) >= _noise_acf_cache_max_size:
        # Remove oldest entry (FIFO-like behavior)
        oldest_key = next(iter(_noise_acf_cache))
        del _noise_acf_cache[oldest_key]
        log.debug("Evicted oldest entry from noise ACF cache")

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
    acf_vals = _acf_noerrs(x, lags, denom)
            
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
        "subband_acfs_err": [],
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
        results["subband_acfs_err"].append(acf_obj.err)
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
                    sub_freq: float, # <-- Parameter added
                    sigma_self_mhz: Optional[float] = None,
                    noise_template: Optional[np.ndarray] = None,
                    config=None):
    """
    Fit every scattering candidate to one ACF.
    """
    fit_results: Dict[str, Optional["lmfit.ModelResult"]] = {}

    # --- data slice & weights ---
    m = (np.abs(acf_object.lags) <= fit_lagrange_mhz) & (acf_object.lags != 0)
    x, y = acf_object.lags[m], acf_object.acf[m]
    w = None if acf_object.err is None else 1.0 / np.maximum(acf_object.err[m], 1e-9)

    # --- optional components ---
    has_sn   = sigma_self_mhz is not None
    has_tpl  = noise_template is not None

    if has_sn:
        sn_model, sn_params = _self_noise_model(sigma_self_mhz)
    if has_tpl:
        tpl_model, tpl_params = _make_noise_model(noise_template, acf_object.lags)

    # --- iterate over baseline registry ---
    init_cfg = config.get('analysis', {}).get('fitting', {}).get('init_guess', {})
    # Note: The `_baseline_registry` needs to be called with the raw `init_cfg`
    for key, mfn, prefix, seed, hook in _baseline_registry(init_cfg):

        # Check if there is a frequency-evolving guess configuration for this model key
        if key in init_cfg and 'components' in init_cfg[key]:
            adv_cfg = init_cfg[key]
            ref_freq = adv_cfg['reference_frequency_mhz']
            p0_dict = adv_cfg.get('constants', {}).copy()

            for i, comp_cfg in enumerate(adv_cfg['components']):
                comp_idx = i + 1
                alpha = comp_cfg['gamma_scaling_index']
                gamma_ref = comp_cfg['gamma_guess']
                scaled_gamma = gamma_ref * (sub_freq / ref_freq) ** alpha
                p0_dict[f'{prefix}gamma{comp_idx}'] = scaled_gamma
                p0_dict[f'{prefix}m{comp_idx}'] = comp_cfg['m_guess']

        else:
            # Fallback to the old method (static guesses from YAML or defaults)
            merged_seed = seed.copy()
            if key in init_cfg:
                merged_seed.update(init_cfg.get(key, {}))
            p0_dict = merged_seed

        # Build the baseline model and its parameters from our calculated p0_dict
        base_model = Model(mfn, prefix=prefix)
        base_params = base_model.make_params(**p0_dict)
        if hook:
            hook(base_params)

        # Decide which composite we will fit, using the correctly generated base_params
        if has_sn and has_tpl:
            model  = sn_model + tpl_model + base_model
            params = sn_params.copy() + tpl_params.copy() + base_params.copy()

        elif has_sn:
            model  = sn_model + base_model
            params = sn_params.copy() + base_params.copy()

        elif has_tpl:
            model  = tpl_model + base_model
            params = tpl_params.copy() + base_params.copy()

        else:
            model, params = base_model, base_params.copy()

        # Run the fit
        label = f"fit_{'sn_tpl_' if has_sn and has_tpl else 'sn_' if has_sn else 'tpl_' if has_tpl else ''}{key}"
        try:
            fit_results[label] = model.fit(y, params, x=x, weights=w,
                                           method='nelder', max_nfev=4000)
        except Exception as e:
            log.debug(f"{label} failed ({e})")
            fit_results[label] = None

    return fit_results

def _interpret_scaling_index(alpha: float, alpha_err: float) -> str:
    """Interpret the frequency scaling index based on physical expectations.
    
    Based on theoretical predictions from Bhat et al. (2004) and Nimmo et al. (2025):
    - α ≈ 4.0-4.4: Kolmogorov turbulence (diffractive scintillation)
    - α ≈ 3.9: Global average with inner scale effects (Bhat et al. 2004)
    - α ≈ 3.0: Emission region partially resolved by screen (Nimmo et al. 2025)
    - α ≈ 2.0: Refractive scintillation
    - α ≈ 1.0: Two screens resolving each other (Nimmo et al. 2025)
    - α ≈ 0: No frequency dependence (intrinsic structure or instrumental)
    - α < 0 or α > 6: Likely unphysical, suggests fit issues
    
    Parameters
    ----------
    alpha : float
        The fitted scaling index (δν_DC ∝ ν^α)
    alpha_err : float
        The 1-σ uncertainty on alpha
        
    Returns
    -------
    str
        Human-readable interpretation of the scaling
    
    References
    ----------
    Bhat et al. 2004, ApJ, 605, 759 (multifrequency pulse broadening)
    Nimmo et al. 2025 (FRB 20221022A scintillation constraints)
    """
    # Handle NaN or invalid values
    if not np.isfinite(alpha) or not np.isfinite(alpha_err):
        return "Unable to determine (fit failed or invalid)"
    
    # Define physical regimes with 2-σ tolerance
    tol = 2.0 * alpha_err if alpha_err > 0 else 0.5
    
    if alpha < -1.0:
        return "Unphysical (negative scaling suggests fit issues or systematic errors)"
    elif abs(alpha - 0.0) < tol:
        return "No significant frequency scaling (intrinsic structure or instrumental)"
    elif 0.5 < alpha < 1.5:
        return (
            f"Very shallow scaling (α = {alpha:.2f} ± {alpha_err:.2f}) - "
            "may indicate two screens resolving each other (Nimmo et al. 2025 Eq. 27)"
        )
    elif abs(alpha - 2.0) < tol:
        return "Refractive scintillation regime (α ≈ 2)"
    elif 2.5 < alpha < 3.5:
        return (
            f"Intermediate scaling (α = {alpha:.2f} ± {alpha_err:.2f}) - "
            "emission region may be partially resolved by scattering screen "
            "(Nimmo et al. 2025 Eq. 26)"
        )
    elif 3.5 <= alpha <= 4.2:
        if abs(alpha - 3.9) < tol:
            return (
                f"Kolmogorov with inner scale effects (α = {alpha:.2f} ± {alpha_err:.2f}) - "
                "consistent with Bhat et al. 2004 global fit (α ≈ 3.9), "
                "suggests inner scale l_i ~ 300-800 km"
            )
        elif abs(alpha - 4.0) < tol:
            return "Kolmogorov diffractive scintillation (α ≈ 4.0)"
        else:
            return f"Diffractive scintillation regime (α = {alpha:.2f} ± {alpha_err:.2f})"
    elif 4.2 < alpha <= 5.0:
        if abs(alpha - 4.4) < tol:
            return "Classic Kolmogorov (α ≈ 4.4) - no significant inner scale effect"
        else:
            return f"Diffractive scintillation regime (α = {alpha:.2f} ± {alpha_err:.2f})"
    elif alpha > 5.0:
        return f"Steep scaling (α = {alpha:.2f}) - may indicate scattering-dominated regime"
    else:
        return f"Intermediate regime (α = {alpha:.2f} ± {alpha_err:.2f})"


# =============================================================================
# Modulation Index & Emission Region Diagnostics (Nimmo et al. 2025)
# =============================================================================

def interpret_modulation_index(m: float, m_err: float = 0.0) -> Dict:
    """Interpret modulation index based on Nimmo et al. (2025) framework.
    
    The modulation index m is defined as σ_I / <I>, where σ_I is the standard
    deviation of the intensity and <I> is the mean. For the ACF, the peak
    amplitude equals m².
    
    Physical interpretation:
    - m ≈ 1: Point source (unresolved emission region)
    - m < 1: Emission region partially resolved by scattering screen
    - m << 1 (0.1-0.3): Weak scintillation regime
    
    Parameters
    ----------
    m : float
        Measured modulation index from ACF fit
    m_err : float, optional
        Uncertainty on m
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'm': measured value
        - 'm_err': uncertainty
        - 'interpretation': human-readable interpretation
        - 'emission_resolved': bool, whether emission appears resolved
        - 'resolution_regime': categorical label
        
    References
    ----------
    Nimmo et al. 2025, FRB 20221022A scintillation analysis
    Rickett 1990, ARA&A, 28, 561 (scintillation theory)
    
    Examples
    --------
    >>> result = interpret_modulation_index(0.78, 0.07)
    >>> print(result['interpretation'])
    "Marginally resolved emission region (m = 0.78 ± 0.07)..."
    """
    result = {
        'm': m,
        'm_err': m_err,
        'interpretation': '',
        'emission_resolved': False,
        'resolution_regime': 'unknown',
    }
    
    if not np.isfinite(m):
        result['interpretation'] = "Invalid modulation index measurement"
        return result
    
    # Use error to define tolerance for "consistent with 1"
    tol = max(2.0 * m_err, 0.05) if m_err > 0 else 0.05
    
    if m > 1.0 + tol:
        result['interpretation'] = (
            f"Super-unity modulation (m = {m:.2f} ± {m_err:.2f}) - "
            "may indicate calibration issues, RFI contamination, or "
            "intrinsic intensity variations beyond scintillation"
        )
        result['resolution_regime'] = 'anomalous'
    elif abs(m - 1.0) <= tol or m > 0.95:
        result['interpretation'] = (
            f"Point source / unresolved emission (m = {m:.2f} ± {m_err:.2f}) - "
            "emission region is smaller than the diffractive scale of the "
            "scattering screen (Nimmo et al. 2025)"
        )
        result['resolution_regime'] = 'unresolved'
        result['emission_resolved'] = False
    elif 0.7 < m <= 0.95:
        result['interpretation'] = (
            f"Marginally resolved emission region (m = {m:.2f} ± {m_err:.2f}) - "
            "emission region is comparable to or slightly larger than the "
            "diffractive scale; consistent with magnetospheric emission "
            "(Nimmo et al. 2025 Fig. 3)"
        )
        result['resolution_regime'] = 'marginally_resolved'
        result['emission_resolved'] = True
    elif 0.3 < m <= 0.7:
        result['interpretation'] = (
            f"Partially resolved emission (m = {m:.2f} ± {m_err:.2f}) - "
            "emission region significantly resolved by scattering screen; "
            "may constrain emission mechanism and/or screen distance"
        )
        result['resolution_regime'] = 'partially_resolved'
        result['emission_resolved'] = True
    else:  # m <= 0.3
        result['interpretation'] = (
            f"Heavily suppressed modulation (m = {m:.2f} ± {m_err:.2f}) - "
            "either weak scintillation regime or very extended emission region; "
            "check if observation is in strong scintillation regime"
        )
        result['resolution_regime'] = 'weak_or_resolved'
        result['emission_resolved'] = True
    
    return result


def estimate_emission_region_size(
    m: float,
    delta_nu_dc_mhz: float,
    d_source_screen_pc: float,
    freq_mhz: float,
    m_err: float = 0.0,
    delta_nu_err_mhz: float = 0.0,
) -> Dict:
    """Estimate lateral emission region size from modulation index.
    
    Uses the relationship between modulation index and source resolution
    from Nimmo et al. (2025) Eq. 22-23:
    
        m = 1 / sqrt(1 + 4(R_★obs/χ)²)
        
        χ = (1/ν) * sqrt(c * d_s2★ * Δν_DC / 2π)  [screen resolution]
        
    Solving for R_★obs:
        R_★obs = sqrt((c * d_s2★ * Δν_DC) / (8π ν²) * (1/m² - 1))
    
    Parameters
    ----------
    m : float
        Measured modulation index (0 < m ≤ 1 for resolved sources)
    delta_nu_dc_mhz : float
        Decorrelation bandwidth in MHz
    d_source_screen_pc : float
        Distance from source to scattering screen in parsecs
    freq_mhz : float
        Observing frequency in MHz
    m_err : float, optional
        Uncertainty on modulation index
    delta_nu_err_mhz : float, optional
        Uncertainty on decorrelation bandwidth
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'R_obs_km': estimated emission region size in km
        - 'R_obs_err_km': uncertainty in km (if errors provided)
        - 'chi_km': diffractive scale of screen in km
        - 'is_upper_limit': bool, True if m ≈ 1 (gives upper limit)
        - 'physical_context': comparison to known scales
        
    References
    ----------
    Nimmo et al. 2025, Eq. 21-23
    Kumar et al. 2024, MNRAS, 527, 457 (FRB scintillation constraints)
    
    Examples
    --------
    >>> # FRB 20221022A parameters from Nimmo et al.
    >>> result = estimate_emission_region_size(
    ...     m=0.78, delta_nu_dc_mhz=0.124, d_source_screen_pc=11000,
    ...     freq_mhz=600, m_err=0.07
    ... )
    >>> print(f"R_obs < {result['R_obs_km']:.0f} km")
    """
    c_m_s = 2.998e8  # speed of light in m/s
    pc_to_m = 3.086e16  # parsec to meters
    
    # Convert units
    delta_nu_hz = delta_nu_dc_mhz * 1e6
    freq_hz = freq_mhz * 1e6
    d_m = d_source_screen_pc * pc_to_m
    
    result = {
        'R_obs_km': np.nan,
        'R_obs_err_km': np.nan,
        'chi_km': np.nan,
        'is_upper_limit': False,
        'physical_context': '',
    }
    
    # Calculate screen diffractive scale χ (Nimmo Eq. 21)
    # χ = (1/ν) * sqrt(c * d * Δν / 2π)
    chi_m = (1.0 / freq_hz) * np.sqrt(c_m_s * d_m * delta_nu_hz / (2 * np.pi))
    chi_km = chi_m / 1e3
    result['chi_km'] = chi_km
    
    # Handle edge cases
    if m >= 1.0:
        # Unresolved: R_obs < χ (upper limit)
        result['R_obs_km'] = chi_km
        result['is_upper_limit'] = True
        result['physical_context'] = (
            f"Unresolved (m ≥ 1): R_obs < χ = {chi_km:.1f} km (upper limit)"
        )
        return result
    
    if m <= 0.0:
        result['physical_context'] = "Invalid modulation index (m ≤ 0)"
        return result
    
    # Calculate R_★obs from Nimmo Eq. 23:
    # R_★obs = sqrt((c * d * Δν) / (8π ν²) * (1/m² - 1))
    factor = (c_m_s * d_m * delta_nu_hz) / (8 * np.pi * freq_hz**2)
    m_factor = (1.0 / m**2) - 1.0
    
    if m_factor <= 0:
        result['R_obs_km'] = 0.0
        result['physical_context'] = "Point source (m_factor ≤ 0)"
        return result
    
    R_obs_m = np.sqrt(factor * m_factor)
    R_obs_km = R_obs_m / 1e3
    result['R_obs_km'] = R_obs_km
    
    # Error propagation (simplified, assumes dominant error from m)
    if m_err > 0:
        # ∂R/∂m ∝ -m⁻³ * (1/m² - 1)^(-1/2) * sqrt(factor)
        dm_factor = np.sqrt(factor) * (m**(-3)) / np.sqrt(m_factor)
        R_obs_err_km = abs(dm_factor * m_err) / 1e3
        result['R_obs_err_km'] = R_obs_err_km
    
    # Physical context
    context_parts = [f"Estimated R_obs = {R_obs_km:.1f} km"]
    
    # Compare to known scales
    if R_obs_km < 100:
        context_parts.append("consistent with pulsar emission (~10-100 km)")
    elif R_obs_km < 1000:
        context_parts.append("consistent with pulsar/magnetar magnetosphere (~100-1000 km)")
    elif R_obs_km < 1e4:
        context_parts.append("consistent with neutron star light cylinder (~1000-10,000 km)")
    elif R_obs_km < 1e5:
        context_parts.append("larger than typical magnetosphere; may indicate shock emission")
    else:
        context_parts.append("very large; likely non-magnetospheric origin")
    
    result['physical_context'] = "; ".join(context_parts)
    
    return result


def two_screen_coherence_constraint(
    delta_nu_1_mhz: float,
    delta_nu_2_mhz: float,
    freq_mhz: float,
    d_source_mpc: float,
    C1: float = 1.0,
    C2: float = 1.0,
) -> Dict:
    """Calculate two-screen coherence constraint from Nimmo et al. (2025).
    
    When two scintillation scales are observed (e.g., one Galactic, one 
    extragalactic), mutual coherence requires:
    
        Δν_s1 * Δν_s2 >= C₁ * C₂ * ν² * (d_s1★ * d_s2★ * d_⊕s1) / (d_⊕★² * d_⊕s2)
    
    For an extragalactic source with one Galactic screen (s1) and one 
    host-galaxy screen (s2), this simplifies to (Nimmo Eq. 10):
    
        d_⊕s1 * d_s2★ <= Δν_s1 * Δν_s2 * d_⊕★² / (C₁ * C₂ * ν²)
    
    Parameters
    ----------
    delta_nu_1_mhz : float
        Decorrelation bandwidth of screen 1 (closest to observer) in MHz
    delta_nu_2_mhz : float
        Decorrelation bandwidth of screen 2 (closest to source) in MHz
    freq_mhz : float
        Observing frequency in MHz
    d_source_mpc : float
        Distance to source in Mpc
    C1, C2 : float, optional
        Geometry constants (typically 1-2), default 1.0
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'd_product_kpc2': upper limit on d_⊕s1 * d_s2★ in kpc²
        - 'example_constraints': dict with example screen distance scenarios
        
    References
    ----------
    Nimmo et al. 2025, Eq. 7-11
    
    Examples
    --------
    >>> # FRB 20221022A: 6 kHz and 124 kHz scales at 600 MHz, d = 65 Mpc
    >>> result = two_screen_coherence_constraint(
    ...     delta_nu_1_mhz=0.006, delta_nu_2_mhz=0.124,
    ...     freq_mhz=600, d_source_mpc=65.189
    ... )
    >>> print(f"d_⊕s1 * d_s2★ <= {result['d_product_kpc2']:.1f} kpc²")
    """
    # Convert to Hz
    delta_nu_1_hz = delta_nu_1_mhz * 1e6
    delta_nu_2_hz = delta_nu_2_mhz * 1e6
    freq_hz = freq_mhz * 1e6
    
    # Convert distance to meters then to kpc for result
    d_source_m = d_source_mpc * 3.086e22  # Mpc to m
    d_source_kpc = d_source_mpc * 1e3  # Mpc to kpc
    
    # Constraint: d_⊕s1 * d_s2★ <= Δν₁ * Δν₂ * d_⊕★² / (C₁ * C₂ * ν²)
    # Result in kpc² (after unit conversion)
    numerator = delta_nu_1_hz * delta_nu_2_hz * d_source_kpc**2
    denominator = C1 * C2 * freq_hz**2
    
    d_product_kpc2 = numerator / denominator
    
    result = {
        'd_product_kpc2': d_product_kpc2,
        'example_constraints': {},
    }
    
    # Example scenarios
    galactic_distances = [0.1, 0.3, 0.64, 1.0, 3.0]  # kpc
    for d_gal in galactic_distances:
        d_host_max = d_product_kpc2 / d_gal
        result['example_constraints'][f'd_gal_{d_gal}kpc'] = {
            'd_galactic_kpc': d_gal,
            'd_host_max_kpc': d_host_max,
            'd_host_max_pc': d_host_max * 1e3,
        }
    
    return result


def scattering_scintillation_consistency(
    tau_d_ms: float,
    delta_nu_dc_mhz: float,
    C: float = 1.0,
    tau_d_err_ms: float = 0.0,
    delta_nu_err_mhz: float = 0.0,
) -> Dict:
    """Check consistency between scattering timescale and decorrelation bandwidth.
    
    The fundamental relationship is (Bhat et al. 2004, Nimmo et al. 2025):
    
        τ_s = C / (2π Δν_DC)
    
    where C is a geometry-dependent constant typically in range 1-2.
    
    Parameters
    ----------
    tau_d_ms : float
        Measured scattering timescale in milliseconds
    delta_nu_dc_mhz : float
        Measured decorrelation bandwidth in MHz
    C : float, optional
        Geometry constant (default 1.0)
    tau_d_err_ms : float, optional
        Uncertainty on scattering timescale
    delta_nu_err_mhz : float, optional
        Uncertainty on decorrelation bandwidth
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'tau_from_scint_ms': τ predicted from scintillation measurement
        - 'nu_from_scat_mhz': Δν predicted from scattering measurement
        - 'C_implied': implied geometry constant
        - 'consistent': bool, whether measurements are consistent
        - 'interpretation': string describing consistency
        
    References
    ----------
    Bhat et al. 2004, ApJ, 605, 759
    Cordes & Rickett 1998, ApJ, 507, 846
    
    Examples
    --------
    >>> result = scattering_scintillation_consistency(
    ...     tau_d_ms=0.1, delta_nu_dc_mhz=0.05
    ... )
    >>> print(f"Implied C = {result['C_implied']:.2f}")
    """
    # Convert units
    tau_d_s = tau_d_ms * 1e-3
    delta_nu_hz = delta_nu_dc_mhz * 1e6
    
    result = {
        'tau_from_scint_ms': np.nan,
        'nu_from_scat_mhz': np.nan,
        'C_implied': np.nan,
        'consistent': False,
        'interpretation': '',
    }
    
    # τ predicted from scintillation: τ = C / (2π Δν)
    tau_from_scint_s = C / (2 * np.pi * delta_nu_hz)
    tau_from_scint_ms = tau_from_scint_s * 1e3
    result['tau_from_scint_ms'] = tau_from_scint_ms
    
    # Δν predicted from scattering: Δν = C / (2π τ)
    nu_from_scat_hz = C / (2 * np.pi * tau_d_s)
    nu_from_scat_mhz = nu_from_scat_hz / 1e6
    result['nu_from_scat_mhz'] = nu_from_scat_mhz
    
    # Implied C from measurements: C = 2π τ Δν
    C_implied = 2 * np.pi * tau_d_s * delta_nu_hz
    result['C_implied'] = C_implied
    
    # Check consistency (C should be in range ~0.5-2.5)
    if 0.3 < C_implied < 3.0:
        result['consistent'] = True
        result['interpretation'] = (
            f"Consistent: implied C = {C_implied:.2f} is within expected range (0.5-2). "
            f"τ_scint = {tau_from_scint_ms:.3f} ms, Δν_scat = {nu_from_scat_mhz:.4f} MHz"
        )
    elif C_implied < 0.3:
        result['consistent'] = False
        result['interpretation'] = (
            f"Inconsistent: C = {C_implied:.2f} << 1. "
            "Scattering may be from different screen than scintillation, "
            "or one measurement may be affected by systematics."
        )
    else:  # C > 3
        result['consistent'] = False
        result['interpretation'] = (
            f"Inconsistent: C = {C_implied:.2f} >> 1. "
            "May indicate multiple scattering screens, anisotropic scattering, "
            "or measurement systematics."
        )
    
    return result


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
        log.warning("No successful fits for any model. Defaulting to 'lorentzian_component'.")
        return 'lorentzian_component'

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
        sub_freq = acf_results['subband_center_freqs_mhz'][i] 
        sub_bandwidth = (acf_results['subband_num_channels'][i] * acf_results['subband_channel_widths_mhz'][i])
        current_fit_lagrange = min(fit_lagrange_mhz, sub_bandwidth / 2.0)
        tpl = noise_templates[i] if noise_templates else None
        fit_result = _fit_acf_models(
                ACF(acf_data, lags),
                current_fit_lagrange,
                sub_freq=sub_freq,
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

        # Handle different model types
        if 'power' in best_model_name:
            # Power-law model: C(Δν) = c · |Δν|^n
            # No direct "bandwidth" - use characteristic scale at 1 MHz
            prefix = 'p_'
            c_val = p[f'{prefix}c'].value
            c_err = p[f'{prefix}c'].stderr if p[f'{prefix}c'].stderr is not None else np.nan
            n_val = p[f'{prefix}n'].value
            n_err = p[f'{prefix}n'].stderr if p[f'{prefix}n'].stderr is not None else np.nan

            # For power-law, store c and n instead of bandwidth and modulation
            # Use c as proxy for "bandwidth scale" and set modulation to NaN
            bw, bw_err = c_val, c_err
            mod, mod_err = np.nan, np.nan  # Power-law has no modulation index
            component_params.append((bw, mod, bw_err, mod_err))

        elif 'lor_gen' in best_model_name or 'gen' in best_model_name:
            # Generalized Lorentzian: has gamma, alpha, m
            prefix = 'lg_'
            bw, bw_err = get_bw_params(f'{prefix}gamma', False)
            mod = p[f'{prefix}m'].value
            mod_err = get_mod_err(f'{prefix}m')
            component_params.append((bw, mod, bw_err, mod_err))

        else:
            # Standard Lorentzian or Gaussian
            is_gauss = 'gauss' in best_model_name
            prefix = 'g_' if is_gauss else 'l_'
            p_root = 'sigma' if is_gauss else 'gamma'
            bw, bw_err = get_bw_params(f'{prefix}{p_root}', is_gauss)
            mod = p[f'{prefix}m'].value
            mod_err = get_mod_err(f'{prefix}m')
            component_params.append((bw, mod, bw_err, mod_err))

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
        
        # Interpret the scaling index based on physical expectations
        interpretation = _interpret_scaling_index(alpha_fit, alpha_err)

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
    model_to_fit   = fit_config.get('intra_pulse_fit_model', 'lorentzian_component')
    max_lag_mhz    = acf_config.get('max_lag_mhz', 45.0)
    
    # NEW: Calculate the center frequency of the entire band once.
    band_center_freq = np.mean(masked_spectrum.frequencies)

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
            sub_freq         = band_center_freq, # MODIFIED: Use the pre-calculated band center frequency
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
        prefix = 'g_' if is_gauss else 'l_'
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



# ==============================================================================
# Fit Loading and Reconstruction (Added from notebook refactoring)
# ==============================================================================

def load_saved_fit(config_path, subband_index, model_name, lags, model_config=None):
    """
    Load a saved fit from YAML config and reconstruct the model curves.
    
    This function reads fit results stored in the YAML configuration,
    rebuilds the lmfit composite model, and evaluates it to regenerate
    the best-fit curve and component curves for plotting.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file containing stored fits
    subband_index : int
        Index of the sub-band (0-indexed)
    model_name : str
        Name of the model fit to load, e.g. "Lorentzian+Gaussian"
        This should match the key under stored_fits.subband_{i}
    lags : np.ndarray
        Lag array (MHz) on which to evaluate the model
    model_config : dict, optional
        Model configuration dictionary. If None, uses the default from widgets module.
        Format: {name: {'func': callable, 'prefix': str, 'param_names': list}}
    
    Returns
    -------
    dict or None
        Dictionary containing:
        - 'best_fit_curve': np.ndarray - composite model evaluated on lags
        - 'component_curves': dict - individual component curves
        - 'redchi': float - reduced chi-squared
        - 'bic': float - Bayesian Information Criterion  
        - 'params': dict - parameter values and errors
        - 'fit_range_mhz': list - lag range used for fitting
        
        Returns None if the fit cannot be found or loaded.
    
    Examples
    --------
    >>> from scint_analysis import analysis, plotting
    >>> lags = acf_obj.lags
    >>> fit_data = analysis.load_saved_fit(
    ...     "configs/bursts/freya_dsa.yaml",
    ...     subband_index=0,
    ...     model_name="Lorentzian+Gaussian",
    ...     lags=lags
    ... )
    >>> if fit_data:
    ...     plotting.plot_publication_acf(acf_obj, **fit_data)
    """
    import yaml
    from pathlib import Path
    from lmfit import Model
    from lmfit.models import ConstantModel
    
    # Use default model config if not provided
    if model_config is None:
        try:
            from .widgets import DEFAULT_MODEL_CONFIG
            model_config = DEFAULT_MODEL_CONFIG
        except ImportError:
            # Fallback: define minimal config
            model_config = {
                "Lorentzian": dict(func=lorentzian_component, prefix="l_", param_names=["gamma", "m"]),
                "Gaussian": dict(func=gaussian_component, prefix="g_", param_names=["sigma", "m"]),
                "Gen-Lorentz": dict(func=lorentzian_generalised, prefix="lg_", param_names=["gamma", "alpha", "m"]),
                "Power-Law": dict(func=power_law_model, prefix="p_", param_names=["c", "n"]),
            }
    
    config_path = Path(config_path)
    
    # Load YAML config
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        return None
    
    # Extract fit data
    try:
        fit_data = config_data['analysis']['stored_fits'][f'subband_{subband_index}'][model_name]
        saved_params = fit_data['best_fit_params']
        redchi = fit_data.get('redchi', np.nan)
        bic = fit_data.get('bic', np.nan)
        fit_range = fit_data.get('fit_range_mhz', [lags.min(), lags.max()])
    except KeyError as e:
        logging.error(f"Could not find saved fit for sub-band {subband_index} with model '{model_name}': {e}")
        return None
    
    # Rebuild the lmfit composite model
    component_keys = model_name.split('+')
    signal_model = None
    
    for i, key in enumerate(component_keys, start=1):
        if key not in model_config:
            logging.warning(f"Unknown model component '{key}', skipping")
            continue
        
        cfg = model_config[key]
        m = Model(cfg["func"], prefix=f"{cfg['prefix']}{i}_")
        
        if signal_model is None:
            signal_model = m
        else:
            signal_model += m
    
    if signal_model is None:
        logging.error(f"No valid signal models found in '{model_name}'")
        return None
    
    # Add constant offset
    const_model = ConstantModel(prefix='c_')
    composite_model = signal_model + const_model
    params = composite_model.make_params()
    
    # Set parameter values from saved dictionary
    for param_name, param_info in saved_params.items():
        if param_name in params:
            params[param_name].set(value=param_info['value'])
    
    # Evaluate curves
    best_fit_curve = composite_model.eval(params=params, x=lags)
    component_curves = composite_model.eval_components(params=params, x=lags)
    
    return {
        "best_fit_curve": best_fit_curve,
        "component_curves": component_curves,
        "redchi": redchi,
        "bic": bic,
        "params": saved_params,
        "fit_range_mhz": fit_range,
    }
