"""ACF re-validation harness following Nimmo et al. 2025 (arXiv:2406.11053, Nature;
FRB 20221022A) and the two-screen scintillometry of Pleunis et al. 2025
(arXiv:2505.04576, §5.1).

Independently re-measures the scintillation decorrelation bandwidth Δν on
RFI-flagged, off-pulse-aware spectra, and adds the two-component (Milky-Way wide +
host narrow) Lorentzian fit with the lag-0 self-noise spike omitted. The ACF
estimator and Lorentzian models are ported directly from the Nimmo et al. 2025
release ``scint_funcs.py`` (``autocorr`` / ``lorentz_w_c`` / ``doublelorentz_w_c`` /
``res`` / ``emission_size``), so a re-validation is *independent* of the pipeline's
own ACF code (``analysis.calculate_acf``) — which is the whole point of a cross-check.

Method (Nimmo 2025; Pleunis 2505.04576 §5.1):
  - mean-normalized ACF, ``ACF(δν)=⟨(I-⟨I⟩)/⟨I⟩·(I'-⟨I⟩)/⟨I⟩⟩``, zero-lag bin
    excluded (the frequency-uncorrelated self-noise spike; Eqs 4.22-4.23).
  - single screen: Δν = HWHM of ``m²/(1+(δν/γ)²)+C`` (Pleunis Eq 5.1; γ = Δν).
  - two screens: ``m1²/(1+(δν/γ1)²)+m2²/(1+(δν/γ2)²)+C``, center omitted.
  - modulation index ``m = sqrt(ACF peak)`` (valid in the absence of self-noise).
  - emission-region size from m + screen resolution (Nimmo Eqs 21-23).
"""

import numpy as np
import scipy.constants as cons
from lmfit import Model

from .noise import _robust_std


def _lorentz_w_c(x, gamma, m, c):
    """Pleunis Eq 5.1 / Nimmo ``lorentz_w_c``: m²/(1+(x/γ)²)+C; γ = HWHM = Δν."""
    return m**2 / (1 + (x / gamma) ** 2) + c


def _double_lorentz_w_c(x, gamma1, m1, gamma2, m2, c):
    """Nimmo ``doublelorentz_w_c``: two coherent scales (wide + narrow) + C."""
    return m1**2 / (1 + (x / gamma1) ** 2) + m2**2 / (1 + (x / gamma2) ** 2) + c


def rfi_flag(spec, n_sigma=5.0):
    """Boolean per-channel RFI mask (True = flagged) via robust-σ (MAD) outliers.

    Mirrors Nimmo's ``|spec-mean|/std > 3`` channel flag (``make_scallop_model``),
    using a MAD-based σ so a few bright RFI channels do not inflate the threshold.
    """
    spec = np.asarray(spec, dtype=float)
    med = float(np.nanmedian(spec))
    sig = float(_robust_std(spec))
    if not np.isfinite(sig) or sig == 0:
        return np.zeros(spec.shape, dtype=bool)
    return np.abs(spec - med) > n_sigma * sig


def off_pulse_mask(prof, k=3.0):
    """Boolean off-pulse mask (True = off-pulse) on a 1D time profile.

    A bin is off-pulse if it is within ``k`` robust-σ of the median (i.e. NOT part
    of the burst). Used to source the noise/baseline statistics for the ACF.
    """
    prof = np.asarray(prof, dtype=float)
    med = float(np.nanmedian(prof))
    sig = float(_robust_std(prof))
    if not np.isfinite(sig) or sig == 0:
        # ponytail: MAD=0 on a noiseless flat baseline -> fall back to std so the
        # burst (a clear outlier) is still excluded; upgrade to an iterative
        # off-pulse σ only if real baselines ever trip this.
        sig = float(np.nanstd(prof))
    if not np.isfinite(sig) or sig == 0:
        return np.ones(prof.shape, dtype=bool)
    return (prof - med) <= k * sig


def _acf_masked(x, keep, denom, maxlag):
    """Mean-subtracted, mask-aware ACF normalized by ``denom``, lags k=1..maxlag
    (zero-lag excluded). ``x`` is the spectrum already mean-subtracted on kept
    channels; ``keep`` is the 0/1 keep-mask. Mathematically equivalent to Nimmo et
    al. 2025 ``autocorr`` (the 3N-shift masked-overlap form), without its
    plotting-index bookkeeping. Returns ``acf[k-1]`` = correlation at lag k.
    """
    n = len(x)
    maxlag = int(min(maxlag, n - 1))
    out = np.zeros(maxlag)
    for k in range(1, maxlag + 1):
        m = keep[: n - k] * keep[k:]
        sm = m.sum()
        if sm > 0:
            out[k - 1] = np.nansum(x[: n - k] * x[k:] * m) / (sm * denom)
    return out


def _mean_normalized_acf(
    spec, keep, channel_width_mhz, max_lag_mhz=None, first_lag=1, offspec_mean=None
):
    """Two-sided mean-normalized ACF + lags in MHz, low lags omitted.

    Lag 0 is always absent (the self-noise spike). ``first_lag`` is the first lag
    RETAINED: the telescope-agnostic default ``first_lag=1`` drops only lag 0;
    ``first_lag=2`` reproduces Nimmo et al. 2025's CHIME-upchannelized treatment,
    which also drops lag 1 (adjacent fine channels share FFT/upchannelization noise)
    — that exclusion is an upchannelization artifact, not universal, so it is opt-in
    here. ``offspec_mean``, when given, sets Nimmo's ``(xmean - offspec_mean)²``
    denominator (otherwise ``xmean²``).

    Returns ``(lags_mhz, acf, peak)``: lags/acf mirrored about (absent) lag 0;
    ``peak`` is the first RETAINED ACF lag — a self-noise-free seed for the fit
    amplitude (the reported modulation index is the FITTED amplitude, not this value).
    """
    spec = np.asarray(spec, dtype=float)
    keep = np.asarray(keep, dtype=float)
    n = len(spec)
    kept = keep != 0
    xmean = float(np.nanmean(spec[kept])) if kept.any() else float(np.nanmean(spec))
    denom = (xmean - offspec_mean) ** 2 if offspec_mean is not None else xmean**2
    if not np.isfinite(denom) or denom == 0:
        denom = 1.0
    x = np.zeros(n)
    x[kept] = spec[kept] - xmean  # masked channels contribute 0 via `keep`

    band = n * channel_width_mhz
    if max_lag_mhz is None:
        max_lag_mhz = 0.25 * band
    maxlag = max(2, int(max_lag_mhz / channel_width_mhz))
    acf_all = _acf_masked(x, keep, denom, maxlag)  # lags 1..maxlag
    drop = max(0, int(first_lag) - 1)  # extra leading lags to omit (lag 0 already gone)
    acf_pos = acf_all[drop:]
    lags_pos = np.arange(1 + drop, len(acf_all) + 1) * channel_width_mhz
    lags = np.concatenate((-lags_pos[::-1], lags_pos))
    acf = np.concatenate((acf_pos[::-1], acf_pos))
    peak = float(acf_pos[0]) if len(acf_pos) else np.nan
    return lags, acf, peak


def _hwhm_init(acf_pos, channel_width_mhz):
    """Data-driven HWHM (MHz) init: first lag where the one-sided ACF drops below
    half its lag-1 value (falls back to a quarter of the lag span)."""
    if not len(acf_pos):
        return channel_width_mhz
    half = acf_pos[0] / 2.0
    below = np.where(acf_pos < half)[0]
    hwhm_bin = (below[0] + 1) if len(below) else max(1, len(acf_pos) // 4)
    return hwhm_bin * channel_width_mhz


def revalidate_dnu(
    spec, channel_width_mhz, max_lag_mhz=None, rfi_n_sigma=5.0, first_lag=1, offspec_mean=None
):
    """Single-screen Δν (MHz) = HWHM (γ) of the Nimmo/Pleunis Lorentzian fit to the
    mean-normalized, low-lag-excluded ACF of an RFI-flagged spectrum (Eq 5.1).

    ``first_lag=2`` reproduces Nimmo's CHIME-upchannelized treatment (drop lag 1 in
    addition to the lag-0 self-noise spike); the telescope-agnostic default drops
    only lag 0. ``offspec_mean`` (e.g. the off-pulse spectrum mean located via
    ``off_pulse_mask``) sets Nimmo's ``(xmean-offspec_mean)²`` normalization. Returns
    ``nan`` if the fit does not converge.
    """
    spec = np.asarray(spec, dtype=float)
    keep = (~rfi_flag(spec, n_sigma=rfi_n_sigma)).astype(float)
    lags, acf, peak = _mean_normalized_acf(
        spec, keep, channel_width_mhz, max_lag_mhz, first_lag, offspec_mean
    )
    gamma_init = _hwhm_init(acf[len(acf) // 2 :], channel_width_mhz)
    m_init = float(np.sqrt(max(peak, 1e-3)))

    model = Model(_lorentz_w_c)
    model.set_param_hint("gamma", min=channel_width_mhz / 10.0)
    model.set_param_hint("m", min=0.0)
    result = model.fit(acf, x=lags, gamma=gamma_init, m=m_init, c=0.0)
    if not result.success:
        return float("nan")
    return abs(float(result.params["gamma"].value))


def fit_two_screen_acf(
    spec, channel_width_mhz, max_lag_mhz=None, rfi_n_sigma=5.0, first_lag=1, offspec_mean=None
):
    """Two-screen (MW wide + host narrow) double-Lorentzian fit, center omitted.

    The new Nimmo/Pleunis capability: fit ``m1²/(1+(δν/γ1)²)+m2²/(1+(δν/γ2)²)+C`` to
    the mean-normalized ACF with the lag-0 self-noise spike excluded (the ACF starts
    at lag ≥ 1, so the contaminated center is omitted by construction; ``first_lag=2``
    additionally drops lag 1 per Nimmo's CHIME-upchannelized treatment).

    Returns the wide and narrow Δν (=γ) and their FITTED modulation amplitudes
    ``m_wide``/``m_narrow``, the combined modulation index
    ``m_total = sqrt(m_wide²+m_narrow²)`` — the fitted ACF-peak amplitude above the
    constant C, NOT the observed lag-1 value — and ``center_omitted``. The amplitudes
    are ``nan`` if the fit fails.
    """
    spec = np.asarray(spec, dtype=float)
    keep = (~rfi_flag(spec, n_sigma=rfi_n_sigma)).astype(float)
    lags, acf, peak = _mean_normalized_acf(
        spec, keep, channel_width_mhz, max_lag_mhz, first_lag, offspec_mean
    )
    acf_pos = acf[len(acf) // 2 :]
    span = len(acf_pos) * channel_width_mhz
    narrow_init = _hwhm_init(acf_pos, channel_width_mhz)
    wide_init = max(narrow_init * 8.0, 0.4 * span)  # seed the two scales apart
    m_init = float(np.sqrt(max(peak, 1e-3) / 2.0))

    model = Model(_double_lorentz_w_c)
    model.set_param_hint("gamma1", min=channel_width_mhz / 10.0)
    model.set_param_hint("gamma2", min=channel_width_mhz / 10.0)
    model.set_param_hint("m1", min=0.0)
    model.set_param_hint("m2", min=0.0)
    result = model.fit(
        acf, x=lags, gamma1=wide_init, m1=m_init, gamma2=narrow_init, m2=m_init, c=0.0
    )
    if not result.success:
        return {
            "dnu_wide_mhz": float("nan"),
            "dnu_narrow_mhz": float("nan"),
            "m_wide": float("nan"),
            "m_narrow": float("nan"),
            "m_total": float("nan"),
            "center_omitted": True,
        }
    g1 = abs(float(result.params["gamma1"].value))
    g2 = abs(float(result.params["gamma2"].value))
    m1 = abs(float(result.params["m1"].value))
    m2 = abs(float(result.params["m2"].value))
    # Order by scale: wide = larger Δν, narrow = smaller.
    (dnu_wide, m_wide), (dnu_narrow, m_narrow) = (
        ((g1, m1), (g2, m2)) if g1 >= g2 else ((g2, m2), (g1, m1))
    )
    return {
        "dnu_wide_mhz": dnu_wide,
        "dnu_narrow_mhz": dnu_narrow,
        "m_wide": m_wide,
        "m_narrow": m_narrow,
        "m_total": float(np.sqrt(m_wide**2 + m_narrow**2)),
        "center_omitted": True,
    }


def res(lens_dist_kpc, lda_m, scat_lens_ms):
    """Physical resolution of a scattering screen, in km. Port of Nimmo et al. 2025
    ``res``. ``lens_dist_kpc`` = source↔lens distance (kpc), ``lda_m`` = wavelength
    (m), ``scat_lens_ms`` = scattering time imparted by the screen (ms)."""
    lens_dist_m = lens_dist_kpc * cons.parsec * 1000
    scat_lens_s = scat_lens_ms / 1000.0
    return ((lda_m / np.pi) * np.sqrt(lens_dist_m / (4 * cons.c * scat_lens_s))) / 1000


def emission_size(phys_res_km, mod_ind):
    """Physical emission size (km) from the screen resolution + modulation index.
    Port of Nimmo et al. 2025 ``emission_size`` (Eqs 22-23): σ=√((1/m²-1)/4)."""
    sigma = np.sqrt((1 / (float(mod_ind) ** 2) - 1) / 4.0)
    return sigma * phys_res_km
