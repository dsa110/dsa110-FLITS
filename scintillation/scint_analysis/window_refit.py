"""Interactive-refit backend for the CHIME per-subband scintillation notebook.

load_raw(name)  -> caches the grid-regularized raw spectrum (de-scalloping OFF) once per burst.
refit(name, burst_lims, off_lims, rfi_bands_mhz) -> applies de-scallop + auto-RFI + equal-S/N
    4-subband ACF + per-subband Lorentzian fit (harmonic-masked) and returns a results dict.
    Only the cheap stage re-runs on a window change; the multi-hundred-MB npz load is cached.
"""
from __future__ import annotations
import os, sys, copy
import numpy as np
from scipy.optimize import curve_fit

# Resolve the repo root from this file's location: scint_analysis/ -> scintillation/ -> <root>.
# FLITS_ROOT env var overrides if set (e.g. when run from an external checkout).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCINT_DIR = os.path.dirname(_HERE)                 # .../scintillation
R = os.environ.get("FLITS_ROOT", os.path.dirname(_SCINT_DIR))
if _SCINT_DIR not in sys.path:
    sys.path.insert(0, _SCINT_DIR)
if R not in sys.path:
    sys.path.insert(0, R)
from scint_analysis import freya_scintillation as fs
from scint_analysis import config as config_module
from scint_analysis import analysis as ana
from scint_analysis import auto_rfi_flag as arf

LAG_MAX = 5.0
N_SKIP = 1
COMB_SPACING_MHZ = 0.390625
COMB_HALFWIDTH_MHZ = 0.05
_MIN_OFF = 50
SIGMA_RFI = 5.0

_BASECFG = {}      # name -> base config dict (cheap; the npz load dominates and is cached by the OS)


def _base_config(name):
    if name in _BASECFG:
        return copy.deepcopy(_BASECFG[name])
    cfg = config_module.load_config(f"{R}/scintillation/configs/bursts/{name}_chime.yaml")
    _BASECFG[name] = cfg
    return copy.deepcopy(cfg)


def load_raw(name):
    """Kept for API compatibility; warms the config cache. The spectrum itself is rebuilt
    per window because the pipeline's RFI mask + de-scalloping both depend on the windows."""
    return _base_config(name)


def default_windows(name):
    """Return the pipeline's auto-determined (burst_lims, off_lims) for a burst, so the
    notebook can seed its sliders without a checked-in window_choices.json. Uses the same
    determine_windows path as prepare_spectrum_from_config, with de-scalloping OFF (windows
    do not depend on it)."""
    c = _base_config(name); an = c.setdefault("analysis", {})
    an.setdefault("bandpass_normalization", {})["enable"] = False
    an.setdefault("baseline_subtraction", {})["enable"] = False
    _spec, bl, ol = fs.prepare_spectrum_from_config(c)
    return [int(bl[0]), int(bl[1])], ([int(ol[0]), int(ol[1])] if ol else None)


def _build_spec(name, burst, off):
    """Mirror run_persubband_fits' spectrum build EXACTLY so the notebook and the batch
    driver produce identical numbers: manual windows -> full prepare_spectrum_from_config
    with bandpass de-scallop ON, except wide bursts (< _MIN_OFF off bins) fall back to a
    per-channel time-median flat-field."""
    c = _base_config(name); an = c.setdefault("analysis", {})
    an.setdefault("acf", {})["num_subbands"] = 4; an["acf"]["use_snr_subbanding"] = True
    an.setdefault("grid_regularization", {})["enable"] = True
    an.setdefault("bandpass_normalization", {})["enable"] = True
    an.setdefault("rfi_masking", {})["manual_burst_window"] = list(burst)
    an["rfi_masking"]["manual_noise_window"] = list(off)
    use_median = (off[1] - off[0]) < _MIN_OFF
    if use_median:
        c["analysis"]["bandpass_normalization"]["enable"] = False
    spec, bl, ol = fs.prepare_spectrum_from_config(c)
    if use_median:
        colmask = np.ones(spec.power.shape[1], bool); colmask[burst[0]:burst[1]] = False
        gain = np.ma.filled(np.ma.median(spec.power[:, colmask], axis=1), np.nan)
        med = np.nanmedian(gain[np.isfinite(gain) & (gain > 0)])
        bad = ~(np.isfinite(gain) & (gain > 1e-3 * med)); g = np.where(bad, 1.0, gain)
        m0 = spec.power.mask if spec.power.mask is not np.ma.nomask else np.zeros(spec.power.shape, bool)
        spec.power = np.ma.MaskedArray(spec.power.data / g[:, None], mask=m0 | bad[:, None])
    method = "time-median flat-field (off-pulse < 50 bins)" if use_median else \
             "off-pulse mean (bandpass_normalization)"
    return spec, c, method


# Reuse the pipeline's Lorentzian-with-baseline model rather than defining a fifth parallel
# copy (CLAUDE.md: no duplicate implementations). Signature is (lag, amplitude, gamma, baseline),
# matching the (l, A, gamma, c0) order curve_fit expects below.
lorentz = fs._lorentzian_with_baseline


def _fit_subband(lags, acf):
    lags = np.asarray(lags, float); acf = np.asarray(acf, float)
    i0 = int(np.argmin(np.abs(lags)))
    norm = acf[i0] if acf[i0] != 0 else np.nanmax(acf)
    a = acf / norm
    pos = lags > 0
    lp = lags[pos]; ap = a[pos]
    o = np.argsort(lp); lp = lp[o]; ap = ap[o]
    sel = lp <= LAG_MAX
    lp = lp[sel][N_SKIP - 1:]; ap = ap[sel][N_SKIP - 1:]
    keep = ana.harmonic_lag_mask(lp, COMB_SPACING_MHZ, COMB_HALFWIDTH_MHZ)
    lp = lp[keep]; ap = ap[keep]
    if lp.size < 8:
        return dict(ok=False, reason="too few lags")
    outer = lp > 0.5 * lp.max()
    noise = float(np.std(ap[outer])) if outer.sum() > 3 else float(np.std(ap))
    A0 = max(ap[0] - np.median(ap[outer]), 1e-3)
    try:
        p, cov = curve_fit(lorentz, lp, ap, p0=[A0, 0.5, 0.0],
                           bounds=([0, 0.01, -1], [10, 20, 1]), maxfev=20000)
        perr = np.sqrt(np.diag(cov))
    except Exception as e:
        return dict(ok=False, reason=str(e), noise=noise)
    A, gamma, c0 = p; Aerr, gerr, _ = perr
    m = float(np.sqrt(max(A, 0)))
    dlag = float(np.median(np.diff(lp)))
    amp_snr = A / noise if noise > 0 else np.inf
    railed = (gamma < 0.02) or (gamma > 0.9 * 20) or (gamma > 0.9 * LAG_MAX)
    resolved = bool((amp_snr > 3) and (not railed) and (gamma > 2 * dlag) and (gerr < gamma))
    return dict(ok=True, A=float(A), gamma=float(gamma), gamma_err=float(gerr), c0=float(c0),
                m=m, noise=noise, amp_snr=float(amp_snr), resolved=resolved,
                lp=lp, ap=ap, model=lorentz(lp, *p))


def refit(name, burst_lims, off_lims, rfi_bands_mhz=None):
    burst = (int(burst_lims[0]), int(burst_lims[1]))
    off = (int(off_lims[0]), int(off_lims[1]))
    # The off-pulse window feeds the off-pulse-only RFI statistics and the de-scallop gain.
    # If it overlaps the burst, burst/scintillation structure leaks into those statistics and
    # can be masked as RFI before the ACF fits. The interactive sliders are independent, so
    # this is easy to trigger by accident -- fail loudly rather than silently corrupt the fit.
    if off[1] > burst[0] and burst[1] > off[0]:
        raise ValueError(
            f"off-pulse window {off} overlaps the burst window {burst}; "
            "move the off-pulse slider clear of the on-pulse region")
    rfi_bands_mhz = rfi_bands_mhz or []
    spec, c, method = _build_spec(name, burst, off)
    freqs = np.asarray(spec.frequencies, float)
    # user painted RFI bands (whole-channel) on top of pipeline + auto flag
    band_mask = np.zeros(spec.power.shape[0], bool)
    for lo, hi in rfi_bands_mhz:
        band_mask |= (freqs >= min(lo, hi)) & (freqs <= max(lo, hi))
    flag, info = arf.auto_flag(spec.power, off, sigma=SIGMA_RFI, iters=6)
    m1 = spec.power.mask if spec.power.mask is not np.ma.nomask else np.zeros(spec.power.shape, bool)
    already = m1[:, off[0]:off[1]].all(axis=1)
    spec.power = np.ma.MaskedArray(spec.power.data, mask=m1 | flag[:, None] | band_mask[:, None])
    res = ana.calculate_acfs_for_subbands(spec, c, burst_lims=burst, noise_desc=None)
    cf = np.asarray(res["subband_center_freqs_mhz"], float)
    order = np.argsort(cf)[::-1]
    fits = {int(i): _fit_subband(res["subband_lags_mhz"][i], res["subband_acfs"][i]) for i in order}
    resolved = [(cf[i], fits[int(i)]["gamma"], fits[int(i)]["gamma_err"]) for i in order
                if fits[int(i)]["ok"] and fits[int(i)]["resolved"]]
    alpha = None
    if len(resolved) >= 2:
        fr = np.array([r[0] for r in resolved]); gm = np.array([r[1] for r in resolved])
        ge = np.array([r[2] for r in resolved]); lw = 1.0 / (ge / gm) ** 2
        Amat = np.vstack([np.log(fr / np.mean(fr)), np.ones_like(fr)]).T
        W = np.diag(lw); cov = np.linalg.inv(Amat.T @ W @ Amat)
        beta = cov @ (Amat.T @ W @ np.log(gm))
        # n==2 is a zero-degree-of-freedom power-law fit: the slope passes exactly through the
        # two points and the CHIME support guard (subband_support_verdict) treats it as
        # diagnostic-only. We still report it (matching the batch driver's >=2 threshold so the
        # tuning numbers mirror run_persubband_fits) but flag it so the display does not present
        # a two-point slope as a firmly measured alpha.
        alpha = dict(alpha=float(beta[0]), alpha_err=float(np.sqrt(cov[0, 0])), n=len(resolved),
                     provisional=(len(resolved) < 3))
    return dict(name=name, burst=burst, off=off, method=method, center_freqs=cf, order=list(order),
                fits=fits, alpha=alpha, rfi_new=int((flag & ~already).sum()),
                rfi_total=int(flag.sum()), ntime=spec.power.shape[1], nchan=spec.power.shape[0])
