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
from scint_analysis import chime_artifact_guards as guards

LAG_MAX = 5.0
N_SKIP = 1
COMB_SPACING_MHZ = 0.390625
COMB_HALFWIDTH_MHZ = 0.05
_MIN_OFF = 50
SIGMA_RFI = 5.0
M_PHYS = 1.2       # max physical modulation index admitted to the alpha power-law fit
ALPHA_BOUNDS = (1.5, 6.0)

_BASECFG = {}      # name -> base config dict (cheap; the npz load dominates and is cached by the OS)


def alpha_is_physical(alpha):
    return bool(alpha and ALPHA_BOUNDS[0] < alpha["alpha"] < ALPHA_BOUNDS[1])


def _base_config(name):
    if name in _BASECFG:
        return copy.deepcopy(_BASECFG[name])
    # "<burst>_hi" auto-derives from the standard config with the input swapped to the
    # _hi product (600-800 MHz, per-burst upchannelization: 0.76-24.4 kHz channels).
    # Deliberately NOT the hand-tuned casey/freya *_chime_hi.yaml files: those restrict
    # band/subbands per burst, and the campaign requires one uniform rule sample-wide.
    if name.endswith("_hi"):
        cfg = config_module.load_config(f"{R}/scintillation/configs/bursts/{name[:-3]}_chime.yaml")
        cfg["input_data_path"] = cfg["input_data_path"].replace("_chime.npz", "_chime_hi.npz")
        cfg["burst_id"] = name
    else:
        cfg = config_module.load_config(f"{R}/scintillation/configs/bursts/{name}_chime.yaml")
    _BASECFG[name] = cfg
    return copy.deepcopy(cfg)


def product_available(name):
    """Return whether the configured local burst product exists."""
    path = os.path.expandvars(os.path.expanduser(_base_config(name)["input_data_path"]))
    return os.path.exists(path)


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


def _lorentz2(l, A_n, g_n, A_b, g_b, c0):
    """Narrow scintle + broad envelope: two Lorentzians sharing one baseline. The CHIME
    ACFs superpose a narrow scintillation component on a broad intrinsic-envelope/
    scattering component (zach_hi 622 MHz: unfit narrow feature at lag<0.15 MHz under a
    ~5-MHz ramp, owner-identified 2026-07-17); a single Lorentzian latches onto whichever
    dominates the least-squares and the other is censored or folded into m."""
    return A_n / (1 + (l / g_n) ** 2) + A_b / (1 + (l / g_b) ** 2) + c0


DBIC_2COMP = 6.0   # M2 must beat M1 by this (Kass-Raftery "strong"); injection round 4
                   # measures the false-positive rate of exactly this threshold
SCALE_SEP = 4.0    # required gamma_b/gamma_n separation for the 2-comp decomposition
                   # to be identifiable rather than a degenerate split of one scale


def _fit_subband(lags, acf, *, excision_bins=0):
    lags = np.asarray(lags, float); acf = np.asarray(acf, float)
    # No renormalization: calculate_acf already divides by (mean_on - mean_off)^2, so the
    # ACF amplitude IS m^2. The old argmin|lag| anchor hit a synthetic lag-0=1.0 bin that
    # commit dad9786 removed; renormalizing by the first REAL lag (~0.6 for chromatica)
    # inflated every A and m by ~1.7x (diagnosed 2026-07-17, reproduced archive to the
    # digit with norm=1.0).
    a = acf
    pos = lags > 0
    lp = lags[pos]; ap = a[pos]
    o = np.argsort(lp); lp = lp[o]; ap = ap[o]
    sel = lp <= LAG_MAX
    skip = max(N_SKIP - 1, int(excision_bins))
    lp = lp[sel][skip:]; ap = ap[sel][skip:]
    keep = ana.harmonic_lag_mask(lp, COMB_SPACING_MHZ, COMB_HALFWIDTH_MHZ)
    lp = lp[keep]; ap = ap[keep]
    if lp.size < 8:
        return dict(ok=False, reason="too few lags")
    outer = lp > 0.5 * lp.max()
    noise = float(np.std(ap[outer])) if outer.sum() > 3 else float(np.std(ap))
    # clip the guess inside the fit bounds: weighted spectra of noise-dominated
    # subbands can put ap[0] above the A<=10 bound, which makes curve_fit raise
    # ("initial guess outside bounds") instead of returning a gated non-detection
    A0 = float(np.clip(ap[0] - np.median(ap[outer]), 1e-3, 9.9))
    # The narrow-gamma reach is set by the channel width (injection round 1), so the
    # gamma fit floor and the low-side rail must scale with the lag grid, not sit at
    # a fixed MHz value: a hardcoded 0.02 floor (tuned for 24.4 kHz products) rejected
    # hamilton_hi's genuinely resolvable gamma~0.014 at 6.1 kHz channels.
    dlag = float(np.median(np.diff(lp)))
    glo = max(0.25 * dlag, 1e-4)
    try:
        p, cov = curve_fit(lorentz, lp, ap, p0=[A0, max(0.5, 2 * glo), 0.0],
                           bounds=([0, glo, -1], [10, 20, 1]), maxfev=20000)
        perr = np.sqrt(np.diag(cov))
    except Exception as e:
        return dict(ok=False, reason=str(e), noise=noise)
    A, gamma, c0 = p; Aerr, gerr, _ = perr
    npts = lp.size
    model = lorentz(lp, *p)
    rss1 = float(np.sum((ap - model) ** 2))
    bic1 = npts * np.log(max(rss1, 1e-30) / npts) + 3 * np.log(npts)

    # Two-component candidate: narrow scintle + broad envelope (see _lorentz2). Multi-
    # start on the narrow scale — 2-comp Lorentzian fits are initialization-sensitive
    # and a single bad start would silently fall back to the censoring single-component
    # behavior this model exists to fix.
    best2 = None
    for gn0 in (max(4 * glo, 0.02), 0.1, 0.3):
        if gn0 >= LAG_MAX:
            continue
        p0 = [max(A0 / 2, 1e-3), gn0, max(A0 / 2, 1e-3), max(3.0, min(gamma, 19.0)), 0.0]
        try:
            p2, cov2 = curve_fit(_lorentz2, lp, ap, p0=p0,
                                 bounds=([0, glo, 0, glo, -1], [10, 20, 10, 20, 1]),
                                 maxfev=40000)
        except Exception:
            continue
        r2 = float(np.sum((ap - _lorentz2(lp, *p2)) ** 2))
        if best2 is None or r2 < best2[0]:
            best2 = (r2, p2, cov2)
    model_sel, gamma_b, gamma_b_err, A_b, dbic2 = "1L", None, None, None, None
    if best2 is not None:
        rss2, p2, cov2 = best2
        perr2 = np.sqrt(np.diag(cov2))
        if p2[1] > p2[3]:   # enforce gamma_n < gamma_b, permuting errors with params
            p2 = [p2[2], p2[3], p2[0], p2[1], p2[4]]
            perr2 = [perr2[2], perr2[3], perr2[0], perr2[1], perr2[4]]
        bic2 = npts * np.log(max(rss2, 1e-30) / npts) + 5 * np.log(npts)
        dbic2 = float(bic1 - bic2)
        # adopt only a decisively better AND identifiable decomposition
        if dbic2 >= DBIC_2COMP and p2[3] > SCALE_SEP * p2[1] and p2[0] > 0 and p2[2] > 0:
            model_sel = "2L"
            A, gamma, c0 = float(p2[0]), float(p2[1]), float(p2[4])
            Aerr, gerr = float(perr2[0]), float(perr2[1])
            A_b, gamma_b, gamma_b_err = float(p2[2]), float(p2[3]), float(perr2[3])
            model = _lorentz2(lp, *p2)
    m = float(np.sqrt(max(A, 0)))
    amp_snr = A / noise if noise > 0 else np.inf
    # A railed at its upper bound is as diagnostic as a railed gamma: weak-burst
    # weighted spectra can drive the ACF amplitude to the A<=10 bound (m=sqrt(10)
    # =3.16 exactly — johndoeII/mahi 739-MHz artifact, 2026-07-17), which is an
    # envelope/noise pathology, never a physical modulation index
    railed = (gamma < 2 * glo) or (gamma > 0.9 * 20) or (gamma > 0.9 * LAG_MAX) \
        or (A > 0.9 * 10)
    # Shape gate (single-component winners only): a smooth envelope decays quasi-
    # linearly across the fitted lags and can pass every amplitude/rail gate with a
    # physical m — require the Lorentzian to beat a 2-param line by dBIC >= 6 or the
    # scale is not constrained within LAG_MAX. When the two-component model wins, the
    # narrow component must also beat the simpler line model. Beating one nonlinear
    # alternative does not establish that the winning shape is physically useful.
    rss_sel = float(np.sum((ap - model) ** 2))
    rss_lin = float(np.sum((ap - np.polyval(np.polyfit(lp, ap, 1), lp)) ** 2))
    k_sel = 5 if model_sel == "2L" else 3
    dbic = (npts * np.log(max(rss_lin, 1e-30) / npts) + 2 * np.log(npts)) \
        - (npts * np.log(max(rss_sel, 1e-30) / npts) + k_sel * np.log(npts))
    shape_ok = bool(dbic >= 6.0)
    resolved = bool((amp_snr > 3) and (not railed) and (gamma > 2 * dlag)
                    and (gerr < gamma) and shape_ok)
    return dict(ok=True, A=float(A), gamma=float(gamma), gamma_err=float(gerr), c0=float(c0),
                m=m, noise=noise, amp_snr=float(amp_snr), resolved=resolved,
                shape_ok=shape_ok, dbic_line=float(dbic), model_sel=model_sel,
                A_b=A_b, gamma_b=gamma_b, gamma_b_err=gamma_b_err, dbic_2comp=dbic2,
                lp=lp, ap=ap, model=model)


def summarize_artifact_controls(
    *, on_dnu_mhz, off_dnu_mhz, excision_widths, n_valid_subbands
):
    """Apply the repository CHIME guards and fail closed on any non-pass verdict."""
    null = guards.off_pulse_null_verdict(on_dnu_mhz, off_dnu_mhz)
    stability = guards.low_lag_stability_verdict(on_dnu_mhz, excision_widths)
    support = guards.subband_support_verdict(n_valid_subbands)
    checks = {
        "off_pulse_null": null.get("null_pass"),
        "low_lag_stability": stability.get("stable"),
        "subband_support": support.get("sufficient"),
    }
    failed = [name for name, value in checks.items() if value is not True]
    return {
        "status": "pass" if not failed else "fail",
        "failed_checks": failed,
        "off_pulse_null": null,
        "low_lag_stability": stability,
        "subband_support": support,
    }


def _artifact_controls(spec, res, fits, order, burst, off):
    """Run the established CHIME controls on the campaign's reference subband."""
    centers = np.asarray(res["subband_center_freqs_mhz"], float)
    ref_index = int(np.nanargmin(np.abs(centers - np.nanmedian(centers))))
    ref_fit = fits[ref_index]
    on_dnu = float(ref_fit["gamma"]) if ref_fit.get("ok") else None
    lags = res["subband_lags_mhz"][ref_index]
    acf = res["subband_acfs"][ref_index]
    excision_widths = {}
    for k in (1, 2, 3):
        fit = _fit_subband(lags, acf, excision_bins=k)
        excision_widths[k] = float(fit["gamma"]) if fit.get("ok") else None

    channel_slice = tuple(res["subband_channel_slices"][ref_index])
    channel_width = float(res["subband_channel_widths_mhz"][ref_index])
    width = max(int(burst[1] - burst[0]), 4)
    starts = list(range(int(off[0]) + 2, int(off[1]) - width, width + 4))[:6]
    off_widths = []
    max_lag_bins = int(LAG_MAX / channel_width) if channel_width > 0 else None
    for start in starts:
        try:
            spectrum = spec.get_spectrum((start, start + width))[channel_slice[0]:channel_slice[1]]
            off_acf = ana.calculate_acf(
                spectrum,
                channel_width,
                off_burst_spectrum_mean=None,
                max_lag_bins=max_lag_bins,
            )
            fit = _fit_subband(off_acf.lags, off_acf.acf)
        except Exception:
            continue
        if fit.get("ok"):
            off_widths.append(float(fit["gamma"]))

    valid = sum(
        1
        for fit in fits.values()
        if fit.get("ok")
        and fit.get("resolved")
        and fit.get("shape_ok")
        and np.isfinite(fit.get("m", np.nan))
        and fit["m"] <= M_PHYS
    )
    summary = summarize_artifact_controls(
        on_dnu_mhz=on_dnu,
        off_dnu_mhz=off_widths,
        excision_widths=excision_widths,
        n_valid_subbands=valid,
    )
    summary.update(
        {
            "reference_subband_index": ref_index,
            "reference_frequency_mhz": float(centers[ref_index]),
            "off_pulse_slice_starts": starts,
        }
    )
    return summary


def refit(name, burst_lims, off_lims, rfi_bands_mhz=None, first_fit_lag=1,
          time_weights=None, subband_channel_slices=None, validate_artifacts=False,
          off_pulse_null=False):
    """first_fit_lag=1 keeps the lag-1 bin, which carries most of the constraint for
    gamma near the channel width (FFL=2 in the drifted configs railed chromatica's
    517 MHz subband; FFL=1 reproduces the archived resolved fit). Uniform for all
    bursts; the 1-vs-2 bias is under injection-harness validation.

    time_weights (full-time-length array, optional): profile-proportional weights for
    the burst-spectrum extraction (matched estimator — see core.get_spectrum). When
    given, burst_lims should be the tail-expanded burst extent so de-scalloping and
    RFI statistics exclude the whole burst; the weights handle tail down-weighting."""
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
    c["analysis"]["acf"]["first_fit_lag"] = int(first_fit_lag)
    if subband_channel_slices is not None:
        c["analysis"]["acf"]["subband_channel_slices"] = [
            [int(start), int(end)] for start, end in subband_channel_slices
        ]
    if time_weights is not None:
        c["analysis"]["acf"]["time_weights"] = np.asarray(time_weights, float)
        method += " + matched time-weighting"
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

    # Per-subband off-pulse ACF null (experiment arm A). Run the IDENTICAL subband machinery
    # on burst-free noise slices inside the off-pulse window (same channels, same subbanding,
    # same fitter) and compare, subband-by-subband, against the on-pulse gamma. A real
    # scintillation scale lives only in the burst; if the off-pulse noise reproduces the
    # on-pulse gamma (within off_pulse_null_verdict's bracket), that subband's scale is
    # instrumental. This is the reviewer-approved (R5) stronger control: it tests exactly the
    # subbands that enter the alpha fit, not the single reference subband of _artifact_controls.
    null_by_sub = {}
    if off_pulse_null:
        on_dur = max(1, burst[1] - burst[0])
        o0, o1 = off
        starts = list(range(o0, max(o0, o1 - on_dur) + 1, on_dur))[:8]   # cap 8 tiles
        off_widths = {int(i): [] for i in order}
        for s0 in starts:
            slc = (s0, s0 + on_dur)
            try:
                ro = ana.calculate_acfs_for_subbands(spec, c, burst_lims=slc, noise_desc=None)
            except Exception:
                continue
            cfo = np.asarray(ro.get("subband_center_freqs_mhz", []), float)
            if cfo.size == 0:
                continue
            for i in order:                       # match off subbands to on by nearest center
                j = int(np.argmin(np.abs(cfo - cf[i])))
                if abs(cfo[j] - cf[i]) > 30.0:     # no comparable subband on this noise slice
                    continue
                fo = _fit_subband(ro["subband_lags_mhz"][j], ro["subband_acfs"][j])
                if fo.get("ok") and fo.get("resolved"):
                    off_widths[int(i)].append(float(fo["gamma"]))
        for i in order:
            f = fits[int(i)]
            on_g = f["gamma"] if (f.get("ok") and f.get("resolved")) else None
            null_by_sub[int(i)] = guards.off_pulse_null_verdict(on_g, off_widths[int(i)])
    # Finite-scintle error: with N_ISS ~ 1 + eta*BW/gamma independent scintles per
    # subband (eta=0.2, Cordes & Lazio estimator-filling convention), the fractional
    # gamma uncertainty from sampling a finite number of scintles is 1/sqrt(N_ISS) —
    # irreducible at fixed bandwidth, and the dominant term for broad gamma
    # (injection round 1: scatter grows to 20-33% by gamma=3 MHz for exactly this
    # reason). Derived from (gamma, BW) only; does not feed back into the fit.
    for i in order:
        f = fits[int(i)]
        if f.get("ok"):
            bw = float(res["subband_num_channels"][i]) * float(res["subband_channel_widths_mhz"][i])
            n_iss = 1.0 + 0.2 * bw / max(f["gamma"], 1e-6)
            f["subband_bw_mhz"] = bw
            f["gamma_scintle_err"] = float(f["gamma"] / np.sqrt(n_iss))
    # The alpha fit additionally requires a physical modulation index: m>M_PHYS
    # passes the resolved gate but is envelope-contaminated (point-source strong
    # scintillation has m<=1; the margin absorbs self-/finite-scintle noise), and
    # one contaminated subband can swing a 3-4 point slope wildly (hamilton_hi's
    # flagged 751-MHz band produced alpha=+33). The per-subband fit is still
    # reported — only the power-law selection excludes it.
    resolved = [(cf[i], fits[int(i)]["gamma"], fits[int(i)]["gamma_err"],
                 fits[int(i)]["gamma_scintle_err"]) for i in order
                if fits[int(i)]["ok"] and fits[int(i)]["resolved"]
                and fits[int(i)]["m"] <= M_PHYS]
    alpha = None
    if len(resolved) >= 2:
        fr = np.array([r[0] for r in resolved]); gm = np.array([r[1] for r in resolved])
        ge = np.array([np.hypot(r[2], r[3]) for r in resolved])
        lw = 1.0 / (ge / gm) ** 2
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
    artifact_controls = (
        _artifact_controls(spec, res, fits, order, burst, off) if validate_artifacts else None
    )
    return dict(name=name, burst=burst, off=off, method=method, center_freqs=cf, order=list(order),
                fits=fits, alpha=alpha, rfi_new=int((flag & ~already).sum()),
                rfi_total=int(flag.sum()), ntime=spec.power.shape[1], nchan=spec.power.shape[0],
                subband_channel_slices=res["subband_channel_slices"],
                artifact_controls=artifact_controls, off_pulse_null=null_by_sub)
