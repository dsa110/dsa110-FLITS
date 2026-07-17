"""Objective on-/off-pulse time-window selection for CHIME per-subband scintillation.

Replaces hand-slider window tuning with one deterministic rule applied uniformly to all
bursts (owner requirement: same estimation methodology sample-wide, no per-burst method
cherry-picking; morphology differences become a per-burst *systematic*, not a method choice).

Why not a simple band-mean threshold: the raw band-mean profile is dominated by bandpass
structure and edge artifacts (upchannelization zero-pad tails, wrap spikes), and a broad
faint burst (oran: ~150 bins) never clears a per-bin threshold that a narrow bright one
(chromatica: ~14 bins) trivially does. Verified on the 2026-07-17 preview figures: the
band-mean rule locked onto pad-boundary spikes for most bursts.

Rule:
  1. Validity span: drop time bins where most channels are masked/zero (zero-pad tails,
     dead edges), trim the residual edge bins.
  2. Per-channel robust standardization ((x - median_t)/MAD_t) -> channel-summed S/N
     profile (unit-variance per bin under noise; bright channels can no longer dominate).
  3. Boxcar matched-filter bank over log-spaced widths: the (width, center) with maximum
     matched S/N sets the burst scale — the same detection rule for 10-bin and 300-bin
     bursts (standard single-pulse-search practice).
  4. Merge satellite components above k_sat*sigma (at the matched smoothing scale) within
     a width-scaled gap; expand edges down to k_edge*sigma to keep scattering-tail fluence.
  5. Off-pulse = longest contiguous valid stretch outside on-pulse +/- width-scaled guard
     (feeds de-scalloping + RFI statistics; window_refit fails loudly on overlap).
"""
from __future__ import annotations
import numpy as np

_TINY = 1e-12


def valid_span(power, min_frac=0.5, trim=2):
    """(t0, t1) half-open span where >= min_frac of channels are unmasked and nonzero."""
    data = np.asarray(power.data if np.ma.isMaskedArray(power) else power, float)
    bad = ~np.isfinite(data) | (data == 0.0)
    if np.ma.isMaskedArray(power) and power.mask is not np.ma.nomask:
        bad |= power.mask
    frac_ok = 1.0 - bad.mean(axis=0)
    ok = frac_ok >= min_frac
    if not ok.any():
        return None
    runs = _runs_above(ok)
    t0, t1 = max(runs, key=lambda r: r[1] - r[0])
    return (int(t0) + trim, int(t1) - trim) if t1 - t0 > 2 * trim else (int(t0), int(t1))


def snr_profile(power):
    """Channel-summed standardized time profile: ~N(0,1) per bin under pure noise."""
    p = power if np.ma.isMaskedArray(power) else np.ma.MaskedArray(power)
    med = np.ma.median(p, axis=1, keepdims=True)
    mad = np.ma.median(np.abs(p - med), axis=1, keepdims=True) * 1.4826
    z = (p - med) / np.ma.maximum(mad, _TINY)
    nvalid = np.maximum((~np.ma.getmaskarray(z)).sum(axis=0), 1)
    return np.ma.filled(z.sum(axis=0), 0.0) / np.sqrt(nvalid)


def _runs_above(mask):
    """[(start, end)] half-open index runs where boolean mask is True."""
    m = np.asarray(mask, bool)
    idx = np.flatnonzero(np.diff(np.r_[0, m.view(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


def _box_smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def matched_peak(prof, max_width=None):
    """(center, width, snr) of the maximum boxcar matched-filter response.

    Matched S/N of a width-w boxcar on a unit-variance profile is sum(w bins)/sqrt(w);
    scanning log-spaced widths finds narrow and broad bursts with the same statistic.
    """
    n = prof.size
    if max_width is None:
        max_width = max(4, n // 3)
    widths, w = [], 1
    while w <= max_width:
        widths.append(w)
        w *= 2
    best = (0, 1, -np.inf)
    c = np.cumsum(np.r_[0.0, prof])
    for w in widths:
        s = (c[w:] - c[:-w]) / np.sqrt(w)          # matched S/N at each start
        i = int(np.argmax(s))
        if s[i] > best[2]:
            best = (i + w // 2, w, float(s[i]))
    return best


def _mad_stats(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, (mad if mad > 0 else float(np.std(x)) or 1.0)


def _detrend(x, width):
    """Subtract a running-median baseline (slow gain/background drift, not the burst)."""
    from scipy.ndimage import median_filter
    w = max(3, int(width) | 1)
    return x - median_filter(x, size=w, mode="nearest")


OFF_SNR_MAX = 3.0   # off-window purity threshold: a candidate off run whose matched-filter
                    # response at the burst scale exceeds this carries residual burst/tail
                    # power and is rejected as a de-scalloping/RFI reference (see gate below)


def select_windows(profile, k_sat=3.0, k_edge=0.75, guard_frac=1.0, min_snr=5.0):
    """Deterministic (burst_lims, off_lims) from a standardized S/N profile.

    Coordinates are in the profile's own frame; callers slicing to a validity span must
    offset the result. Returns None when no matched-filter peak clears min_snr — callers
    fall back to the pipeline's per-burst default windows and must FLAG that fallback.
    """
    raw = np.asarray(profile, float)
    n = raw.size
    # Baseline drift is the dominant failure mode (verified on the 2026-07-17 previews:
    # johndoeII's real spike drowned in a wandering baseline; phineas/zach edge-expanded
    # into drift): remove a slow running-median baseline first. The detrend window must
    # stay much wider than the burst — start at span/4, then widen to 4x the matched
    # width if the first pass finds a broad burst (tiny captures can't be detrended
    # without eating the burst; those fall through to the caller's fallback).
    det_w = max(9, n // 4)
    d = _detrend(raw, det_w)
    med, sig = _mad_stats(d)
    prof = (d - med) / sig
    center, width, snr = matched_peak(prof)
    if width > det_w // 4:
        det_w = min(n // 2, 4 * width)
        if det_w <= width:            # capture too small to separate burst from baseline
            return None
        d = _detrend(raw, det_w)
        med, sig = _mad_stats(d)
        prof = (d - med) / sig
        center, width, snr = matched_peak(prof)
    # re-estimate noise with the burst region excluded (burst inflates whole-span MAD)
    keep = np.ones(n, bool)
    keep[max(0, center - 2 * width):min(n, center + 2 * width)] = False
    if keep.sum() >= 16:
        med, sig = _mad_stats(d[keep])
        prof = (d - med) / sig
        center, width, snr = matched_peak(prof)
    if snr < min_snr:
        return None

    # smooth at the matched scale so per-bin thresholds see burst structure, not noise;
    # take the smoothed noise level from the data (correlated bins break 1/sqrt(w))
    w_s = max(3, width // 2) | 1
    s = _box_smooth(prof, w_s)
    mu, sig = _mad_stats(s[keep]) if keep.sum() >= 16 else _mad_stats(s)

    # Burst CORE = the k_sat-significance envelope around the matched peak, not
    # center +/- width/2: scattered profiles are asymmetric (fast rise, exponential
    # tail), so a symmetric matched window clips to 2-4 bins on narrow bursts and
    # throws away most of the burst spectrum S/N (batch-1 regression, 2026-07-17).
    sat = _runs_above(s > mu + k_sat * sig)
    main = next((r for r in sat if r[0] <= center < r[1]), None)
    if main is None and sat:
        main = min(sat, key=lambda r: min(abs(r[0] - center), abs(r[1] - center)))
    lo, hi = main if main else (max(0, center - width // 2),
                                min(n, center + (width + 1) // 2))
    # satellite components merged within a width-scaled gap (multi-component bursts
    # are one window, not the peak component)
    for r in sat:
        if 0 < r[0] - hi <= width:
            hi = r[1]
        elif 0 < lo - r[1] <= width:
            lo = r[0]

    core = (int(lo), int(hi))                          # significance envelope, no tail

    floor = mu + k_edge * sig                          # scattering-tail edge expansion
    while lo > 0 and s[lo - 1] > floor:
        lo -= 1
    while hi < n and s[hi] > floor:
        hi += 1
    burst = (int(lo), int(hi))
    # ACF-vs-fluence window distinction (2026-07-17 chromatica 2x2 isolation): including
    # the scattering tail inflates fitted gamma 2-4x in every subband and raises the ACF
    # baseline — low-S/N tail bins dilute scintle contrast so the fit latches onto the
    # broad intrinsic-envelope structure. The matched CORE is the ACF window; the
    # tail-expanded window is kept for fluence work and as a scan variant that measures
    # this systematic. Off-window choice moved gamma by <5% in the same test.

    guard = max(5, int(round(guard_frac * (hi - lo))))
    free = np.ones(n, bool)
    free[max(0, lo - guard):min(n, hi + guard)] = False
    off_runs = _runs_above(free)
    if not off_runs:
        return None

    # Off-window purity gate (ENFORCED). The off range feeds de-scalloping + RFI
    # statistics, so residual burst power there corrupts everything downstream. Two
    # failure modes seen 2026-07-17: (a) oran — the default off window rides the rising
    # burst envelope; (b) chromatica/zach — the LARGEST free run is the post-burst region
    # carrying the scattering tail (off_snr 7-11), and picking it purely by length
    # collapsed the recovered alpha (chromatica +1.87 -> -0.05). Previously off_snr was
    # computed only for the size-max run and never acted on. Now: score every candidate
    # run at the burst scale on the standardized profile and take the LARGEST run whose
    # off_snr clears OFF_SNR_MAX; fall back to the least-contaminated run only if none do.
    def _osnr(r):
        op = prof[r[0]:r[1]]
        return float(matched_peak(op, max_width=min(width * 2, max(4, op.size // 2)))[2]) \
            if op.size >= 8 else np.inf
    scored = [(r, r[1] - r[0], _osnr(r)) for r in off_runs if r[1] - r[0] >= 8]
    if not scored:                                   # no run long enough to characterize
        off = max(off_runs, key=lambda r: r[1] - r[0])
        off_snr = _osnr(off)
    else:
        clean = [t for t in scored if t[2] <= OFF_SNR_MAX]
        if clean:
            off, _, off_snr = max(clean, key=lambda t: t[1])
        else:                                        # all contaminated: least-bad wins
            off, _, off_snr = min(scored, key=lambda t: t[2])
    # Matched (profile-proportional) time weights: the injection harness (round 2,
    # 2026-07-17) found weighting the burst spectrum by the time profile is the least
    # biased gamma estimator (x1.05-1.18 of truth, best resolved rate) — it keeps the
    # scattering tail's S/N without letting its diluted scintle contrast dominate.
    # Data-driven analogue of the harness's noiseless-profile weights: the baseline-
    # subtracted smoothed profile, clipped at zero, zeroed outside the tail-expanded
    # window (noise bins get exactly zero weight instead of noisy small weights).
    wts = np.clip(s - mu, 0.0, None)
    wts[:burst[0]] = 0.0
    wts[burst[1]:] = 0.0
    tot = wts.sum()
    weights = (wts / tot) if tot > 0 else None
    return dict(burst_lims=[burst[0], burst[1]], burst_core=[core[0], core[1]],
                off_lims=[int(off[0]), int(off[1])], weights=weights,
                matched=dict(center=int(center), width=int(width), snr=float(snr)),
                off_snr=off_snr, smoothed=s, sigma_smooth=sig, guard=guard,
                params=dict(k_sat=k_sat, k_edge=k_edge, guard_frac=guard_frac,
                            min_snr=min_snr))


# The scan grid is the *systematic-error* instrument: refitting gamma across these window
# variants measures how much the measurement depends on the (uniform) selection rule.
SCAN_GRID = [dict(k_sat=k, k_edge=e) for k in (2.0, 3.0, 4.0) for e in (0.5, 0.75, 1.5)]


def window_variants(profile, **base):
    """Deduplicated window choices across SCAN_GRID (base kwargs override per-variant)."""
    seen, out = set(), []
    for g in SCAN_GRID:
        r = select_windows(profile, **{**g, **base})
        if r is None:
            continue
        key = (tuple(r["burst_lims"]), tuple(r["off_lims"]))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
