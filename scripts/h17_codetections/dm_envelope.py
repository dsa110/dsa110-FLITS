"""Coherent trial-DM likelihood envelope for faint, scatter-broadened FRBs (telescope-agnostic).

Expert-endorsed method (.agents/audit-chime-side-dm.md, P5): the S/N-max / arrival-regression /
structure-max estimators all fail on smooth low-S/N bursts (scattering S/N-DM bias, no sub-structure,
zero-fill lever-arm loss). Instead COHERENTLY re-dedisperse the baseband at a grid of trial DMs (the
caller does this in the baseband docker image) and at each trial fit a scattering-aware 2-D template
with a SHARED arrival time. The chi^2(DM) envelope is the DM likelihood: min -> DM, Delta-chi^2=1
(rescaled) half-width -> honest sigma that goes WIDE (an exclusion interval, not a spurious tight
point) when the burst does not constrain DM.

Coherent dedispersion is a phase rotation, not a roll, so every trial keeps the FULL band (no zero-fill
triangle) -> the dispersive lever arm is preserved. DM enters only as the grid axis + a flat prior, so
the estimate is independent of the reference DM the comparison is against.

Pure numpy/scipy (no flits imports) so the SAME module runs in the baseband docker image and host.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erfc, erfcx

K_DM = 4.148808e3  # s MHz^2 pc^-1 cm^3 (== flits.common.constants.K_DM; inlined for docker)
_SQRT2 = np.sqrt(2.0)


def _emg_bank(t, t0, sigma, taus):
    """Unit exponentially-modified Gaussians, one per channel: shape (n_ch, n_t).

    Gaussian(sigma) convolved one-sided exp(-t/tau_ch); numerically stable piecewise (erfcx on the
    rising edge z>=0, direct exp*erfc on the tail z<0). ``taus`` is per-channel (n_ch,), ``t`` is (n_t,).
    """
    sig = abs(sigma) + 1e-12
    dt = t[None, :] - t0  # (1, n_t)
    tau = np.abs(taus)[:, None] + 1e-12  # (n_ch, 1)
    z = (sig / tau - dt / sig) / _SQRT2  # (n_ch, n_t)
    out = np.empty(z.shape, float)
    pos = z >= 0
    gauss = np.broadcast_to(np.exp(-0.5 * (dt / sig) ** 2), z.shape)
    out[pos] = gauss[pos] * erfcx(z[pos])
    a = np.clip(0.5 * (sig / tau) ** 2 - dt / tau, -700.0, 700.0)
    out[~pos] = np.exp(a[~pos]) * erfc(z[~pos])
    return out


def _chi2(params, I, t, freqs, w, dof):
    """Chi^2 of a shared-arrival scattering template with per-channel amp+base marginalised (closed form).

    params = (t0, log_sigma, log_tau1ghz); tau(nu) = tau1ghz * (nu/1000 MHz)^-4 (alpha fixed = 4).
    """
    t0, log_sig, log_tau = params
    sigma = np.exp(log_sig)
    tau1 = np.exp(log_tau)
    taus = tau1 * (freqs / 1000.0) ** -4.0
    T = _emg_bank(t, t0, sigma, taus)  # (n_ch, n_t)
    n = t.size
    ST = T.sum(1)
    STT = (T * T).sum(1)
    SI = I.sum(1)
    STI = (T * I).sum(1)
    SII = (I * I).sum(1)
    det = n * STT - ST**2 + 1e-30
    amp = (n * STI - ST * SI) / det
    base = (SI - amp * ST) / n
    rss = SII - amp * STI - base * SI  # per-channel residual sum of squares
    return float(np.sum(w * np.maximum(rss, 0.0)))  # chi^2 (w = 1/noise^2 per channel)


def _bin_freq(wf, freqs, nbin):
    """Block-average channels to ~nbin (the per-channel amp marginalisation needs sub-bands, not 800)."""
    nf = wf.shape[0]
    if nf <= nbin:
        return wf, freqs
    k = nf // nbin
    m = (nf // k) * k
    return wf[:m].reshape(m // k, k, wf.shape[1]).mean(1), freqs[:m].reshape(m // k, k).mean(1)


def fit_waterfall(wf, freqs, dt, nbin=192):
    """Fit the shared-arrival scattering template to one coherently-dedispersed waterfall (n_ch, n_t).

    Returns (chi2, dof, params, chi2_red). ``wf`` already masked/cropped; ``freqs`` [MHz]; ``dt`` [s].
    Channels are binned to ~``nbin`` sub-bands (speed; amp+base still marginalised per sub-band) and
    weighted 1/noise^2 from each sub-band's MAD over time.
    """
    wf, freqs = _bin_freq(np.nan_to_num(np.asarray(wf, float)), np.asarray(freqs, float), nbin)
    n_ch, n_t = wf.shape
    t = np.arange(n_t) * dt
    # per-channel noise: MAD over time (robust to the burst), weight 1/noise^2
    noise = 1.4826 * np.median(np.abs(wf - np.median(wf, axis=1, keepdims=True)), axis=1) + 1e-9
    w = 1.0 / noise**2
    prof = (wf - np.median(wf, axis=1, keepdims=True)).sum(0)
    pk = int(np.argmax(np.convolve(prof, np.ones(max(int(1e-3 / dt), 1)), "same")))
    t0_0 = t[pk]
    p0 = [t0_0, np.log(2e-3), np.log(1e-3)]  # ~2 ms width, ~1 ms tau guess
    dof = max(n_ch * n_t - 3 - 2 * n_ch, 1)
    best = None
    for log_tau0 in (np.log(3e-4), np.log(1e-3), np.log(3e-3)):  # multi-start over scattering scale
        try:
            r = minimize(
                _chi2,
                [t0_0, np.log(2e-3), log_tau0],
                args=(wf, t, freqs, w, dof),
                method="Nelder-Mead",
                options={"xatol": dt * 0.2, "fatol": 1.0, "maxiter": 2000},
            )
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        return (
            float(_chi2(p0, wf, t, freqs, w, dof)),
            dof,
            p0,
            _chi2(p0, wf, t, freqs, w, dof) / dof,
        )
    return float(best.fun), dof, list(best.x), float(best.fun / dof)


def fit_t0(wf, freqs, dt, sigma, tau1ghz, nbin=192):
    """Chi^2 of the shared-arrival template with the burst SHAPE fixed (sigma, tau1ghz) — only the
    shared arrival t0 is optimised (1-D). Used per trial DM in the fast envelope: the scattering/width
    are DM-independent, so they are fit once at the reference and frozen here. Returns (chi2, dof, t0).
    """
    wf, freqs = _bin_freq(np.nan_to_num(np.asarray(wf, float)), np.asarray(freqs, float), nbin)
    n_ch, n_t = wf.shape
    t = np.arange(n_t) * dt
    noise = 1.4826 * np.median(np.abs(wf - np.median(wf, axis=1, keepdims=True)), axis=1) + 1e-9
    w = 1.0 / noise**2
    ls, lt = np.log(abs(sigma) + 1e-12), np.log(abs(tau1ghz) + 1e-12)
    f = lambda t0: _chi2([t0, ls, lt], wf, t, freqs, w, 1)  # noqa: E731
    r = minimize_scalar(f, bounds=(t[0], t[-1]), method="bounded", options={"xatol": dt * 0.2})
    dof = max(n_ch * n_t - 1 - 2 * n_ch, 1)
    return float(r.fun), dof, float(r.x)


def _parabola_min(x, y):
    """Sub-grid minimum of a sampled curve: parabola through the 3 points around argmin. Returns
    (x_min, curvature a) with a>0 for a real minimum, else (grid argmin, nan)."""
    i = int(np.argmin(y))
    if not (0 < i < len(x) - 1):
        return float(x[i]), float("nan")
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    a = ((y0 + y2) - 2 * y1) / (2 * (x1 - x0) ** 2)
    if a <= 0:
        return float(x1), float("nan")
    return float(x1 - 0.5 * (y2 - y0) / (2 * a * (x1 - x0))), float(a)


def bootstrap_dm(trials, crops, freqs, dt, sigma, tau1ghz, n_boot=80, seed=0, sigma_max=1.0):
    """Honest DM + sigma from a SUB-BAND bootstrap of the coherent chi^2 envelope.

    ``crops`` is (n_trial, n_sb, n_t): the per-trial coherently-shifted sub-band waterfalls (burst shape
    frozen at sigma, tau1ghz). The full-sub-band envelope gives the DM point and shape; resampling the
    sub-bands with replacement and re-minimising gives sigma from sub-band-to-sub-band scatter -- the
    real information content (~n_sb timing points), NOT the n_ch*n_t per-sample count that makes the raw
    curvature absurdly overconfident.
    """
    trials = np.asarray(trials, float)
    rng = np.random.default_rng(seed)
    n_sb = crops.shape[1]

    def envelope(idx):
        return np.array(
            [
                fit_t0(crops[k][idx], freqs[idx], dt, sigma, tau1ghz, nbin=n_sb)[0]
                for k in range(len(trials))
            ]
        )

    full = envelope(np.arange(n_sb))
    dm0, a0 = _parabola_min(trials, full)
    c0 = float(full.min())
    dof = max(n_sb * crops.shape[2] - 1 - 2 * n_sb, 1)
    mins = []
    for _ in range(n_boot):
        dm_b, a_b = _parabola_min(trials, envelope(rng.integers(0, n_sb, n_sb)))
        if np.isfinite(a_b):
            mins.append(dm_b)
    mins = np.array(mins)
    dm = float(np.median(mins)) if mins.size else dm0
    sigma_dm = float(np.std(mins)) if mins.size >= 5 else float("inf")
    interior = bool(0 < int(np.argmin(full)) < len(full) - 1)
    excl95 = float(np.percentile(np.abs(mins - dm), 95)) if mins.size >= 5 else None
    constrains = bool(
        interior and np.isfinite(sigma_dm) and sigma_dm <= sigma_max and mins.size >= 0.5 * n_boot
    )
    return {
        "dm": dm,
        "sigma": sigma_dm if np.isfinite(sigma_dm) else None,
        "excl95_pc": excl95,
        "chi2_red_min": float(c0 / dof),
        "interior_min": interior,
        "constrains_dm": constrains,
        "n_boot_ok": len(mins),
        "dm_grid": trials.tolist(),
        "chi2_full": full.tolist(),
    }


def summarize(trial_dms, chi2, dof, sigma_max=1.0, gof_lo=0.3, gof_hi=3.0):
    """Turn a chi^2(trial DM) envelope into DM, sigma, and a constrains/exclusion verdict.

    Rescale so the reduced chi^2 at the minimum is 1 (absorbs noise mis-estimate), then sigma_DM is the
    Delta-chi^2 = 1 half-width and the 95% exclusion the Delta-chi^2 = 4 half-width (parabola near min).
    """
    trial_dms = np.asarray(trial_dms, float)
    chi2 = np.asarray(chi2, float)
    i0 = int(np.argmin(chi2))
    c0 = chi2[i0]
    scale = dof / c0 if c0 > 0 else 1.0  # rescale -> reduced chi^2 = 1 at min
    dchi = (chi2 - c0) * scale
    interior = 0 < i0 < len(chi2) - 1
    # local parabola for sub-grid DM + curvature sigma
    dm = float(trial_dms[i0])
    sigma = float("inf")
    if interior:
        x = trial_dms[i0 - 1 : i0 + 2]
        y = (chi2[i0 - 1 : i0 + 2] - c0) * scale
        a = ((y[0] + y[2]) - 2 * y[1]) / (2 * (x[1] - x[0]) ** 2)  # curvature
        if a > 0:
            dm = float(x[1] - 0.5 * ((y[2] - y[0]) / (2 * a * (x[1] - x[0]))))
            sigma = float(1.0 / np.sqrt(a))  # Delta-chi^2=1 half-width
    excl95 = _half_width(trial_dms, dchi, i0, 4.0)
    chi2_red_min = float(c0 / dof)
    constrains = bool(
        interior and np.isfinite(sigma) and sigma <= sigma_max and gof_lo <= chi2_red_min <= gof_hi
    )
    return {
        "dm": dm,
        "sigma": sigma if np.isfinite(sigma) else None,
        "excl95_pc": excl95,
        "chi2_red_min": chi2_red_min,
        "interior_min": interior,
        "constrains_dm": constrains,
        "dm_grid": trial_dms.tolist(),
        "dchi2": dchi.tolist(),
    }


def _half_width(dms, dchi, i0, level):
    """DM half-width where the rescaled Delta-chi^2 envelope crosses ``level`` on each side of the min."""

    def cross(side):
        idx = range(i0, -1, -1) if side < 0 else range(i0, len(dms))
        prev = i0
        for j in idx:
            if dchi[j] >= level:
                d0, d1 = dchi[prev], dchi[j]
                f = (level - d0) / (d1 - d0) if d1 != d0 else 0.0
                return abs(dms[prev] + f * (dms[j] - dms[prev]) - dms[i0])
            prev = j
        return None

    lo, hi = cross(-1), cross(1)
    vals = [v for v in (lo, hi) if v is not None]
    return float(max(vals)) if vals else None  # widest side; None if envelope never reaches level
