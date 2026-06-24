"""Telescope-agnostic DM measurement: wide incoherent search + scatter-corrected arrival regression.

Purpose-built to replace the structure-maximizing ``DMPhaseEstimator`` for co-detection bursts that
are smooth / scatter-broadened / low-S/N and lack the temporal sub-structure structure-max needs
(see .agents/audit-chime-side-dm.md; expert methodology verdict 2026-06-24). Works on any band given
a ``freqs`` array — validated on both CHIME (400-800 MHz) and DSA-110 (~1.28-1.53 GHz) regimes.

Two stages, on an intensity waterfall already coherently dedispersed at ``dm_ref``:
  STAGE 1 (wide, reference-independent): incoherently re-dedisperse over a broad trial-DM grid and
    take the band-collapsed peak S/N -> a coarse DM that can be far from dm_ref (so a real offset
    CAN be found -> the agreement test is an exclusion, not a circular delta~0).
  STAGE 2 (precise, scattering-aware): at the coarse DM, split the band into sub-bands, fit each
    band-collapsed profile with an exponentially-modified Gaussian (Gaussian (x) one-sided
    exp(-t/tau)); the fitted Gaussian centre t0 is the scattering-DECONVOLVED arrival time.
    Weighted-linear-regress t0 vs K_DM*(nu^-2 - nu_ref^-2): slope = residual DM, covariance = sigma_DM.
  Smooth low-S/N bursts -> few sub-bands survive / large sigma_DM -> "does not constrain DM" (no
  fabricated value), leaning on other association pillars.

Pure numpy/scipy (no flits imports) so the SAME module runs in the baseband docker image and host.
K_DM mirrors flits.common.constants.K_DM.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc, erfcx

K_DM = 4.148808e3  # s MHz^2 pc^-1 cm^3 (== flits.common.constants.K_DM; inlined for docker)


def exgauss(t, t0, sigma, tau, amp, base):
    """Exponentially-modified Gaussian (Gaussian(sigma) convolved one-sided exp(-t/tau)).

    ``t0`` is the centre of the intrinsic Gaussian = the scattering-deconvolved arrival time.
    Numerically stable piecewise: erfcx(z)=exp(z^2)erfc(z) keeps the rising edge (z>=0) from
    overflowing, while the scattering tail (z<0, where erfcx itself overflows) uses the direct
    exp*erfc form whose exponent is then small.
    """
    sigma = abs(sigma) + 1e-12
    tau = abs(tau) + 1e-12
    dt = np.asarray(t, float) - t0
    z = (sigma / tau - dt / sigma) / np.sqrt(2.0)
    out = np.empty(np.broadcast(dt, z).shape, float)
    pos = z >= 0
    out[pos] = np.exp(-0.5 * (dt[pos] / sigma) ** 2) * erfcx(z[pos])
    a = np.clip(0.5 * (sigma / tau) ** 2 - dt[~pos] / tau, -700.0, 700.0)
    out[~pos] = np.exp(a) * erfc(z[~pos])
    return base + amp * out


def _dedisperse(wf, freqs, dt, ddm, nu_ref):
    """Incoherently shift each channel by the dispersive delay of a residual DM ``ddm``.

    Zero-fills the shifted-in edge (does NOT wrap): a short recording window (e.g. a ~143 ms CHIME
    singlebeam) would otherwise wrap a large shift back into the burst and fake an alignment.
    """
    shifts = np.round(K_DM * ddm * (1.0 / freqs**2 - 1.0 / nu_ref**2) / dt).astype(int)
    out = np.zeros_like(wf)
    n = wf.shape[1]
    for j, s in enumerate(shifts):
        if abs(s) >= n:
            continue
        if s > 0:
            out[j, : n - s] = wf[j, s:]
        elif s < 0:
            out[j, -s:] = wf[j, : n + s]
        else:
            out[j] = wf[j]
    return out


def _coarse_dm(wf, freqs, dt, nu_ref, dm_window, dm_step):
    """Stage 1: band-collapsed peak S/N over a wide residual-DM grid. Returns (ddm*, grid, snr, i)."""
    grid = np.arange(-dm_window, dm_window + dm_step, dm_step)
    snr = np.empty(grid.size)
    for k, ddm in enumerate(grid):
        prof = _dedisperse(wf, freqs, dt, ddm, nu_ref).sum(0)
        noise = np.std(prof[: max(prof.size // 5, 5)]) + 1e-12
        snr[k] = (prof.max() - np.median(prof)) / noise
    i = int(np.argmax(snr))
    return float(grid[i]), grid, snr, i


def _fit_subband_arrival(profile, dt, min_snr=4.0):
    """Fit a scattered pulse to a 1-D profile; return (t0_s, t0_err_s, snr) or None."""
    n = profile.size
    t = np.arange(n) * dt
    base0 = float(np.median(profile))
    noise = float(np.std(profile[: max(n // 5, 5)])) + 1e-12
    pk = int(np.argmax(profile))
    amp0 = float(profile[pk] - base0)
    if amp0 / noise < min_snr:
        return None
    half = base0 + 0.5 * amp0
    above = np.where(profile > half)[0]
    w0 = max((above.max() - above.min()) * dt, 2 * dt) if above.size > 1 else 4 * dt
    p0 = [t[pk] - 0.3 * w0, 0.4 * w0, 0.4 * w0, amp0, base0]
    bounds = (
        [0.0, dt, 0.0, 0.2 * amp0, base0 - 5 * noise],
        [t[-1], n * dt, n * dt, 5 * amp0, base0 + 5 * noise],
    )
    try:
        popt, pcov = curve_fit(
            exgauss, t, profile, p0=p0, bounds=bounds, sigma=np.full(n, noise), maxfev=6000
        )
    except (RuntimeError, ValueError):
        return None
    t0, _sigma, _tau, amp, _base = popt
    t0_err = float(np.sqrt(abs(pcov[0, 0])))
    if not np.isfinite(t0_err) or t0_err <= 0 or amp / noise < min_snr or t0_err > 0.5 * n * dt:
        return None
    return float(t0), t0_err, float(amp / noise)


def measure_dm(
    wf,
    freqs,
    dt,
    dm_ref,
    n_subband=8,
    dm_window=50.0,
    dm_step=0.5,
    min_snr=4.0,
    min_good_subbands=3,
    dm_err_max=20.0,
):
    """Measure DM from a coherently-dedispersed (at ``dm_ref``) intensity waterfall (n_freq, n_time).

    ``freqs`` are channel centres [MHz] (any order); ``dt`` is [s]; ``dm_ref`` is [pc/cm^3].
    Returns a dict: dm, dm_err (None if unconstrained), constrains_dm, reason, coarse_dm,
    n_good_subbands, snr, plus per-subband (freq, t0, t0_err, snr) and the coarse S/N(DM) curve.
    """
    wf = np.nan_to_num(np.asarray(wf, float))  # masked/NaN channels -> 0 (real data has them)
    freqs = np.asarray(freqs, float)
    order = np.argsort(freqs)
    wf, freqs = wf[order], freqs[order]
    nu_ref = float(freqs.max())

    ddm_c, grid, snr_curve, ic = _coarse_dm(wf, freqs, dt, nu_ref, dm_window, dm_step)
    coarse_dm = float(dm_ref + ddm_c)
    peak_snr = float(snr_curve[ic])
    wf_dd = _dedisperse(wf, freqs, dt, ddm_c, nu_ref)  # align at coarse DM -> small residual

    edges = np.linspace(0, wf_dd.shape[0], n_subband + 1, dtype=int)
    sub_nu, sub_t0, sub_err, sub_snr = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b - a < 1:
            continue
        fit = _fit_subband_arrival(
            np.nansum(np.nan_to_num(wf_dd[a:b]), axis=0), dt, min_snr=min_snr
        )
        if fit is None:
            continue
        sub_nu.append(float(freqs[a:b].mean()))
        sub_t0.append(fit[0])
        sub_err.append(fit[1])
        sub_snr.append(fit[2])

    n_good = len(sub_nu)
    base = {
        "dm_ref": float(dm_ref),
        "coarse_dm": coarse_dm,
        "peak_snr": peak_snr,
        "n_good_subbands": n_good,
        "snr": float(np.sqrt(np.sum(np.square(sub_snr)))) if sub_snr else 0.0,
        "subbands": [
            {"freq_mhz": f, "t0_s": t, "t0_err_s": e, "snr": s}
            for f, t, e, s in zip(sub_nu, sub_t0, sub_err, sub_snr, strict=True)
        ],
        "coarse_curve": {"dm": (dm_ref + grid).tolist(), "snr": snr_curve.tolist()},
        "railed": bool(ic == 0 or ic == grid.size - 1),
    }
    if n_good < min_good_subbands:
        return {
            **base,
            "dm": None,
            "dm_err": None,
            "constrains_dm": False,
            "reason": f"only {n_good} sub-bands above S/N {min_snr} (<{min_good_subbands})",
        }

    nu = np.array(sub_nu)
    x = K_DM * (1.0 / nu**2 - 1.0 / nu_ref**2)  # s per unit residual DM
    t0 = np.array(sub_t0)
    w = 1.0 / np.array(sub_err) ** 2
    X = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ ((X * w[:, None]).T @ t0)
    slope = beta[0]
    # inflate the formal (stat-only) covariance by the reduced chi^2 of the linear fit: if the
    # sub-band arrival times scatter more than their fit errors (profile evolution, scattering,
    # bandpass) sigma_DM grows honestly — avoids the over-confident stat-only error.
    resid = t0 - X @ beta
    dof = max(x.size - 2, 1)
    chi2_red = float(np.sum(w * resid**2) / dof)
    dm_err = float(np.sqrt(abs(cov[0, 0]) * max(chi2_red, 1.0)))
    dm = float(coarse_dm + slope)
    constrains = bool(np.isfinite(dm_err) and dm_err < dm_err_max and not base["railed"])
    reason = "ok"
    if base["railed"]:
        reason = "coarse DM railed at search-grid edge"
    elif not constrains:
        reason = f"sigma_DM={dm_err:.1f} >= {dm_err_max} pc/cm^3"
    return {**base, "dm": dm, "dm_err": dm_err, "constrains_dm": constrains, "reason": reason}
