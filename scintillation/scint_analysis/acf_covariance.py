"""MC correlated-lag covariance for the ACF estimator (A1 trigger, limb-i likelihood).

Null-conditioned: realizations are single-screen with the *fitted* gamma-hat,
matched band / channelization / S/N — the correct conditioning for a
false-escalation test. The A1 design requires the two-component model to be
judged against a likelihood that knows neighbouring ACF lags are correlated
(scintles are smooth over ~gamma); with a diagonal likelihood the extra
component feeds on finite-scintle sample variance.

Realization physics: a thin screen with an exponential delay profile
p(t) = exp(-t/tau_s) gives a field autocorrelation R_E(dnu) = 1/(1 - 2*pi*i*
dnu*tau_s), hence an intensity ACF |R_E|^2 that is Lorentzian with HWHM
Delta nu_d = 1/(2*pi*tau_s) — the production HWHM convention (C1 = 1). We
synthesize E(nu) as the FFT of a complex white Gaussian weighted by sqrt(p(t))
on a doubled grid (to suppress circular wraparound), keep n_chan channels,
and pass the intensity through the production ``calculate_acf`` path so the
covariance is that of the estimator actually used, not of an idealization.
"""

from __future__ import annotations

import numpy as np

from .analysis import calculate_acf


def _one_realization(rng, n_chan, gamma_bins, mod_index=1.0, snr=np.inf):
    """One single-screen intensity spectrum (mean 1) with radiometer noise.

    Parameters
    ----------
    n_chan : int
        Number of frequency channels.
    gamma_bins : float
        Scintillation HWHM in channel units (gamma / channel_width).
    mod_index : float
        Target modulation index m (1 = fully modulated point source).
    snr : float
        Per-channel S/N; additive Gaussian noise of std 1/snr on the
        mean-1 spectrum. np.inf disables noise.
    """
    n_fft = 2 * n_chan  # doubled grid: suppress circular ACF wraparound
    tau_s = 1.0 / (2.0 * np.pi * gamma_bins)  # delay in 1/channel units
    t = np.arange(n_fft) / n_fft  # delay grid conjugate to channel index
    envelope = np.exp(-np.clip(t / max(tau_s, 1e-12), 0.0, 700.0))
    e_t = (rng.normal(size=n_fft) + 1j * rng.normal(size=n_fft)) * np.sqrt(
        envelope / 2.0
    )
    e_nu = np.fft.fft(e_t)[:n_chan]
    inten = np.abs(e_nu) ** 2
    inten = inten / inten.mean()
    if mod_index < 1.0:
        inten = 1.0 + mod_index * (inten - 1.0)
    if np.isfinite(snr):
        inten = inten + rng.normal(0.0, 1.0 / snr, n_chan)
    return inten


def _ledoit_wolf_lambda(A):
    """Shrinkage intensity toward the diagonal (Ledoit-Wolf-style estimate)."""
    n, p = A.shape
    X = A - A.mean(axis=0)
    S = X.T @ X / n
    # variance of the sample-covariance entries
    var_s = ((np.einsum("ni,nj->nij", X, X) - S) ** 2).sum(axis=0).sum() / n**2
    off = S - np.diag(np.diag(S))
    denom = float((off**2).sum())
    if denom <= 0:
        return 1.0
    return float(np.clip(var_s / denom, 0.0, 1.0))


def mc_acf_covariance(gamma_hwhm_mhz, mod_index, band_width_mhz,
                      channel_width_mhz, snr, n_real=500, max_lag_bins=120,
                      seed=0, shrink=None, return_diag_reference=False):
    """Correlated-lag covariance of the production ACF under a matched null.

    Returns the (max_lag_bins-1) x (max_lag_bins-1) covariance of the
    one-sided positive-lag ACF values produced by ``calculate_acf`` on
    single-screen realizations with the given gamma, modulation index, band,
    channelization, and S/N. Shrunk toward its diagonal (Ledoit-Wolf lambda
    unless ``shrink`` is given) and jittered for Cholesky stability.

    With ``return_diag_reference=True`` also returns the production quadrature
    diagonal error (stat + finite-scintle) of the first realization, for
    scale checks against the estimator's own error model.
    """
    rng = np.random.default_rng(seed)
    n_chan = int(round(band_width_mhz / channel_width_mhz))
    gamma_bins = gamma_hwhm_mhz / channel_width_mhz
    if gamma_bins <= 0:
        raise ValueError("gamma_hwhm_mhz must be positive")

    acfs = []
    diag_ref = None
    for _ in range(n_real):
        spec = _one_realization(rng, n_chan, gamma_bins,
                                mod_index=mod_index, snr=snr)
        acf_obj = calculate_acf(np.ma.masked_invalid(spec),
                                channel_width_mhz,
                                max_lag_bins=max_lag_bins)
        if acf_obj is None:
            continue
        pos = acf_obj.lags > 0
        vals = acf_obj.acf[pos]
        acfs.append(vals)
        if diag_ref is None and acf_obj.err is not None:
            diag_ref = acf_obj.err[pos]

    if len(acfs) < 30:
        raise ValueError(
            f"only {len(acfs)}/{n_real} realizations produced ACFs; "
            "cannot estimate a covariance"
        )
    n_lag = min(len(a) for a in acfs)
    A = np.array([a[:n_lag] for a in acfs])
    cov = np.cov(A, rowvar=False)

    lam = _ledoit_wolf_lambda(A) if shrink is None else float(shrink)
    cov = (1.0 - lam) * cov + lam * np.diag(np.diag(cov))
    cov = cov + 1e-12 * np.eye(n_lag)

    if return_diag_reference:
        return cov, (diag_ref[:n_lag] if diag_ref is not None else None)
    return cov
