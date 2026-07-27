"""Drop-in accelerated replacements for the two M3 hot kernels in
scat_analysis.burstfit (analytic_gaussian_exp_convolution, gaussian_powerlaw_convolution).

Both are algebraically identical to the originals -- same math, different array
organization -- so they are numerically equivalent to ~1e-15 (validated: max|Delta|
< 1e-14 on the kernel output over hundreds of random realistic (nf, nt, beta, tau, sig)
and max|Delta logL| < 1e-9 on stored posterior samples, the charter's bar). No special
functions are approximated; erfcx stays scipy's. The speedups come from removing wasted
work, not from changing the model:

  gaussian_powerlaw_convolution:  the power-law tail exp(-s_c)(lag/(tau s_c))^(-beta/2)
    factorizes -- tau enters only per-channel and lag only per-time -- so
      (lag/(tau s_c))^p = lag^p * (tau s_c)^(-p)
    turns an (nf, nt) fractional power (~24k pow evals) into an (nt,) vector times an
    (nf,) vector (an outer product). The FFTs are unchanged and remain the floor.

  analytic_gaussian_exp_convolution:  evaluate erfcx over the whole (clipped) array and
    fold the tiny-tau (Gaussian-limit) and deep-tail (b <= -25 asymptotic) edge cases in
    with np.where, instead of fancy-index masking + broadcast + per-branch scatter. Same
    values; the masking/allocation overhead (which dominated the original) is gone.

Use via burstfit_fast.patch() (monkeypatches scat_analysis.burstfit at runtime) or import
the functions directly. Deploy to the campaign only between runs -- never edit a shared
burstfit.py while fits are pending, since new jobs import it at start.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erfcx
try:
    from scipy.fft import next_fast_len as _next_fast_len
except Exception:  # pragma: no cover - scipy always present in the fit env
    def _next_fast_len(n):
        return 1 << int(np.ceil(np.log2(max(n, 1))))

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)


def analytic_gaussian_exp_convolution(t, mu, sig, tau):
    """Gaussian (x) exponential, erfcx-stable; equivalent to the burstfit original."""
    if t.ndim == 1:
        t = t[None, :]
    tmm = t - mu
    inv_tau = 1.0 / tau
    gauss_exp = np.exp(-0.5 * (tmm / sig) ** 2)
    b = (sig / (_SQRT2 * tau)) - (tmm / (_SQRT2 * sig))
    with np.errstate(over="ignore", invalid="ignore"):
        # main branch everywhere; clip b so erfcx never overflows (deep tail overwritten below)
        res = (0.5 * inv_tau) * gauss_exp * erfcx(np.maximum(b, -25.0))
        deep = b <= -25.0
        if deep.any():
            asymp = inv_tau * np.exp(0.5 * (sig / tau) ** 2 - tmm / tau)
            res = np.where(deep, asymp, res)
    is_gauss = (tau < 1e-9) | (sig > 100.0 * tau)
    if is_gauss.any():
        res = np.where(is_gauss, gauss_exp / (_SQRT2PI * sig), res)
    np.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return res


def gaussian_powerlaw_convolution(t, mu, sig, tau, beta):
    """Gaussian (x) power-law-tail thin-screen PBF; equivalent to the burstfit original."""
    if t.ndim == 1:
        t = t[None, :]
    T = t.shape[1]
    dt = t[0, 1] - t[0, 0]
    g = (1.0 / (_SQRT2PI * sig)) * np.exp(-0.5 * ((t - mu) / sig) ** 2)
    beta = float(np.clip(beta, 2.01, 3.99))
    s_c = max(2.0 * np.log(2.0 / (4.0 - beta)), 1e-3)
    p = -0.5 * beta
    lag1d = np.arange(T) * dt
    s = lag1d[None, :] / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        lagpow = np.empty(T)
        lagpow[0] = 0.0                                  # lag=0 is always in the core branch
        lagpow[1:] = lag1d[1:] ** p                      # one fractional power over (T-1,)
        cscale = np.exp(-s_c) * (tau * s_c) ** (-p)      # (nf, 1)
        tail = cscale * lagpow[None, :]                  # outer product, cheap
    h = np.where(s <= s_c, np.exp(-np.minimum(s, s_c)), tail)
    h = h / np.clip(h.sum(axis=1, keepdims=True) * dt, 1e-30, None)
    # Linear convolution needs L >= 2T-1; the original uses L=2T. When 2T has a large
    # prime factor (e.g. an auto-TF window with T prime -> L=2*T), pocketfft falls off
    # its O(N log N) path and the rfft/irfft dominate the whole fit (up to ~13x slower).
    # Padding to the next 5-smooth length is numerically identical -- the extra zeros are
    # discarded by the [:T] slice -- and restores the fast path.
    L = _next_fast_len(2 * T)
    conv = np.fft.irfft(np.fft.rfft(g, L, axis=1) * np.fft.rfft(h, L, axis=1), L, axis=1)
    return conv[:, :T] * dt


def patch():
    """Monkeypatch scat_analysis.burstfit to use the accelerated kernels. Returns the
    originals so a caller can restore them (used by the validation harness)."""
    from scat_analysis import burstfit as bf
    orig = (bf.analytic_gaussian_exp_convolution, bf.gaussian_powerlaw_convolution)
    bf.analytic_gaussian_exp_convolution = analytic_gaussian_exp_convolution
    bf.gaussian_powerlaw_convolution = gaussian_powerlaw_convolution
    return orig
