"""Power-law pulse-broadening function with a finite inner scale (Ostashov-Shishov
thin screen), and its convolution with a Gaussian intrinsic pulse.

Owner-chartered PL-PBF lane. Extends the production ``gaussian_powerlaw_convolution``
(burstfit.py:197 -- exp core + power-law tail, crossover s_c = 2 ln(2/(4-beta)),
nests the EMG at beta->4) with the THIRD regime it lacks: an inner-scale exponential
cutoff. Standalone (no production edits) so it can be built and injection-validated
while the EMG mass-refit runs; integration into the fit is a later, gated step.

Physics (Cordes, Ocker, Chatterjee et al. 2025, sec. 11.2, Fig 40 caption):
  As beta->4 from below the thin-screen PBF is exponential exp(-t/tau_e) for small
  t <~ t* = 2 tau_e ln(2/(4-beta)), a power-law (t/tau_e)^(-beta/2) for larger t, and
  -- with a FINITE inner scale -- decays exponentially again, exp(-t/tau_i), for
  t >~ tau_i, where tau_i ∝ l_i^(-2). The inner-scale ratio is zeta = l_i/l_d:
  SMALL zeta (0.01) => tau_i is many 1/e-scales out => full heavy tail; LARGE zeta
  (>~10) => tau_i short => the power-law regime is squeezed out and only the
  exponential core survives (the monoscale / Gaussian-image limit -> recovers EMG).

Parameterization: we fit s_i = tau_i / tau_e (the tail-cutoff lag in 1/e-scale units),
the quantity the data actually constrains. s_i -> inf recovers the production
no-inner-scale PL-PBF; the exact s_i(zeta) mapping (their Fig 41) is a post-hoc
physical relabel, filled in when the excerpt lands. tau here is the 1/e width tau_e
(NOT the mean, which diverges for beta<4 without the cutoff).
"""

from __future__ import annotations

import numpy as np

# thin-screen integrable, non-degenerate open interval for beta (matches production)
BETA_MIN, BETA_MAX = 2.01, 3.99


def pbf_innerscale(lag, tau, beta, s_i):
    """Area-normalizable one-sided PBF p(t) on a causal lag grid (lag >= 0), in three
    regimes of s = lag/tau (Cordes Fig 40), with s_c = 2 ln(2/(4-beta)) the core->tail
    crossover and s_i = tau_i/tau_e the inner-scale cutoff lag:

        s <= s_c        : exp(-s)                                   [exponential core]
        s_c < s < s_i   : exp(-s_c) (s/s_c)^(-beta/2)               [power-law, cont. at s_c]
        s >= s_i        : [pl(s_i)] exp(-(s-s_i)/s_i)               [inner-scale exp cutoff]

    Limits:
      s_i -> inf  : the cutoff branch never triggers -> the production no-inner-scale
                    PL-PBF (exact nesting of gaussian_powerlaw_convolution).
      s_i <= s_c  : the power-law window is closed (inner scale inside the core) -> pure
                    exponential (the monoscale / large-inner-scale limit; Fig 40 ζ large).
      beta -> 4   : s_c -> inf -> all core -> exponential.

    NOTE the cutoff runs from s_i (not s_c): the power-law is CLEAN up to s_i, matching
    Fig 40 where each curve overlays the shared t^(-beta/2) line until it peels off near
    tau_i. (An earlier cutoff-from-s_c form softened the power-law and biased beta high.)
    """
    beta = float(np.clip(beta, BETA_MIN, BETA_MAX))
    s_c = max(2.0 * np.log(2.0 / (4.0 - beta)), 1e-3)
    s = np.asarray(lag, float) / float(tau)
    core = np.exp(-s)
    if not np.isfinite(s_i):
        with np.errstate(over="ignore", invalid="ignore"):
            tail = np.exp(-s_c) * (np.maximum(s, s_c) / s_c) ** (-0.5 * beta)
        h = np.where(s <= s_c, core, tail)
    elif s_i <= s_c:
        h = core  # power-law window closed -> pure exponential
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            pl = np.exp(-s_c) * (np.maximum(s, s_c) / s_c) ** (-0.5 * beta)
            pl_si = np.exp(-s_c) * (s_i / s_c) ** (-0.5 * beta)
            cut = pl_si * np.exp(-(s - s_i) / s_i)
        tail = np.where(s < s_i, pl, cut)
        h = np.where(s <= s_c, core, tail)
    return np.where(np.isfinite(h), h, 0.0)


def zeta_from_si(s_i):
    """Inner-scale ratio zeta = l_i/l_d from s_i = tau_i/tau_e. The paper's derivation
    gives s_i = zeta^-2 (tau_i propto l_i^-2), but that mapping is INCONSISTENT with
    Fig 40 (zeta=1 shows a clear power-law tail to t/tau~30, whereas zeta^-2=1 would
    close the window): the preprint is a draft with known internal inconsistencies.
    We therefore fit s_i directly (the peel-off lag the data constrains) and treat zeta
    as an uncertain relabel. Returned per the stated s_i=zeta^-2 for reference only."""
    return float(s_i) ** -0.5 if np.isfinite(s_i) and s_i > 0 else 0.0


def si_chromatic(s_i0, freq_GHz, nu0_GHz, beta):
    """Chromatic inner-scale lag s_i(nu) = s_i0 (nu/nu0)^(+4/(beta-2)) from ζ(ν) ∝
    ν^(-2/(β-2)) with l_i fixed (Cordes Fig 58 caption). Tail is LONGER at high ν
    (DSA), shorter at low ν (CHIME) -- a falsifiable structural prediction."""
    return s_i0 * (np.asarray(freq_GHz, float) / nu0_GHz) ** (4.0 / (beta - 2.0))


def gaussian_pbf_innerscale_convolution(t, mu, sig, tau, beta, s_i):
    """Gaussian G(t; mu, sig) (x) inner-scale PL-PBF, area-normalized, same convention
    and zero-padded-FFT linear convolution as burstfit.gaussian_powerlaw_convolution
    (so EMG-vs-PL-PBF evidence is on the same amplitude footing).

    Grid caveat (Cordes: the tail can span t/tau ~ 1..1e4-1e6 at small zeta): the
    kernel is evaluated only on [0, T*dt]; a tail longer than the window is truncated
    (physically fine -- the likelihood only sees the on-pulse window), and L=2T zero
    padding keeps the linear convolution wrap-around-free.
    """
    t = np.atleast_2d(t)
    T = t.shape[1]
    dt = t[0, 1] - t[0, 0]
    g = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((t - mu) / sig) ** 2)

    lag = (np.arange(T) * dt)[None, :]
    h = pbf_innerscale(lag, tau, beta, s_i)
    h = h / np.clip(h.sum(axis=1, keepdims=True) * dt, 1e-30, None)

    L = 2 * T
    conv = np.fft.irfft(np.fft.rfft(g, L, axis=1) * np.fft.rfft(h, L, axis=1), L, axis=1)
    return conv[:, :T] * dt
