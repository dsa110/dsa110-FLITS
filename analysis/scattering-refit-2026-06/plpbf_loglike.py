"""PL-PBF joint likelihood: swap the EMG (Gaussian(x)exponential-PBF) kernel for the
inner-scale power-law PBF while keeping the SAME shared-zeta gain-marginal joint
structure as production / relaxed-alpha.

Physics (Cordes, Ocker, Chatterjee et al. 2025 PTA-noise preprint, sec. 11.2 / Fig 40,
and the chromatic inner scale of Fig 58): a power-law density-wavenumber spectrum
2 < beta < 4 with a finite inner scale l_i gives a thin-screen PBF with three regimes in
s = lag/tau_e (s_c = 2 ln(2/(4-beta)) is the core->tail crossover, s_i = tau_i/tau_e the
inner-scale cutoff lag):

    s <= s_c      : exp(-s)                            [exponential core]
    s_c < s < s_i : exp(-s_c) (s/s_c)^(-beta/2)        [CLEAN power-law, continuous at s_c]
    s >= s_i      : pl(s_i) exp(-(s-s_i)/s_i)          [inner-scale exp cutoff]

Unlike the relaxed-alpha diagnostic wedge, this is a SELF-CONSISTENT physical model:
alpha stays TIED to beta (alpha = 2 beta/(beta-2), inherited from FRBParams.alpha), and
the inner scale is CHROMATIC -- s_i(nu) = s_i0 (nu/nu0)^(+4/(beta-2)) at fixed l_i (Fig 58
caption; tail LONGER at high nu / DSA, shorter at low nu / CHIME), a falsifiable structural
prediction the joint CHIME+DSA lever arm can test.

Nesting (exact): s_i -> inf reproduces the production PL-PBF
(``gaussian_powerlaw_convolution``); s_i <= s_c (or beta -> 4) reproduces the pure
exponential -- our current EMG. So EMG, production PL-PBF, and PL-PBF-with-inner-scale
are on the same amplitude/evidence footing, and the ONLY change vs production is the PBF
tail shape (the dispersion / smearing / gain-marginal machinery is byte-identical).

The kernel regimes match plpbf.pbf_innerscale / plpbf_inject._kernel exactly (the same
three-regime, cutoff-from-s_i form validated against Cordes Fig 40 by injection recovery);
here it is lifted to the per-channel (nf, T) form the production FRBModel.__call__ uses, so
the swap happens only at the PBF-convolution seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scat_analysis.burstfit import (
    BETA_EXP_EPS,
    BETA_THIN_SCREEN_MAX,
    BETA_THIN_SCREEN_MIN,
    FRBModel,
    FRBParams,
    _next_fast_len,
    analytic_gaussian_exp_convolution,
    gaussian_powerlaw_convolution,
)

# clip bounds for the internal PBF (matches plpbf.pbf_innerscale)
_PBF_BETA_MIN, _PBF_BETA_MAX = 2.01, 3.99


@dataclass
class FRBParamsPLPBF(FRBParams):
    """FRBParams carrying the inner-scale cutoff lag s_i (its value at 1 GHz = NU0).

    alpha stays TIED to beta, but via the UNCLAMPED thin-screen relation
    alpha = 2 beta/(beta-2) (overridden below) -- NOT the production
    ``alpha_from_beta``, which hard-clamps alpha to 4.0 for beta >= 3.98
    (BETA_THIN_SCREEN_MAX - BETA_EXP_EPS). That clamp is an exponential-PBF beta->4
    self-consistency artifact: under the PL-PBF it would (a) put a step
    discontinuity in the model/likelihood at beta=3.98 (2*3.98/1.98=4.0202 snapping
    to 4.000), (b) freeze the chromatic lever arm flat across [3.98, 4) exactly where
    a ceiling-adjacent posterior lives (wilhelm's free-alpha beta=3.965+0.015/-0.020
    straddles it), and (c) erase the super-4 chromaticity (beta=3.667 -> alpha=4.40)
    that is the physical heavy-tail signature the three-way test is built to measure.
    The prior is beta in [3,4], so the unclamped tie is the correct thin-screen map
    everywhere in range.

    ``s_i`` survives ``dataclasses.replace`` inside ``log_likelihood_gain_marginal``
    (which resets c0/gamma), so both it and the alpha override reach
    ``FRBModelPLPBF.__call__``. Default s_i = inf nests to the production PL-PBF, so a
    plain construction reproduces ``gaussian_powerlaw_convolution`` exactly (nesting
    smoke).
    """

    s_i: float = np.inf

    @property
    def alpha(self) -> float:
        # Unclamped thin-screen tie (Cordes review sec 11.2 / eq alpha=2beta/(beta-2));
        # keep only the beta>2 integrability guard, drop the production beta>=3.98 clamp.
        b = float(self.beta)
        if b <= 2.0:
            raise ValueError(f"beta must be > 2 for thin-screen alpha, got {b}")
        return 2.0 * b / (b - 2.0)


def _innerscale_perchan(
    time: NDArray[np.floating],
    mu: NDArray[np.floating],
    sig: NDArray[np.floating],
    tau: NDArray[np.floating],
    beta: float,
    s_i_ch: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Per-channel Gaussian(x)inner-scale-PL-PBF, area-normalized, same convention and
    zero-padded-FFT linear convolution as ``burstfit.gaussian_powerlaw_convolution`` --
    ONLY the PBF tail differs (finite inner-scale cutoff). ``tau`` and ``s_i_ch`` are
    (nf, 1); ``s_i_ch`` is the chromatic inner-scale lag already scaled to each channel.
    Mirrors plpbf_inject._kernel's tail construction (cutoff from s_i, NOT s_c).
    """
    time = np.atleast_2d(time)
    T = time.shape[1]
    dt = time[0, 1] - time[0, 0]

    # Area-normalized Gaussian (same convention as the exp / PL paths)
    g = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((time - mu) / sig) ** 2)

    beta = float(np.clip(beta, _PBF_BETA_MIN, _PBF_BETA_MAX))
    s_c = max(2.0 * np.log(2.0 / (4.0 - beta)), 1e-3)
    lag = (np.arange(T) * dt)[None, :]
    s = lag / tau  # (nf, T)
    core = np.exp(-s)
    with np.errstate(over="ignore", invalid="ignore"):
        pl = np.exp(-s_c) * (np.maximum(s, s_c) / s_c) ** (-0.5 * beta)
        pl_si = np.exp(-s_c) * (np.maximum(s_i_ch, s_c) / s_c) ** (-0.5 * beta)  # value at s_i
        cut = pl_si * np.exp(-(s - s_i_ch) / s_i_ch)
        tail = np.where(s < s_i_ch, pl, cut)          # clean PL up to s_i, then exp cutoff
        tail = np.where(s <= s_c, core, tail)
        h = np.where(s_i_ch <= s_c, core, tail)        # per-channel window closed -> pure exp
    h = np.where(np.isfinite(h), h, 0.0)
    h = h / np.clip(h.sum(axis=1, keepdims=True) * dt, 1e-30, None)

    L = _next_fast_len(2 * T)
    conv = np.fft.irfft(np.fft.rfft(g, L, axis=1) * np.fft.rfft(h, L, axis=1), L, axis=1)
    return conv[:, :T] * dt


class FRBModelPLPBF(FRBModel):
    """FRBModel whose M3/M2 scattering branch uses the inner-scale PL-PBF kernel with a
    CHROMATIC s_i, instead of the EMG (``gaussian_powerlaw_convolution`` /
    ``analytic_gaussian_exp_convolution``). Everything up to the PBF-convolution seam --
    dispersion delay, DM-dependent intra-channel smearing, per-channel amplitude, and the
    downstream ``log_likelihood_gain_marginal`` -- is reused unchanged, so the three-way
    EMG vs production-PL vs inner-scale-PL comparison differs ONLY in the tail shape.

    Upgrade a prepared FRBModel in place with ``model.__class__ = FRBModelPLPBF`` (no new
    fields, no __init__), preserving all prepared state (data, noise, off_pulse, dm_init).
    """

    def __call__(self, p, model_key="M3", freq_subset=None):
        # Non-scattering models, or negligible tau: identical to production.
        if model_key not in {"M2", "M3"} or not (p.tau_1ghz > 1e-6):
            return super().__call__(p, model_key, freq_subset)

        # --- mirror FRBModel.__call__ up to the PBF-convolution seam (burstfit.py:733-775) ---
        freq = self.freq if freq_subset is None else self.freq[freq_subset]
        ref_freq = np.median(self.freq)
        amp = p.c0 * (freq / ref_freq) ** p.gamma

        dd_full = self._dispersion_delay(p.delta_dm)
        mu = p.t0 + (dd_full if freq_subset is None else dd_full[freq_subset])[:, None]

        dm_smear = self.dm_init + p.delta_dm
        zeta_arg = p.zeta if model_key in {"M1", "M3"} else 0.0
        sig_full = self._smearing_sigma(dm_smear, zeta_arg)
        sig = (sig_full if freq_subset is None else sig_full[freq_subset])[:, None]
        sig = np.clip(sig, 1e-6, None)

        alpha = p.alpha  # TIED via FRBParamsPLPBF's UNCLAMPED 2b/(b-2) override, NOT the
        #                  production alpha_from_beta clamp; PL-PBF is self-consistent
        tau = np.clip(p.tau_1ghz * (freq / 1.0) ** (-alpha), 1e-6, None)[:, None]
        beta_pbf = float(np.clip(p.beta, BETA_THIN_SCREEN_MIN, BETA_THIN_SCREEN_MAX))

        s_i = float(getattr(p, "s_i", np.inf))
        if not np.isfinite(s_i):
            # exact nesting to production: exp branch near beta=4, else power-law PBF
            if beta_pbf >= BETA_THIN_SCREEN_MAX - BETA_EXP_EPS:
                return amp[:, None] * analytic_gaussian_exp_convolution(self.time, mu, sig, tau)
            return amp[:, None] * gaussian_powerlaw_convolution(self.time, mu, sig, tau, beta_pbf)

        # chromatic inner-scale lag per channel: s_i(nu) = s_i0 (nu/nu0)^(+4/(beta-2))
        s_i_ch = (s_i * (freq / 1.0) ** (4.0 / (beta_pbf - 2.0)))[:, None]
        return amp[:, None] * _innerscale_perchan(self.time, mu, sig, tau, beta_pbf, s_i_ch)


class JointLogLikelihoodSharedZetaPLPBF:
    """Shared-zeta joint gain-marginal logL with the inner-scale PL-PBF kernel.

    9-vector theta = [tau, beta, log10_s_i, zeta_1ghz, x_zeta, t0_C, ddm_C, t0_D, ddm_D].
    Mirrors ``JointLogLikelihoodSharedZetaFreeAlpha`` exactly except index 2 is log10 s_i
    (not free alpha), alpha stays tied, and the container is FRBParamsPLPBF. The two models
    MUST be FRBModelPLPBF instances (upgrade with ``__class__``) so ``__call__`` dispatches
    to the inner-scale kernel. Picklable (module-level, holds two models) for dynesty.pool.
    """

    def __init__(self, model_C: FRBModel, model_D: FRBModel):
        # upgrade the prepared models in place (no new fields / __init__)
        model_C.__class__ = FRBModelPLPBF
        model_D.__class__ = FRBModelPLPBF
        self.model_C = model_C
        self.model_D = model_D

    @staticmethod
    def _band_ll(
        model: FRBModel,
        tau: float,
        beta: float,
        s_i: float,
        z1: float,
        x: float,
        t0: float,
        ddm: float,
    ) -> float:
        zeta_nu = z1 * np.asarray(model.freq, dtype=float) ** x  # full channel axis
        p = FRBParamsPLPBF(
            c0=1.0,
            t0=t0,
            gamma=0.0,
            zeta=zeta_nu,
            tau_1ghz=tau,
            beta=beta,
            delta_dm=ddm,
            s_i=s_i,
        )
        return model.log_likelihood_gain_marginal(p, "M3")

    def __call__(self, theta: NDArray[np.floating]) -> float:
        tau, beta, log10_s_i, z1, x = (float(theta[i]) for i in range(5))
        s_i = 10.0 ** log10_s_i
        ll = self._band_ll(
            self.model_C, tau, beta, s_i, z1, x, float(theta[5]), float(theta[6])
        ) + self._band_ll(
            self.model_D, tau, beta, s_i, z1, x, float(theta[7]), float(theta[8])
        )
        return ll if np.isfinite(ll) else -1e100
