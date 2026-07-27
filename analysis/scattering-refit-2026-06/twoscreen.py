"""Two-screen forward model (Option A charter, rung 1): production EMG extended
with a SECOND thin-screen exponential PBF.

Pulse shape (per channel):

    Gaussian(sigma) (x) exp(tau_1(nu)) (x) exp(tau_2(nu))

with both screens physically tied to the SAME shared beta:

    tau_k(nu) = tau_k * nu^(-alpha),   alpha = 2 beta/(beta-2)  (UNCLAMPED)

and ONE extra parameter vs production, r = tau_2/tau_1 at 1 GHz (constant across
nu, since both screens carry the same alpha). Nested: r -> 0 recovers the
production single-screen EMG exactly (=> valid Bayes factor).

Closed form (charter's implementation note -- NO numerical convolution):
the convolution of two one-sided exponentials h_k(t)=(1/tau_k)exp(-t/tau_k) is

    (h_1 * h_2)(t) = [exp(-t/tau_2) - exp(-t/tau_1)] / (tau_2 - tau_1),  t>=0

so convolving with the Gaussian and using linearity,

    K(t) = [tau_2 * EMG(sigma,tau_2) - tau_1 * EMG(sigma,tau_1)] / (tau_2 - tau_1)

a difference of two PRODUCTION EMGs (burstfit.analytic_gaussian_exp_convolution),
each area-normalized, so K is area-normalized too (check: the tau prefactors give
(tau_2 - tau_1)/(tau_2 - tau_1) = 1).

Two numerical limits are guarded:

  r -> 0  (tau_2 -> 0): K -> EMG(sigma, tau_1). Nesting to production. Handled by
    an explicit r < R_FLOOR short-circuit (and the base EMG's own tau->0 Gaussian
    guard keeps the generic branch finite anyway).

  r -> 1  (tau_2 -> tau_1): the divided difference is 0/0 with catastrophic
    cancellation. The exact limit is the derivative

        K(t) = d/dtau [ tau * EMG(sigma,tau) ] |_{tau_1}

    which we evaluate in a cancellation-free closed form (see _dtau_tau_emg).
    The switch at |r-1| = R_UNIT_EPS is seamless: at |r-1|=1e-3 the generic
    difference still retains ~13 float64 digits, matching the derivative branch.

This module mirrors plpbf.FRBModelPLPBF / plpbf_loglike exactly -- the ONLY change
vs production is the PBF convolution seam; dispersion delay, DM smearing,
per-channel amplitude, and the downstream log_likelihood_gain_marginal are reused
byte-for-byte, so the three-way production / two-screen / free-alpha comparison
differs only in the tail shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcx

from scat_analysis.burstfit import (
    FRBModel,
    FRBParams,
    analytic_gaussian_exp_convolution,
)

# r below this is treated as the single-screen nesting limit (production EMG).
R_FLOOR = 1e-6
# |r - 1| below this uses the analytic tau_1 = tau_2 derivative limit.
R_UNIT_EPS = 1e-3


@dataclass
class FRBParamsTwoScreen(FRBParams):
    """FRBParams carrying the second-screen ratio r = tau_2/tau_1 at 1 GHz.

    alpha is UNCLAMPED (2 beta/(beta-2), same override as FRBParamsPLPBF /
    FRBParamsFreeAlpha) -- both screens share it, so r is achromatic and the
    composite width still scales exactly nu^-4 at fixed beta; the chromatic wedge,
    if any, comes purely from shape non-self-similarity (sigma and the time bin do
    not scale with nu), which is exactly what Stage 0 tests.

    ``r`` survives dataclasses.replace inside log_likelihood_gain_marginal (which
    only resets c0/gamma/zeta), so it reaches FRBModelTwoScreen.__call__. Default
    r = 0.0 nests to production, so a plain construction reproduces the single
    screen EMG exactly (nesting smoke).
    """

    r: float = 0.0

    @property
    def alpha(self) -> float:
        b = float(self.beta)
        if b <= 2.0:
            raise ValueError(f"beta must be > 2 for thin-screen alpha, got {b}")
        return 2.0 * b / (b - 2.0)


def _dtau_tau_emg(
    t: NDArray[np.floating],
    mu: NDArray[np.floating],
    sig: NDArray[np.floating],
    tau: NDArray[np.floating],
) -> NDArray[np.floating]:
    """d/dtau [ tau * EMG(sigma,tau) ] -- the r -> 1 (tau_1 = tau_2) kernel limit.

    EMG(sigma,tau)(t) = (1/2tau) exp(-u^2/2sigma^2) erfcx(b),
        u = t - mu,  b = sigma/(sqrt2 tau) - u/(sqrt2 sigma).
    Then tau*EMG = (1/2) exp(-u^2/2sigma^2) erfcx(b), and with
    d/dtau b = -sigma/(sqrt2 tau^2) and erfcx'(b) = 2 b erfcx(b) - 2/sqrt(pi),

        d/dtau[tau EMG] = (sigma/(sqrt2 tau^2)) exp(-u^2/2sigma^2) [1/sqrt(pi) - b erfcx(b)].

    Using the identity exp(-u^2/2sigma^2) erfcx(b) = 2 tau * EMG (from the EMG def)
    this becomes cancellation-free and reuses the fully stability-guarded base EMG:

        = sigma/(sqrt(2 pi) tau^2) exp(-u^2/2sigma^2)  -  (sqrt2 sigma b / tau) * EMG.

    Only a bare Gaussian (always safe) is evaluated raw; the erfcx factor rides on
    the guarded analytic_gaussian_exp_convolution, so deep-tail (b << 0) stability
    is inherited rather than re-derived.
    """
    t = np.atleast_2d(t)
    u = t - mu
    b = sig / (np.sqrt(2.0) * tau) - u / (np.sqrt(2.0) * sig)
    gauss = np.exp(-0.5 * (u / sig) ** 2)
    emg = analytic_gaussian_exp_convolution(t, mu, sig, tau)
    return sig / (np.sqrt(2.0 * np.pi) * tau ** 2) * gauss - (np.sqrt(2.0) * sig * b / tau) * emg


def two_screen_perchan(
    time: NDArray[np.floating],
    mu: NDArray[np.floating],
    sig: NDArray[np.floating],
    tau1: NDArray[np.floating],
    r: float,
) -> NDArray[np.floating]:
    """Per-channel Gaussian (x) exp(tau_1) (x) exp(tau_2=r*tau_1), area-normalized.

    ``mu``, ``sig``, ``tau1`` are (nf, 1); returns (nf, T). Same time grid and
    normalization convention as analytic_gaussian_exp_convolution (this IS a linear
    combination of two of them).
    """
    time = np.atleast_2d(time)
    r = float(r)
    if r < R_FLOOR:
        # nesting limit: second screen vanishes -> single-screen production EMG
        return analytic_gaussian_exp_convolution(time, mu, sig, tau1)
    if abs(r - 1.0) < R_UNIT_EPS:
        # tau_1 ~ tau_2 limit: the exact divided difference [f(t2)-f(t1)]/(t2-t1)
        # equals f'(tau_mid) at the MIDPOINT tau_mid=(tau1+tau2)/2 to 2nd order
        # (mean-value / centered-difference), so evaluate the derivative there --
        # NOT at the tau1 endpoint, which would leave an O(|r-1|) ~ 5e-4 jump at the
        # switch. With the midpoint the closed and derivative branches agree to
        # O((tau2-tau1)^2) ~ 1e-6 at |r-1|=R_UNIT_EPS (seamless likelihood surface).
        tau_mid = 0.5 * (1.0 + r) * tau1
        return _dtau_tau_emg(time, mu, sig, tau_mid)
    tau2 = r * tau1
    e1 = analytic_gaussian_exp_convolution(time, mu, sig, tau1)
    e2 = analytic_gaussian_exp_convolution(time, mu, sig, tau2)
    return (tau2 * e2 - tau1 * e1) / (tau2 - tau1)


class FRBModelTwoScreen(FRBModel):
    """FRBModel whose M2/M3 scattering branch uses the two-screen kernel instead of
    the single EMG. Everything up to the PBF-convolution seam is reused unchanged.

    Upgrade a prepared FRBModel in place with ``model.__class__ = FRBModelTwoScreen``
    (no new fields, no __init__), preserving prepared state (data, noise, off_pulse,
    dm_init). Mirrors FRBModelPLPBF.__call__ line-for-line except the final kernel.
    """

    def __call__(self, p, model_key="M3", freq_subset=None):
        if model_key not in {"M2", "M3"} or not (p.tau_1ghz > 1e-6):
            return super().__call__(p, model_key, freq_subset)

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

        alpha = p.alpha  # UNCLAMPED 2b/(b-2); both screens share it
        tau1 = np.clip(p.tau_1ghz * (freq / 1.0) ** (-alpha), 1e-6, None)[:, None]
        r = float(getattr(p, "r", 0.0))
        return amp[:, None] * two_screen_perchan(self.time, mu, sig, tau1, r)
