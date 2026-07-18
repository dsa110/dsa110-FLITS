"""Relaxed-alpha joint likelihood: decouple the tau(nu) scaling index alpha from
the PBF-shape index beta.

Motivation (Cordes, Ocker, Chatterjee et al. 2025 PTA-noise preprint, sec. 12.6.4):
an exponential PBF is self-consistent ONLY with tau ~ nu^-4. Our production fit is
beta-native (ADR-0006): beta in [3,4] drives BOTH the PBF shape
(``gaussian_powerlaw_convolution``, reducing to the pure exponential as beta->4) AND
the frequency scaling via the thin-screen tie alpha = alpha_from_beta(beta) =
2 beta/(beta-2). That tie forces alpha >= 4, and as beta->4 both the PBF shape ->
exponential and alpha -> 4 simultaneously -- so the observed beta->4 / alpha=4 rail
may be the fit sliding to the exponential PBF's only self-consistent point rather
than a data-driven measurement.

This module frees alpha as an INDEPENDENT sampled parameter (prior alpha in [2,6])
while beta continues to set the PBF shape (prior kept at [3,4] -- NOT widened). If
the freed alpha posterior stays >= 4 the nu^-4 scaling is data-driven and strengthens
the one-sided limit; if it drops below 4 that is evidence of PBF-shape mismatch or a
bounded/inhomogeneous screen (Cordes sec. 11.4 / 12.7), feeding the shape-model
escalation already open for casey + wilhelm.

Everything else is identical to the production shared-zeta gain-marginal path
(``_JointLogLikelihoodGainSharedZeta``): mask-aware binning lives in the prepared
FRBModel, the per-channel amplitude/gain is marginalized analytically, one source
width law zeta(nu)=zeta_1ghz*nu^x_zeta spans both bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scat_analysis.burstfit import FRBModel, FRBParams


@dataclass
class FRBParamsFreeAlpha(FRBParams):
    """FRBParams whose scattering index alpha is an explicit field, not derived
    from beta. beta still governs the PBF shape; only the tau(nu) exponent is
    overridden. ``dataclasses.replace`` (used inside the gain-marginal kernel)
    preserves the subclass and this field, so the override survives model eval.

    This subclass is instantiated ONLY inside the likelihood; the plain FRBParams
    ``init`` objects passed to ``build_priors`` are untouched, so the prior
    machinery never sees this extra field.
    """

    alpha_override: float = 4.0

    @property
    def alpha(self) -> float:
        return float(self.alpha_override)


class JointLogLikelihoodSharedZetaFreeAlpha:
    """Shared-zeta joint gain-marginal logL with alpha decoupled from beta.

    9-vector theta = [tau, beta, alpha, zeta_1ghz, x_zeta, t0_C, ddm_C, t0_D, ddm_D].
    Mirrors ``burstfit_joint._JointLogLikelihoodGainSharedZeta`` exactly except
    for the inserted free alpha at index 2 and the FRBParamsFreeAlpha container.
    Picklable (module-level class holding two FRBModels) for dynesty.pool.
    """

    def __init__(self, model_C: FRBModel, model_D: FRBModel):
        self.model_C = model_C
        self.model_D = model_D

    @staticmethod
    def _band_ll(
        model: FRBModel,
        tau: float,
        beta: float,
        alpha: float,
        z1: float,
        x: float,
        t0: float,
        ddm: float,
    ) -> float:
        zeta_nu = z1 * np.asarray(model.freq, dtype=float) ** x  # full channel axis
        p = FRBParamsFreeAlpha(
            c0=1.0,
            t0=t0,
            gamma=0.0,
            zeta=zeta_nu,
            tau_1ghz=tau,
            beta=beta,
            delta_dm=ddm,
            alpha_override=alpha,
        )
        return model.log_likelihood_gain_marginal(p, "M3")

    def __call__(self, theta: NDArray[np.floating]) -> float:
        tau, beta, alpha, z1, x = (float(theta[i]) for i in range(5))
        ll = self._band_ll(
            self.model_C, tau, beta, alpha, z1, x, float(theta[5]), float(theta[6])
        ) + self._band_ll(
            self.model_D, tau, beta, alpha, z1, x, float(theta[7]), float(theta[8])
        )
        return ll if np.isfinite(ll) else -1e100
