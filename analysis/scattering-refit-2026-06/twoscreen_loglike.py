"""Two-screen joint likelihood: swap the single-screen EMG for the two-screen
kernel (twoscreen.two_screen_perchan) while keeping the SAME shared-zeta
gain-marginal joint structure as production / PL-PBF / relaxed-alpha.

9-vector theta = [tau_1, beta, log10_r, zeta_1ghz, x_zeta, t0_C, ddm_C, t0_D, ddm_D].
Mirrors JointLogLikelihoodSharedZetaPLPBF exactly except index 2 is log10(r)
(r = tau_2/tau_1 at 1 GHz) instead of log10(s_i); alpha stays tied (UNCLAMPED,
shared beta); container is FRBParamsTwoScreen; models are FRBModelTwoScreen.

One extra parameter vs the production 8-vector (r), matching the PL-PBF / free-alpha
footing, so production is the nested r -> 0 (log10_r -> -inf) limit and
ln Z(two-screen) - ln Z(production) is a valid Bayes factor. Picklable (module-level,
holds two models) for dynesty.pool.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scat_analysis.burstfit import FRBModel

from twoscreen import FRBModelTwoScreen, FRBParamsTwoScreen


class JointLogLikelihoodSharedZetaTwoScreen:
    def __init__(self, model_C: FRBModel, model_D: FRBModel):
        # upgrade the prepared models in place (no new fields / __init__)
        model_C.__class__ = FRBModelTwoScreen
        model_D.__class__ = FRBModelTwoScreen
        self.model_C = model_C
        self.model_D = model_D

    @staticmethod
    def _band_ll(
        model: FRBModel,
        tau: float,
        beta: float,
        r: float,
        z1: float,
        x: float,
        t0: float,
        ddm: float,
    ) -> float:
        zeta_nu = z1 * np.asarray(model.freq, dtype=float) ** x  # full channel axis
        p = FRBParamsTwoScreen(
            c0=1.0,
            t0=t0,
            gamma=0.0,
            zeta=zeta_nu,
            tau_1ghz=tau,
            beta=beta,
            delta_dm=ddm,
            r=r,
        )
        return model.log_likelihood_gain_marginal(p, "M3")

    def __call__(self, theta: NDArray[np.floating]) -> float:
        tau, beta, log10_r, z1, x = (float(theta[i]) for i in range(5))
        r = 10.0 ** log10_r
        ll = self._band_ll(
            self.model_C, tau, beta, r, z1, x, float(theta[5]), float(theta[6])
        ) + self._band_ll(
            self.model_D, tau, beta, r, z1, x, float(theta[7]), float(theta[8])
        )
        return ll if np.isfinite(ll) else -1e100
