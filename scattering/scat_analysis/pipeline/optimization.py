"""
Optimization and convergence utilities for the BurstFit pipeline.
"""
from __future__ import annotations

import logging
import warnings
import numpy as np
from scipy.optimize import minimize

from ..burstfit import FRBParams, gelman_rubin

log = logging.getLogger(__name__)

def refine_initial_guess_mle(model, init_guess: FRBParams) -> FRBParams:
    """
    Use MLE (Nelder-Mead) to refine the initial guess before MCMC.

    Optimizes primarily for tau_1ghz, alpha, t0, and c0.
    Keeps nuisance parameters (gamma, delta_dm) fixed or tightly constrained.
    """
    log.info("Refining initial guess via MLE (Nelder-Mead)...")

    # Parameters to float: [tau, alpha, t0, c0]
    # We work in log-space for positive params to ensure positivity
    x0 = [
        np.log(max(init_guess.tau_1ghz, 1e-4)),  # log tau
        init_guess.alpha,  # alpha (linear)
        init_guess.t0,  # t0 (linear)
        np.log(max(init_guess.c0, 1e-4)),  # log c0
    ]

    def obj_func(theta):
        ln_tau, alpha, t0, ln_c0 = theta

        # Constraints
        # Constrain alpha to physically reasonable range for thin screen
        if not (0.1 < alpha < 5.0):
            return 1e20 

        tau_val = np.exp(ln_tau)
        c0_val = np.exp(ln_c0)

        # Build params
        p = FRBParams(
            c0=c0_val,
            t0=t0,
            gamma=init_guess.gamma,  # Fixed
            zeta=init_guess.zeta,  # Fixed
            tau_1ghz=tau_val,
            alpha=alpha,
            delta_dm=init_guess.delta_dm,  # Fixed
        )

        # Negative Log Likelihood
        # Add simple priors to prevent runaway
        nll = -model.log_likelihood(p, "M3")
        return nll

    try:
        res = minimize(
            obj_func, x0, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-2}
        )

        if res.success or res.message:
            log.info(f"MLE Refinement finished: {res.message}")

            ln_tau, alpha, t0, ln_c0 = res.x
            refined_params = FRBParams(
                c0=np.exp(ln_c0),
                t0=t0,
                gamma=init_guess.gamma,
                zeta=init_guess.zeta,
                tau_1ghz=np.exp(ln_tau),
                alpha=alpha,
                delta_dm=init_guess.delta_dm,
            )

            log.info(
                f"  tau:   {init_guess.tau_1ghz:.3f} -> {refined_params.tau_1ghz:.3f} ms"
            )
            log.info(f"  alpha: {init_guess.alpha:.3f} -> {refined_params.alpha:.3f}")
            log.info(f"  t0:    {init_guess.t0:.3f} -> {refined_params.t0:.3f} ms")
            return refined_params
        else:
            log.warning("MLE refinement did not converge, using original guess.")
            return init_guess

    except Exception as e:
        log.warning(f"MLE refinement failed with error: {e}. using original guess.")
        return init_guess


def auto_burn_thin(sampler, safety_factor_burn=3.0, safety_factor_thin=0.5):
    """Automatically determine burn-in and thinning based on autocorrelation time.

    Also computes Gelman-Rubin R̂ for convergence diagnostics.

    Returns
    -------
    tuple
        (burn, thin, convergence_info) where convergence_info is a dict with R̂ values.
    """
    burn = sampler.iteration // 4  # default fallback
    thin = 1
    convergence_info = {}

    try:
        tau = sampler.get_autocorr_time(tol=0.01)
        burn = int(safety_factor_burn * np.nanmax(tau))
        thin = max(1, int(safety_factor_thin * np.nanmin(tau)))
        burn = min(burn, sampler.iteration // 2)
        log.info(f"Auto-determined burn-in: {burn}, thinning: {thin}")
        convergence_info["autocorr_time"] = tau.tolist()
    except Exception as e:
        warnings.warn(f"Could not estimate autocorr time: {e}. Using defaults.")

    # Compute Gelman-Rubin R̂
    try:
        rhat_results = gelman_rubin(sampler, discard=burn)
        convergence_info.update(rhat_results)
        if rhat_results["converged"]:
            log.info(f"Gelman-Rubin R̂ max = {rhat_results['max_rhat']:.4f} (CONVERGED)")
        else:
            log.warning(
                f"Gelman-Rubin R̂ max = {rhat_results['max_rhat']:.4f} (NOT CONVERGED - consider more steps)"
            )
    except Exception as e:
        warnings.warn(f"Could not compute Gelman-Rubin: {e}")
        convergence_info["gelman_rubin_error"] = str(e)

    return burn, thin, convergence_info
