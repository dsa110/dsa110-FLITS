"""Sampling utilities for FRB models."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import emcee

from .models import FRBModel
from .params import FRBParams
from .fitting import VALIDATION_THRESHOLDS as VT


@dataclass
class MCMCDiagnostics:
    """Diagnostics from MCMC sampling."""
    converged: bool
    autocorr_time: np.ndarray
    burn_in: int
    acceptance_fraction: float
    neff_mean: float
    quality_flag: str  # "PASS", "MARGINAL", "FAIL"
    validation_notes: list


def _log_prob_wrapper(
    theta: np.ndarray,
    t: np.ndarray,
    freqs: np.ndarray,
    data: np.ndarray,
    noise_std: float,
    t0: float = 0.0,
    width: float = 1.0,
) -> float:
    """Log-probability for sampling FRB parameters."""
    dm, amp = theta

    # GATE 1: PHYSICAL BOUNDS
    if dm < VT.DM_MIN or dm > VT.DM_MAX:
        return -np.inf
    if amp < VT.AMP_MIN or amp > VT.AMP_MAX:
        return -np.inf
    if width < VT.WIDTH_MIN or width > VT.WIDTH_MAX:
        return -np.inf

    # GATE 2: MODEL EVALUATION
    try:
        params = FRBParams(dm=dm, amplitude=amp, t0=t0, width=width)
        model = FRBModel(params)
        model_spec = model.simulate(t, freqs)
    except Exception:
        return -np.inf

    # GATE 3: SANITY CHECK
    resid = data - model_spec

    if np.any(~np.isfinite(resid)):
        return -np.inf

    # Compute reduced chi-squared
    dof = data.size - len(theta)
    chi_sq = np.sum((resid / noise_std) ** 2)
    red_chi_sq = chi_sq / dof

    # Reject catastrophic misfits
    if red_chi_sq > VT.RED_CHI_SQ_CATASTROPHIC:
        return -np.inf

    # GATE 4: LIKELIHOOD
    log_likelihood = -0.5 * chi_sq

    return log_likelihood


class FRBFitter:
    """Fit :class:`FRBModel` parameters using ``emcee`` MCMC with convergence validation."""

    def __init__(
        self,
        t: np.ndarray,
        freqs: np.ndarray,
        data: np.ndarray,
        noise_std: float = 1.0,
    ) -> None:
        self.t = np.asarray(t)
        self.freqs = np.asarray(freqs)
        self.data = np.asarray(data)
        self.noise_std = float(noise_std)
        self.sampler: emcee.EnsembleSampler | None = None
        self.diagnostics: MCMCDiagnostics | None = None
        self.burn_in = 0

    def sample(
        self,
        initial: np.ndarray,
        nwalkers: int = 32,
        nsteps: int = 100,
        burn_in_factor: float = VT.MCMC_BURN_IN_FACTOR,
        **kwargs,
    ) -> tuple[emcee.EnsembleSampler, MCMCDiagnostics]:
        """Run MCMC with convergence validation.

        Returns
        -------
        sampler : emcee.EnsembleSampler
        diagnostics : MCMCDiagnostics
        """
        ndim = len(initial)
        p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            _log_prob_wrapper,
            args=(self.t, self.freqs, self.data, self.noise_std),
        )

        sampler.run_mcmc(p0, nsteps, progress=False, **kwargs)

        # VALIDATION: Compute convergence metrics
        try:
            tau = sampler.get_autocorr_time(quiet=True)
        except Exception as e:
            print(f"WARNING: Could not compute autocorrelation time: {e}")
            tau = np.full(ndim, np.inf)

        # Check convergence
        converged = np.all(tau < nsteps / VT.MCMC_AUTOCORR_NSTEPS_FACTOR)

        if not converged:
            print(f"⚠️ WARNING: Chains may not be fully converged")
            print(f"   Max autocorr time: {np.max(tau):.1f}")
            print(f"   Recommend: nsteps >= {int(np.max(tau) * VT.MCMC_AUTOCORR_NSTEPS_FACTOR)}")

        # Compute burn-in
        burn_in = int(np.max(tau) * burn_in_factor)
        burn_in = min(burn_in, nsteps // 2)

        # Check acceptance fraction
        acc_frac = sampler.acceptance_fraction.mean()
        if acc_frac < VT.MCMC_ACC_FRAC_MIN or acc_frac > VT.MCMC_ACC_FRAC_MAX:
            print(f"⚠️ WARNING: Acceptance fraction {acc_frac:.3f} is unusual")

        # Compute effective sample size
        flat_samples = sampler.get_chain(discard=burn_in, flat=True)
        neff_mean = flat_samples.shape[0] / np.mean(tau) if np.mean(tau) > 0 else 0

        # Assign quality flag
        validation_notes = []
        quality_flag = "PASS"

        if not converged:
            quality_flag = "MARGINAL"
            validation_notes.append(f"Chains may not be converged (max τ={np.max(tau):.1f})")

        if acc_frac < VT.MCMC_ACC_FRAC_MIN or acc_frac > VT.MCMC_ACC_FRAC_MAX:
            quality_flag = "MARGINAL"
            validation_notes.append(f"Unusual acceptance fraction ({acc_frac:.3f})")

        if neff_mean < ndim * 10:
            quality_flag = "MARGINAL"
            validation_notes.append(f"Low effective sample size ({neff_mean:.0f})")

        # Check for degenerate solutions
        chain = sampler.get_chain(discard=burn_in)
        param_stds = np.std(chain, axis=0)
        if np.any(param_stds < 1e-6):
            quality_flag = "FAIL"
            validation_notes.append("Parameter(s) have near-zero variance")

        diagnostics = MCMCDiagnostics(
            converged=bool(converged),
            autocorr_time=tau,
            burn_in=burn_in,
            acceptance_fraction=float(acc_frac),
            neff_mean=float(neff_mean),
            quality_flag=quality_flag,
            validation_notes=validation_notes,
        )

        self.sampler = sampler
        self.diagnostics = diagnostics
        self.burn_in = burn_in

        # Print summary
        print("=" * 80)
        print("MCMC CONVERGENCE DIAGNOSTICS")
        print("=" * 80)
        print(f"Status: {quality_flag}")
        print(f"Converged: {converged}")
        print(f"Acceptance fraction: {acc_frac:.3f}")
        print(f"Mean autocorr time: {np.mean(tau):.1f} steps")
        print(f"Burn-in: {burn_in} steps")
        print(f"Effective sample size: {neff_mean:.0f}")
        if validation_notes:
            print("Notes:")
            for note in validation_notes:
                print(f"  - {note}")
        print("=" * 80)

        return sampler, diagnostics


__all__ = ["FRBFitter", "_log_prob_wrapper", "MCMCDiagnostics"]
