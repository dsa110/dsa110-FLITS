"""Sampling utilities for FRB models."""
from __future__ import annotations

import numpy as np
import emcee

from .models import FRBModel
from .params import FRBParams



def _log_prob_wrapper(
    theta: np.ndarray,
    t: np.ndarray,
    freqs: np.ndarray,
    data: np.ndarray,
    noise_std: float,
    t0: float = 0.0,
    width: float = 1.0,
) -> float:
    """Log-probability for sampling FRB parameters.

    This function combines a simple Gaussian likelihood with uniform priors on
    ``dm`` and ``amplitude`` (both must be positive).
    """
    dm, amp = theta
    if dm < 0 or amp < 0 or width <= 0:
        return -np.inf

    params = FRBParams(dm=dm, amplitude=amp, t0=t0, width=width)
    model = FRBModel(params)
    model_spec = model.simulate(t, freqs)
    resid = data - model_spec
    return -0.5 * np.sum((resid / noise_std) ** 2)


class FRBFitter:
    """Fit :class:`FRBModel` parameters using ``emcee`` MCMC."""

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

    def sample(
        self,
        initial: np.ndarray,
        nwalkers: int = 32,
        nsteps: int = 100,
        **kwargs,
    ) -> emcee.EnsembleSampler:
        """Run the MCMC sampler and return the ``emcee`` sampler instance."""
        ndim = len(initial)
        p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            _log_prob_wrapper,
            args=(self.t, self.freqs, self.data, self.noise_std),
        )
        sampler.run_mcmc(p0, nsteps, progress=False, **kwargs)
        self.sampler = sampler
        return sampler


__all__ = ["FRBFitter", "_log_prob_wrapper"]
