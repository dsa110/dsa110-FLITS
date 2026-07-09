"""Posterior-predictive check via replicated data (ADR-0008 Gate 4).

A posterior-predictive check replicates datasets from the posterior and compares
a summary statistic's replicated distribution to the observed value. This is
*not* a goodness-of-fit χ² at the best-fit point — it probes whether the
posterior's *implied data distribution* is consistent with the observation, a
failure mode the point-estimate χ² cannot see (an over-confident posterior on a
misspecified model).

The summary statistic is caller-supplied because it is model-family-dependent
(ADR-0008: e.g. secondary-pulse residual power, ACF Lorentzian width, or the
per-band τ ratio — chosen to probe the model family under test). This module
provides the replication and comparison machinery.

Correctness criterion (tested in tests/test_ppc.py):
- A well-specified model yields a two-tailed posterior-predictive p-value in
  [0.025, 0.975] (the observed summary falls within the central 95% of the
  replicated distribution).
- A misspecified model yields p outside that range (the observed summary is in
  a tail of the replicated distribution).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_N_REPLICATES = 500
PPC_LO = 0.025  # two-tailed central-95% acceptance band
PPC_HI = 0.975


@dataclass(frozen=True)
class PPCResult:
    """Posterior-predictive check outcome for one summary statistic.

    ``p_value`` is the CDF percentile of the observed statistic within the
    replicated distribution: ``P(T_rep <= T_obs)``. A well-specified model
    places the observed near the center (p ~ 0.5); a misspecified model places
    it in a tail (p near 0 or 1). The fit passes Gate 4 when
    ``PPC_LO <= p_value <= PPC_HI`` (the observed falls within the central 95%
    of the replicated distribution).
    """

    observed: float
    replicated: np.ndarray  # shape (n_replicates,)
    p_value: float
    pass_gate: bool

    def asdict(self) -> dict:
        return {
            "observed": self.observed,
            "replicated_mean": float(np.mean(self.replicated)),
            "replicated_p2_5": float(np.percentile(self.replicated, 2.5)),
            "replicated_p97_5": float(np.percentile(self.replicated, 97.5)),
            "p_value": self.p_value,
            "pass_gate": self.pass_gate,
        }


def replicate_dataset(
    model_draw: np.ndarray,
    noise_std: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One replicated dataset: model evaluated at a posterior draw + Gaussian noise.

    ``model_draw`` is the model spectrum (n_freq, n_time) at one parameter draw.
    ``noise_std`` is per-frequency-channel noise std (n_freq,), broadcast over
    time. Returns a noisy realization with the same shape as ``model_draw``.
    """
    noise = rng.normal(0.0, noise_std[:, None], size=model_draw.shape)
    return model_draw + noise


def posterior_predictive_check(
    *,
    observed: np.ndarray,
    model_draws: np.ndarray,
    noise_std: np.ndarray,
    summary: Callable[[np.ndarray], float],
    n_replicates: int = DEFAULT_N_REPLICATES,
    rng: np.random.Generator | None = None,
) -> PPCResult:
    """Run a posterior-predictive check.

    Parameters
    ----------
    observed : (n_freq, n_time) array
        The observed data spectrum.
    model_draws : (n_samples, n_freq, n_time) array
        Model spectra evaluated at each posterior draw. The caller draws
        posterior samples and evaluates the forward model at each; this module
        does not own the model (it is model-family-agnostic).
    noise_std : (n_freq,) array
        Per-channel noise standard deviation, broadcast over time.
    summary : callable
        Maps a (n_freq, n_time) spectrum to a scalar summary statistic. Chosen
        by the caller to probe the model family under test (ADR-0008).
    n_replicates : int
        Number of replicated datasets to generate. Defaults to 500.
    rng : optional
        Random generator. A fresh one is created if not supplied.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_samples = model_draws.shape[0]
    if n_samples == 0:
        raise ValueError("model_draws must contain at least one posterior draw")
    draw_idx = rng.choice(n_samples, size=n_replicates, replace=True)
    t_obs = float(summary(observed))
    t_rep = np.empty(n_replicates, dtype=float)
    for i, di in enumerate(draw_idx):
        rep = replicate_dataset(model_draws[di], noise_std, rng)
        t_rep[i] = summary(rep)
    # Posterior-predictive p-value: CDF percentile of the observed within the
    # replicated distribution. Center (p~0.5) = well-specified; tails (p~0 or
    # ~1) = misspecified. Passes when within the central 95% [PPC_LO, PPC_HI].
    n = n_replicates
    p_value = float(np.sum(t_rep <= t_obs)) / n
    return PPCResult(
        observed=t_obs,
        replicated=t_rep,
        p_value=p_value,
        pass_gate=PPC_LO <= p_value <= PPC_HI,
    )
