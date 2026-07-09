"""Tests for the replicated-data posterior-predictive check (ADR-0008 Gate 4).

Correctness criteria:
- A well-specified model (replicates drawn from the same process that generated
  the observation) yields a two-tailed p-value in [0.025, 0.975] — the observed
  summary falls within the central 95% of the replicated distribution.
- A misspecified model (replicates drawn from a different process) yields p
  outside that band — the observed summary is in a tail.
- The p-value is two-tailed and capped at 1.0.
- An empty posterior raises.
"""

import numpy as np

from flits.fitting.ppc import (
    PPC_HI,
    PPC_LO,
    posterior_predictive_check,
    replicate_dataset,
)


def _make_wellspecified(n_freq=20, n_time=64, n_samples=400, seed=0):
    """Observation and replicates drawn from the same Gaussian process."""
    rng = np.random.default_rng(seed)
    truth = np.outer(np.linspace(0.5, 1.5, n_freq), np.exp(-(np.linspace(-2, 4, n_time) ** 2)))
    noise_std = np.full(n_freq, 0.1)
    observed = truth + rng.normal(0.0, noise_std[:, None], size=truth.shape)
    # Posterior draws centered on truth (well-specified).
    draws = np.stack([truth + rng.normal(0.0, 0.02, size=truth.shape) for _ in range(n_samples)])
    return observed, draws, noise_std


def test_wellspecified_model_passes_gate():
    observed, draws, noise_std = _make_wellspecified()
    # Summary: total power (a statistic the model should reproduce).
    res = posterior_predictive_check(
        observed=observed,
        model_draws=draws,
        noise_std=noise_std,
        summary=lambda s: float(np.sum(s)),
        rng=np.random.default_rng(1),
    )
    assert PPC_LO <= res.p_value <= PPC_HI, f"p={res.p_value} outside [{PPC_LO},{PPC_HI}]"
    assert res.pass_gate is True


def test_misspecified_model_fails_gate():
    """Replicates drawn from a process with the wrong amplitude: the observed
    total power sits in a tail of the replicated distribution."""
    observed, draws, noise_std = _make_wellspecified()
    # Corrupt the draws: scale them up so the replicated total power is too high.
    draws_bad = draws * 1.5
    res = posterior_predictive_check(
        observed=observed,
        model_draws=draws_bad,
        noise_std=noise_std,
        summary=lambda s: float(np.sum(s)),
        rng=np.random.default_rng(2),
    )
    assert res.pass_gate is False, f"misspecified model should fail, p={res.p_value}"
    # Observed total power is below the replicated distribution => p near 0.
    assert res.p_value < PPC_LO


def test_pvalue_at_center_passes():
    """When the observed equals the model center, p ~ 0.5 (center of the
    replicated CDF) and the gate passes."""
    rng = np.random.default_rng(3)
    n_freq, n_time = 10, 20
    truth = np.ones((n_freq, n_time))
    noise_std = np.full(n_freq, 0.5)
    observed = truth.copy()  # exactly at the model center
    draws = np.stack([truth + rng.normal(0, 0.01, truth.shape) for _ in range(300)])
    res = posterior_predictive_check(
        observed=observed,
        model_draws=draws,
        noise_std=noise_std,
        summary=lambda s: float(np.sum(s)),
        rng=np.random.default_rng(4),
    )
    assert 0.0 <= res.p_value <= 1.0
    # Observed at center => p ~ 0.5, passes gate.
    assert res.pass_gate is True


def test_replicate_dataset_shape_and_noise():
    rng = np.random.default_rng(5)
    model = np.zeros((4, 2000))
    noise_std = np.array([0.1, 0.2, 0.3, 0.4])
    rep = replicate_dataset(model, noise_std, rng)
    assert rep.shape == model.shape
    # Noise std per channel should match (within sampling tolerance at 2000 samples).
    per_ch_std = rep.std(axis=1)
    assert np.allclose(per_ch_std, noise_std, atol=0.02)


def test_empty_posterior_raises():
    import pytest

    observed = np.zeros((4, 10))
    draws = np.zeros((0, 4, 10))
    noise_std = np.full(4, 0.1)
    with pytest.raises(ValueError):
        posterior_predictive_check(
            observed=observed,
            model_draws=draws,
            noise_std=noise_std,
            summary=lambda s: float(np.sum(s)),
        )


def test_asdict_roundtrip():
    observed, draws, noise_std = _make_wellspecified(n_samples=50)
    res = posterior_predictive_check(
        observed=observed,
        model_draws=draws,
        noise_std=noise_std,
        summary=lambda s: float(np.sum(s)),
        n_replicates=50,
        rng=np.random.default_rng(6),
    )
    d = res.asdict()
    assert "observed" in d and "p_value" in d and "pass_gate" in d
    assert "replicated_p2_5" in d and "replicated_p97_5" in d
