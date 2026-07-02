"""Regression tests for the beta-coherent 2D scattering forward model."""

import numpy as np

from scattering.scat_analysis.burstfit import FRBModel, FRBParams, analytic_gaussian_exp_convolution
from scattering.scat_analysis.turbulence import KOLMOGOROV_BETA, alpha_from_beta


def test_frbparams_alpha_is_derived_from_beta():
    p = FRBParams(c0=1.0, t0=0.0, gamma=0.0, beta=4.0)
    assert p.alpha == 4.0
    p2 = FRBParams(c0=1.0, t0=0.0, gamma=0.0, beta=KOLMOGOROV_BETA)
    assert np.isclose(p2.alpha, 4.4)


def test_m3_couples_pbf_and_scaling_at_beta_four():
    """At beta=4 the model must use the exponential PBF and alpha=4 scaling."""
    time = np.linspace(-2, 18, 400)
    freq = np.array([1.0, 1.4])
    model = FRBModel(time=time, freq=freq, dm_init=0.0)
    p = FRBParams(c0=1.0, t0=5.0, gamma=0.0, zeta=0.2, tau_1ghz=1.0, beta=4.0, delta_dm=0.0)
    spec = model(p, "M3")
    assert spec.shape == (2, time.size)
    assert np.all(np.isfinite(spec))


def test_m3_powerlaw_pbf_used_below_beta_four():
    """Kolmogorov beta uses the power-law PBF branch, not a decoupled env var."""
    time = np.linspace(-2, 18, 400)
    freq = np.array([1.0])
    model = FRBModel(time=time, freq=freq, dm_init=0.0)
    p = FRBParams(
        c0=1.0, t0=5.0, gamma=0.0, zeta=0.2, tau_1ghz=1.0, beta=KOLMOGOROV_BETA, delta_dm=0.0
    )
    spec_pl = model(p, "M3")
    p_exp = FRBParams(
        c0=1.0, t0=5.0, gamma=0.0, zeta=0.2, tau_1ghz=1.0, beta=4.0, delta_dm=0.0
    )
    spec_exp = model(p_exp, "M3")
    # Coupled model: Kolmogorov (beta=11/3) and exponential (beta=4) differ in shape+scaling.
    assert not np.allclose(spec_pl, spec_exp, rtol=1e-3)


def test_m3_sequence_uses_beta():
    p = FRBParams(c0=1.0, t0=2.0, gamma=-1.0, zeta=0.1, tau_1ghz=0.5, beta=3.5, delta_dm=0.0)
    seq = p.to_sequence("M3")
    restored = FRBParams.from_sequence(seq, "M3")
    assert np.isclose(restored.beta, 3.5)
    assert np.isclose(restored.alpha, alpha_from_beta(3.5))


def test_fixed_alpha_bounds_pin_beta_not_full_range():
    """A degenerate alpha window must pin beta, not silently widen to (2, 4].

    An exponential PBF is uniquely beta = 4, so an EMG / fixed-alpha=4 request
    must map to beta fixed at 4 (docs/adr/0006 rationale addendum).
    """
    from scattering.scat_analysis.turbulence import beta_bounds_from_alpha_bounds

    assert beta_bounds_from_alpha_bounds((4.0, 4.0)) == (4.0, 4.0)
    lo, hi = beta_bounds_from_alpha_bounds((4.4, 4.4))
    assert np.isclose(lo, KOLMOGOROV_BETA) and lo == hi
    # alpha < 4 is unreachable on the thin-screen branch: clamp to the exp limit.
    assert beta_bounds_from_alpha_bounds((3.0, 3.0)) == (4.0, 4.0)
    # Non-degenerate windows keep the existing mapping.
    assert beta_bounds_from_alpha_bounds((4.0, 6.0)) == (3.0, 4.0)


def test_alpha_prior_alias_transforms_sigma_and_tolerates_fixed():
    """The deprecated alpha alias must Jacobian-convert sigma and must not
    crash on the (mu, None) fixed-alpha encoding."""
    from scattering.scat_analysis.burstfit import apply_physical_priors, gaussian_prior

    lp_alias = apply_physical_priors(0.0, [3.6], ["beta"], alpha_prior=(4.4, 0.6))
    sigma_beta = 0.6 * 4.0 / (4.4 - 2.0) ** 2
    lp_direct = gaussian_prior(3.6, KOLMOGOROV_BETA, sigma_beta)
    assert np.isclose(lp_alias, lp_direct)
    # Fixed-alpha encoding: no Gaussian factor, no TypeError.
    assert apply_physical_priors(0.0, [4.0], ["beta"], alpha_prior=(4.0, None)) == 0.0


def test_mixed_order_samples_beta_not_alpha():
    """Mixed multi-component M3 entries must sample beta_{i}: the mixed
    likelihood reads get_p('beta') per component, so a legacy alpha_{i} entry
    was sampled but never read."""
    from scattering.scat_analysis.burstfit import FRBFitter

    time = np.linspace(-2, 18, 64)
    freq = np.array([1.0])
    model = FRBModel(time=time, freq=freq, dm_init=0.0)
    fitter = FRBFitter(model, priors={}, components=["M0", "M3"])
    order = fitter.custom_order["mixed"]
    assert "beta_2" in order
    assert not any(name.startswith("alpha_") for name in order)


def test_nested_loglike_alias_transforms_sigma_and_tolerates_fixed():
    """The dynesty likelihood's alpha alias must match apply_physical_priors:
    Jacobian-converted sigma, and no Gaussian factor for (mu, None)."""
    from scattering.scat_analysis.burstfit_nested import _LogLikelihood

    time = np.linspace(-2, 18, 64)
    freq = np.array([1.0])
    model = FRBModel(time=time, freq=freq, dm_init=0.0)
    ll = _LogLikelihood(model, "M3", ("beta",), alpha_prior=(4.4, 0.6))
    assert np.isclose(ll.beta_prior[0], KOLMOGOROV_BETA)
    assert np.isclose(ll.beta_prior[1], 0.6 * 4.0 / (4.4 - 2.0) ** 2)
    ll_fixed = _LogLikelihood(model, "M3", ("beta",), alpha_prior=(4.0, None))
    assert ll_fixed.beta_prior is None


def test_mle_refine_speaks_beta_not_alpha(caplog):
    """MLE init refinement must optimize beta (FRBParams has no alpha field);
    the legacy alpha version TypeError'd and was silently swallowed by the
    broad except, disabling refinement on every run."""
    import logging

    from scattering.scat_analysis.pipeline.optimization import refine_initial_guess_mle

    time = np.linspace(0.0, 20.0, 256)
    freq = np.array([1.0, 1.4])
    truth = FRBParams(c0=5.0, t0=8.0, gamma=0.0, zeta=0.3, tau_1ghz=1.0, beta=4.0)
    clean = FRBModel(time=time, freq=freq, dm_init=0.0)(truth, "M3")
    rng = np.random.default_rng(0)
    data = clean + rng.normal(0.0, 0.05, clean.shape)
    model = FRBModel(time=time, freq=freq, data=data, dm_init=0.0)
    init = FRBParams(c0=4.0, t0=7.5, gamma=0.0, zeta=0.3, tau_1ghz=0.7, beta=3.8)

    with caplog.at_level(logging.WARNING):
        out = refine_initial_guess_mle(model, init)

    assert not any("failed" in rec.message for rec in caplog.records)
    assert 2.0 < out.beta <= 4.0


def test_multicomp_order_has_beta_prior_from_core_translation():
    """M3_multi order includes 'beta'; the core translation helper must emit a
    matching prior entry (missing 'beta' previously KeyError'd walker init)."""
    from scattering.scat_analysis.pipeline.core import _beta_controls

    bounds, prior = _beta_controls(None, 4.4, 0.6)
    assert bounds[0] >= 2.0 and bounds[1] <= 4.0 and bounds[0] < bounds[1]
    assert prior == (4.4, 0.6)
    pin, none_prior = _beta_controls(4.0, 4.4, 0.6)
    assert pin == (4.0, 4.0) and none_prior is None
