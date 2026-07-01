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
