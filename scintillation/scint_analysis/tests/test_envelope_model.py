"""Tests for the P4 envelope models (exploratory class, P4 record).

Correctness invariants:

* each family recovers a smooth synthetic envelope well below the residual
  noise floor;
* the multiplicative residual returns an injected fine-scale signal exactly
  when the true envelope is supplied, and to good accuracy when the fitted
  envelope is much smoother than the signal;
* the clip rule masks (rather than divides by) a vanishing envelope.
"""

from __future__ import annotations

import numpy as np
import pytest

from scint_analysis import envelope_model as em


def _grid(n: int = 4096):
    nu = np.linspace(627.0, 800.0, n)
    good = np.ones(n, dtype=bool)
    good[500:520] = False
    rng = np.random.default_rng(11)
    return nu, good, rng


def _smooth_envelope(nu: np.ndarray) -> np.ndarray:
    # ~20 MHz undulations, order-unity: a stand-in for intrinsic structure
    return 1.0 + 0.5 * np.sin(2 * np.pi * (nu - nu[0]) / 40.0) + 0.2 * np.cos(
        2 * np.pi * (nu - nu[0]) / 21.0
    )


@pytest.mark.parametrize(
    "family,scale",
    [("M1_spline", 5.0), ("M2_gp", 5.0), ("M3_delaycut", 50)],
)
def test_families_recover_smooth_envelope(family, scale):
    nu, good, rng = _grid()
    envelope_true = _smooth_envelope(nu)
    noise = 0.01 * rng.standard_normal(nu.size)
    spectrum = np.where(good, envelope_true + noise, np.nan)
    chain = em.EnvelopeChain(family, scale, nu, good, noise_variance=1e-4)
    fitted = chain.envelope(spectrum)
    err = np.nanstd((fitted - envelope_true)[good])
    assert err < 0.02  # envelope recovered below the injected noise level


def test_residual_exact_with_true_envelope():
    nu, good, rng = _grid()
    envelope_true = _smooth_envelope(nu) + 1.0
    signal = 0.01 * rng.standard_normal(nu.size)  # fine-scale multiplicative
    spectrum = envelope_true * (1.0 + signal)
    r = em.residual(spectrum, envelope_true)
    assert np.nanmax(np.abs(r - signal)) < 1e-12


def test_residual_preserves_fine_signal_through_fit():
    nu, good, rng = _grid()
    envelope_true = _smooth_envelope(nu) + 1.0
    # ~0.2 MHz correlated signal: far below the 5 MHz smoothing scale
    phase = np.cumsum(rng.standard_normal(nu.size)) * 0.5
    signal = 0.02 * np.sin(2 * np.pi * (nu - nu[0]) / 0.2 + phase)
    spectrum = np.where(good, envelope_true * (1.0 + signal), np.nan)
    chain = em.EnvelopeChain("M1_spline", 5.0, nu, good, noise_variance=1e-4)
    r = chain.residual(spectrum)
    ok = np.isfinite(r)
    correlation = np.corrcoef(r[ok], signal[ok])[0, 1]
    assert correlation > 0.95  # signal survives the subtraction
    assert np.nanstd(r[ok] - signal[ok]) < 0.3 * np.nanstd(signal[ok])


def test_clip_rule_masks_vanishing_envelope():
    spectrum = np.ones(100)
    envelope = np.ones(100)
    envelope[:4] = 1e-9  # low tail below the 5th-percentile floor
    r = em.residual(spectrum, envelope)
    assert np.isnan(r[:4]).all()
    assert np.isfinite(r[4:]).all()


@pytest.mark.parametrize(
    "family,scale",
    [("M1_spline", 5.0), ("M2_gp", 5.0), ("M3_delaycut", 50)],
)
def test_batched_fit_matches_looped(family, scale):
    nu, good, rng = _grid(1024)
    base = _smooth_envelope(nu)
    batch = np.stack(
        [np.where(good, base + 0.01 * rng.standard_normal(nu.size), np.nan) for _ in range(3)]
    )
    chain = em.EnvelopeChain(family, scale, nu, good, noise_variance=1e-4)
    batched = chain.residual(batch)
    for row_in, row_out in zip(batch, batched):
        single = chain.residual(row_in)
        np.testing.assert_allclose(row_out, single, rtol=1e-10, atol=1e-12)


def test_spline_survives_wide_masked_gap():
    # ~30 MHz hole (the LTE exclusion) with knot spacing far below the gap
    nu = np.linspace(627.0, 800.0, 4096)
    good = (nu < 730.0) | (nu > 760.0)
    envelope_true = _smooth_envelope(nu)
    spectrum = np.where(good, envelope_true, np.nan)
    fitted = em.fit_spline(nu, spectrum, 0.5)
    assert np.all(np.isfinite(fitted))
    assert np.nanmax(np.abs((fitted - envelope_true)[good])) < 0.05


def test_unknown_family_raises():
    nu, good, _ = _grid(256)
    with pytest.raises(ValueError):
        em.EnvelopeChain("M9", 1.0, nu, good, noise_variance=1e-4)
