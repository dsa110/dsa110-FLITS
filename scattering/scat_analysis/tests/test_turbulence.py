"""Tests for the beta-coherent turbulence helpers."""

import numpy as np
import pytest

from scattering.scat_analysis.turbulence import (
    KOLMOGOROV_BETA,
    alpha_from_beta,
    beta_bounds_from_alpha_bounds,
    beta_from_alpha_thin_screen,
    default_joint_beta_bounds,
)


def test_kolmogorov_alpha():
    assert np.isclose(alpha_from_beta(KOLMOGOROV_BETA), 4.4, rtol=1e-6)


def test_exponential_limit():
    assert alpha_from_beta(4.0) == 4.0


def test_alpha_beta_roundtrip_thin_screen():
    for alpha in (4.0, 4.4, 5.0, 6.0):
        beta = beta_from_alpha_thin_screen(alpha)
        assert np.isclose(alpha_from_beta(beta), alpha, rtol=1e-10)


def test_default_joint_bounds_map_legacy_alpha():
    blo, bhi = default_joint_beta_bounds()
    assert blo < bhi
    assert np.isclose(alpha_from_beta(bhi), 4.0, rtol=1e-6)
    assert np.isclose(alpha_from_beta(blo), 6.0, rtol=1e-6)


def test_beta_bounds_from_alpha_bounds_ordering():
    blo, bhi = beta_bounds_from_alpha_bounds((4.0, 6.0))
    assert blo < bhi


def test_alpha_from_beta_rejects_invalid():
    with pytest.raises(ValueError):
        alpha_from_beta(2.0)
