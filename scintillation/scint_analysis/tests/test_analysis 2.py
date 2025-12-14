"""
test_analysis.py
================

Unit tests for scint_analysis/analysis.py - ACF calculation and fitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the parent directories to path for imports
_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scint_analysis.analysis import (
    calculate_acf,
    calculate_acf_noerrs,
    lorentzian_component,
    gaussian_component,
    lorentzian_generalised,
    power_law_model,
    _interpret_scaling_index,
    _noise_descriptor_hash,
    clear_noise_acf_cache,
)
from scint_analysis.core import ACF


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def white_noise_spectrum():
    """White noise spectrum for testing."""
    np.random.seed(42)
    return np.ma.MaskedArray(np.random.normal(100, 10, 256))


@pytest.fixture
def correlated_spectrum():
    """Spectrum with frequency correlations (simulated scintillation)."""
    np.random.seed(42)
    # Generate correlated Gaussian noise
    n = 256
    # Simple correlation via moving average
    white = np.random.normal(0, 1, n + 10)
    correlated = np.convolve(white, np.ones(10) / 10, mode='valid')[:n]
    # Scale to positive intensity values
    spectrum = 100 + 20 * correlated
    return np.ma.MaskedArray(spectrum)


# ============================================================================
# Model Function Tests
# ============================================================================

class TestModelFunctions:
    """Tests for ACF model functions."""

    def test_lorentzian_peak(self):
        """Test Lorentzian peak properties."""
        x = np.linspace(-10, 10, 101)
        gamma = 2.0
        m = 1.0
        
        y = lorentzian_component(x, gamma, m)
        
        # Peak should be at x=0
        assert np.argmax(y) == 50
        # Peak value should be m^2
        assert_allclose(y[50], m ** 2)
        # HWHM should be at gamma
        half_max = m ** 2 / 2
        hwhm_idx = np.argmin(np.abs(y[:50] - half_max))
        hwhm = np.abs(x[hwhm_idx])
        assert_allclose(hwhm, gamma, rtol=0.1)

    def test_gaussian_peak(self):
        """Test Gaussian peak properties."""
        x = np.linspace(-10, 10, 101)
        sigma = 2.0
        m = 1.0
        
        y = gaussian_component(x, sigma, m)
        
        # Peak should be at x=0
        assert np.argmax(y) == 50
        # Peak value should be m^2
        assert_allclose(y[50], m ** 2)

    def test_lorentzian_generalised_reduces_to_standard(self):
        """Test that generalized Lorentzian with alpha=0 gives standard Lorentzian."""
        x = np.linspace(-10, 10, 101)
        gamma = 2.0
        m = 1.0
        
        y_standard = lorentzian_component(x, gamma, m)
        y_general = lorentzian_generalised(x, gamma, alpha=0.0, m=m)
        
        assert_allclose(y_standard, y_general, rtol=1e-6)

    def test_power_law(self):
        """Test power law function."""
        x = np.array([1.0, 2.0, 4.0, 8.0])
        c = 1.0
        n = -2.0
        
        y = power_law_model(x, c, n)
        
        # Should scale as x^n
        assert_allclose(y, c * x ** n, rtol=1e-6)

    def test_power_law_handles_zero(self):
        """Test power law handles x=0 gracefully."""
        x = np.array([0.0, 1.0, 2.0])
        
        y = power_law_model(x, c=1.0, n=-2.0)
        
        # Should not have NaN or Inf
        assert np.all(np.isfinite(y))


# ============================================================================
# ACF Calculation Tests
# ============================================================================

class TestACFCalculation:
    """Tests for ACF calculation functions."""

    def test_acf_returns_object(self, white_noise_spectrum):
        """Test that calculate_acf returns an ACF object."""
        result = calculate_acf(white_noise_spectrum, channel_width_mhz=0.39)
        
        assert result is not None
        assert isinstance(result, ACF)

    def test_acf_symmetry(self, correlated_spectrum):
        """Test that ACF is symmetric."""
        result = calculate_acf(correlated_spectrum, channel_width_mhz=0.39)
        
        # ACF should be symmetric around lag=0
        n = len(result.lags)
        mid = n // 2
        assert_allclose(result.acf[:mid], result.acf[mid+1:][::-1], rtol=0.1)

    def test_acf_peak_at_zero_lag(self, correlated_spectrum):
        """Test that ACF has maximum at zero lag."""
        result = calculate_acf(correlated_spectrum, channel_width_mhz=0.39)
        
        # Find zero lag
        zero_idx = np.argmin(np.abs(result.lags))
        max_idx = np.argmax(result.acf)
        
        assert zero_idx == max_idx

    def test_acf_includes_errors(self, correlated_spectrum):
        """Test that ACF includes error estimates."""
        result = calculate_acf(correlated_spectrum, channel_width_mhz=0.39)
        
        assert result.err is not None
        assert len(result.err) == len(result.acf)
        assert np.all(result.err > 0)

    def test_acf_noerrs_shape(self, white_noise_spectrum):
        """Test calculate_acf_noerrs output shape."""
        result = calculate_acf_noerrs(white_noise_spectrum, channel_width_mhz=0.39)
        
        assert result is not None
        assert isinstance(result, ACF)

    def test_acf_insufficient_data(self):
        """Test ACF returns None for insufficient data."""
        tiny_spectrum = np.ma.MaskedArray(np.random.rand(5))
        
        result = calculate_acf(tiny_spectrum, channel_width_mhz=0.39)
        
        assert result is None

    def test_acf_max_lag_bins(self, correlated_spectrum):
        """Test max_lag_bins parameter."""
        result_full = calculate_acf(correlated_spectrum, channel_width_mhz=0.39)
        result_limited = calculate_acf(
            correlated_spectrum, channel_width_mhz=0.39, max_lag_bins=20
        )
        
        # Limited should have fewer lags
        assert len(result_limited.lags) < len(result_full.lags)


# ============================================================================
# Scaling Index Interpretation Tests
# ============================================================================

class TestScalingInterpretation:
    """Tests for _interpret_scaling_index function."""

    def test_kolmogorov_scaling(self):
        """Test interpretation of Kolmogorov scaling."""
        result = _interpret_scaling_index(4.0, 0.3)
        
        assert "Kolmogorov" in result or "diffractive" in result.lower()

    def test_refractive_scaling(self):
        """Test interpretation of refractive scaling."""
        result = _interpret_scaling_index(2.0, 0.3)
        
        assert "refractive" in result.lower() or "α ≈ 2" in result

    def test_no_scaling(self):
        """Test interpretation of no scaling."""
        result = _interpret_scaling_index(0.0, 0.3)
        
        assert "no" in result.lower() or "intrinsic" in result.lower()

    def test_unphysical_scaling(self):
        """Test interpretation of unphysical scaling."""
        result = _interpret_scaling_index(-2.0, 0.3)
        
        assert "unphysical" in result.lower() or "negative" in result.lower()

    def test_nan_handling(self):
        """Test handling of NaN values."""
        result = _interpret_scaling_index(np.nan, 0.3)
        
        assert "unable" in result.lower() or "invalid" in result.lower()


# ============================================================================
# Noise Descriptor Hash Tests
# ============================================================================

class TestNoiseDescriptorHash:
    """Tests for noise descriptor hashing."""

    def test_hash_consistency(self):
        """Test that same descriptor gives same hash."""
        from scint_analysis.noise import NoiseDescriptor
        
        desc1 = NoiseDescriptor(
            kind="intensity",
            nt=100, nchan=64,
            mu=1.0, sigma=0.0, shift=0.0,
            gamma_k=1.0, gamma_theta=1.0,
            phi_t=0.1, phi_f=0.1,
            g_t=np.ones(100), b_f=np.ones(64)
        )
        
        hash1 = _noise_descriptor_hash(desc1)
        hash2 = _noise_descriptor_hash(desc1)
        
        assert hash1 == hash2

    def test_different_descriptors_different_hash(self):
        """Test that different descriptors give different hashes."""
        from scint_analysis.noise import NoiseDescriptor
        
        desc1 = NoiseDescriptor(
            kind="intensity",
            nt=100, nchan=64,
            mu=1.0, sigma=0.0, shift=0.0,
            gamma_k=1.0, gamma_theta=1.0,
            phi_t=0.1, phi_f=0.1,
            g_t=np.ones(100), b_f=np.ones(64)
        )
        
        desc2 = NoiseDescriptor(
            kind="intensity",
            nt=100, nchan=64,
            mu=2.0,  # Different mu
            sigma=0.0, shift=0.0,
            gamma_k=1.0, gamma_theta=1.0,
            phi_t=0.1, phi_f=0.1,
            g_t=np.ones(100), b_f=np.ones(64)
        )
        
        assert _noise_descriptor_hash(desc1) != _noise_descriptor_hash(desc2)

    def test_none_descriptor(self):
        """Test hash of None descriptor."""
        assert _noise_descriptor_hash(None) == 0


# ============================================================================
# Cache Tests
# ============================================================================

class TestNoiseACFCache:
    """Tests for noise ACF caching."""

    def test_clear_cache(self):
        """Test that cache can be cleared."""
        # This should not raise
        clear_noise_acf_cache()


# ============================================================================
# Integration Tests
# ============================================================================

class TestACFIntegration:
    """Integration tests for ACF calculation pipeline."""

    def test_lorentzian_acf_recovery(self):
        """Test that Lorentzian ACF parameters can be extracted."""
        # Create synthetic Lorentzian ACF
        gamma_true = 5.0  # MHz
        m_true = 0.8
        
        lags = np.linspace(-20, 20, 81)
        acf_true = lorentzian_component(lags, gamma_true, m_true)
        
        # Add small noise
        np.random.seed(42)
        acf_noisy = acf_true + np.random.normal(0, 0.02, len(acf_true))
        
        # Fit Lorentzian
        from lmfit import Model
        lor_model = Model(lorentzian_component)
        params = lor_model.make_params(gamma=3.0, m=0.5)
        params['gamma'].min = 0.01
        params['m'].min = 0
        
        result = lor_model.fit(acf_noisy, params, x=lags)
        
        # Check recovery
        assert_allclose(result.params['gamma'].value, gamma_true, rtol=0.2)
        assert_allclose(result.params['m'].value, m_true, rtol=0.2)

    def test_masked_data_handling(self):
        """Test that masked data is handled correctly in ACF."""
        np.random.seed(42)
        n = 256
        
        # Create spectrum with some masked channels
        spectrum = np.ma.MaskedArray(np.random.normal(100, 10, n))
        spectrum.mask = np.zeros(n, dtype=bool)
        spectrum.mask[100:110] = True  # Mask 10 channels
        
        result = calculate_acf(spectrum, channel_width_mhz=0.39)
        
        assert result is not None
        assert not np.any(np.isnan(result.acf))

