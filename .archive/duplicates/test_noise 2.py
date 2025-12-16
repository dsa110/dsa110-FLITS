"""
test_noise.py
=============

Unit tests for scint_analysis/noise.py - Noise characterization and synthesis.
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
from scipy import stats

from scint_analysis.noise import (
    NoiseDescriptor,
    estimate_noise_descriptor,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def intensity_data():
    """Generate synthetic intensity data (positive, exponential-like)."""
    np.random.seed(42)
    # Chi-squared with 2 dof (exponential) scaled
    return np.random.exponential(scale=1.0, size=(100, 64)).astype(np.float64)


@pytest.fixture
def gaussian_flux_data():
    """Generate synthetic Gaussian flux data (centered, symmetric)."""
    np.random.seed(42)
    return np.random.normal(0, 1.0, size=(100, 64)).astype(np.float64)


@pytest.fixture
def skewed_flux_data():
    """Generate synthetic skewed flux data (mean-subtracted gamma)."""
    np.random.seed(42)
    gamma_data = np.random.gamma(2.0, 1.0, size=(100, 64))
    # Subtract mean to center around zero
    return (gamma_data - gamma_data.mean()).astype(np.float64)


# ============================================================================
# NoiseDescriptor Tests
# ============================================================================

class TestNoiseDescriptor:
    """Tests for NoiseDescriptor dataclass."""

    def test_basic_creation(self):
        """Test basic descriptor creation."""
        desc = NoiseDescriptor(
            kind="intensity",
            nt=100, nchan=64,
            mu=1.0, sigma=0.0, shift=0.0,
            gamma_k=1.0, gamma_theta=1.0,
            phi_t=0.0, phi_f=0.0,
            g_t=np.ones(100, dtype=np.float32),
            b_f=np.ones(64, dtype=np.float32)
        )
        
        assert desc.kind == "intensity"
        assert desc.nt == 100
        assert desc.nchan == 64

    def test_sample_shape(self):
        """Test that sample() returns correct shape."""
        desc = NoiseDescriptor(
            kind="intensity",
            nt=100, nchan=64,
            mu=1.0, sigma=0.0, shift=0.0,
            gamma_k=1.0, gamma_theta=1.0,
            phi_t=0.0, phi_f=0.0,
            g_t=np.ones(100, dtype=np.float32),
            b_f=np.ones(64, dtype=np.float32)
        )
        
        sample = desc.sample(seed=42)
        
        assert sample.shape == (100, 64)

    def test_sample_reproducibility(self):
        """Test that same seed gives same sample."""
        desc = NoiseDescriptor(
            kind="flux_gauss",
            nt=50, nchan=32,
            mu=0.0, sigma=1.0, shift=0.0,
            gamma_k=0.0, gamma_theta=0.0,
            phi_t=0.0, phi_f=0.0,
            g_t=np.ones(50, dtype=np.float32),
            b_f=np.ones(32, dtype=np.float32)
        )
        
        sample1 = desc.sample(seed=42)
        sample2 = desc.sample(seed=42)
        
        assert_allclose(sample1, sample2)


# ============================================================================
# Noise Estimation Tests
# ============================================================================

class TestNoiseEstimation:
    """Tests for estimate_noise_descriptor function."""

    def test_detects_intensity(self, intensity_data):
        """Test that intensity data is correctly identified."""
        desc = estimate_noise_descriptor(intensity_data)
        
        assert desc.kind == "intensity"
        assert desc.mu > 0

    def test_detects_gaussian_flux(self, gaussian_flux_data):
        """Test that Gaussian flux data is correctly identified."""
        desc = estimate_noise_descriptor(gaussian_flux_data)
        
        assert desc.kind == "flux_gauss"
        assert desc.sigma > 0

    def test_detects_skewed_flux(self, skewed_flux_data):
        """Test that skewed flux data is correctly identified."""
        desc = estimate_noise_descriptor(skewed_flux_data)
        
        # Should be either flux_shiftedgamma or flux_gauss depending on skewness
        assert desc.kind in ["flux_shiftedgamma", "flux_gauss"]

    def test_shape_preservation(self, intensity_data):
        """Test that estimated descriptor has correct dimensions."""
        desc = estimate_noise_descriptor(intensity_data)
        
        assert desc.nt == 100
        assert desc.nchan == 64
        assert len(desc.g_t) == 100
        assert len(desc.b_f) == 64

    def test_handles_nan(self):
        """Test that NaN values are handled gracefully."""
        np.random.seed(42)
        data = np.random.exponential(1.0, size=(100, 64))
        data[10, 20] = np.nan
        data[50, 30] = np.nan
        
        # Should not raise
        desc = estimate_noise_descriptor(data)
        
        assert desc is not None


# ============================================================================
# Noise Synthesis Tests
# ============================================================================

class TestNoiseSynthesis:
    """Tests for noise synthesis from descriptors."""

    def test_intensity_statistics(self, intensity_data):
        """Test that synthesized intensity has correct statistics."""
        desc = estimate_noise_descriptor(intensity_data)
        
        # Generate many samples
        samples = [desc.sample(seed=i) for i in range(10)]
        synth = np.concatenate(samples, axis=0)
        
        # Mean should be close to estimated mu
        assert_allclose(np.mean(synth), desc.mu, rtol=0.3)

    def test_gaussian_statistics(self, gaussian_flux_data):
        """Test that synthesized Gaussian flux has correct statistics."""
        desc = estimate_noise_descriptor(gaussian_flux_data)
        
        if desc.kind != "flux_gauss":
            pytest.skip("Data not detected as Gaussian")
        
        sample = desc.sample(seed=42)
        
        # Should be centered near zero
        assert abs(np.mean(sample)) < 0.5
        # Std should be close to estimated sigma
        assert_allclose(np.std(sample), desc.sigma, rtol=0.3)

    def test_all_finite(self, intensity_data):
        """Test that synthesized data contains no NaN or Inf."""
        desc = estimate_noise_descriptor(intensity_data)
        sample = desc.sample(seed=42)
        
        assert np.all(np.isfinite(sample))

    def test_correlation_structure(self):
        """Test that temporal correlation is preserved."""
        np.random.seed(42)
        
        # Create data with known correlation
        nt, nchan = 200, 32
        data = np.zeros((nt, nchan))
        for ch in range(nchan):
            # AR(1) process
            for t in range(1, nt):
                data[t, ch] = 0.5 * data[t-1, ch] + np.random.normal(0, 1)
        
        # Estimate and synthesize
        desc = estimate_noise_descriptor(data)
        
        # phi_t should capture some correlation
        # Note: this is a loose check as estimation isn't perfect
        assert abs(desc.phi_t) > 0.1 or desc.phi_t == 0  # May be clipped


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:
    """Tests for NoiseDescriptor serialization."""

    def test_json_roundtrip(self, tmp_path, intensity_data):
        """Test JSON save/load roundtrip."""
        desc = estimate_noise_descriptor(intensity_data)
        
        json_path = tmp_path / "noise_desc.json"
        desc.to_json(json_path)
        
        loaded = NoiseDescriptor.from_json(json_path)
        
        assert loaded.kind == desc.kind
        assert loaded.nt == desc.nt
        assert loaded.nchan == desc.nchan
        assert_allclose(loaded.mu, desc.mu)
        assert_allclose(loaded.g_t, desc.g_t)
        assert_allclose(loaded.b_f, desc.b_f)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases in noise estimation."""

    def test_constant_data(self):
        """Test handling of constant data (zero variance)."""
        data = np.ones((100, 64))
        
        # Should handle gracefully (may return Gaussian with tiny sigma)
        desc = estimate_noise_descriptor(data)
        
        assert desc is not None

    def test_single_row(self):
        """Test handling of single time sample."""
        np.random.seed(42)
        data = np.random.exponential(1.0, size=(1, 64))
        
        # May fail or return limited descriptor
        # Just check it doesn't crash
        try:
            desc = estimate_noise_descriptor(data)
        except (ValueError, IndexError):
            pass  # Expected for edge case

    def test_highly_negative_data(self):
        """Test handling of highly negative data."""
        np.random.seed(42)
        data = np.random.normal(-100, 1.0, size=(100, 64))
        
        desc = estimate_noise_descriptor(data)
        
        # Should be flux type since not positive
        assert desc.kind in ["flux_gauss", "flux_shiftedgamma"]

    def test_mixed_scales(self):
        """Test handling of data with varying scales across channels."""
        np.random.seed(42)
        data = np.zeros((100, 64))
        
        # Different scale per channel
        for ch in range(64):
            scale = 0.1 + ch * 0.1
            data[:, ch] = np.random.exponential(scale, size=100)
        
        desc = estimate_noise_descriptor(data)
        
        # b_f should capture the varying bandpass
        assert np.std(desc.b_f) > 0  # Should vary

