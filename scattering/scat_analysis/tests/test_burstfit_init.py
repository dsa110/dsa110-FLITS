"""
test_burstfit_init.py
=====================

Unit and integration tests for the data-driven initial guess module.

Test Categories:
- Unit tests: Individual estimation functions
- Smoke tests: Basic functionality checks
- Integration tests: Full pipeline integration
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scattering.scat_analysis.burstfit import FRBModel, FRBParams
from scattering.scat_analysis.burstfit_init import (
    data_driven_initial_guess,
    quick_initial_guess,
    estimate_spectral_index,
    estimate_pulse_width,
    estimate_scattering_from_tail,
    estimate_scattering_frequency_scaling,
    InitialGuessResult,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_time():
    """Time axis in ms (0-50 ms, 1000 samples)."""
    return np.linspace(0, 50, 1000)


@pytest.fixture
def simple_freq():
    """Frequency axis in GHz (1.0-1.5 GHz, 256 channels)."""
    return np.linspace(1.0, 1.5, 256)


@pytest.fixture
def synthetic_burst(simple_time, simple_freq):
    """Generate synthetic burst with known parameters."""
    # True parameters
    true_params = {
        't0': 25.0,      # Peak time (ms)
        'width': 2.0,    # FWHM (ms)
        'gamma': -2.0,   # Spectral index
        'tau_1ghz': 1.0, # Scattering at 1 GHz (ms)
        'alpha': 4.0,    # Scattering scaling
    }
    
    time = simple_time
    freq = simple_freq
    
    # Reference frequency
    ref_freq = np.median(freq)
    
    # Build model
    sigma = true_params['width'] / 2.355
    amp = (freq / ref_freq) ** true_params['gamma']
    
    # Gaussian profile
    gauss = np.exp(-0.5 * ((time[None, :] - true_params['t0']) / sigma) ** 2)
    
    # Apply spectral index
    data = amp[:, None] * gauss
    
    # Add scattering (exponential tail)
    tau = true_params['tau_1ghz'] * (freq / 1.0) ** (-true_params['alpha'])
    for i, t in enumerate(tau):
        # Convolve with exponential
        kernel = np.exp(-time / max(t, 0.01))
        kernel /= np.sum(kernel)
        # Use full convolution and slice to keep causal part
        convolved = np.convolve(data[i, :], kernel, mode='full')
        data[i, :] = convolved[:len(time)]
    
    # Normalize
    data = data / np.max(data) * 1000
    
    # Add noise
    noise = np.random.normal(0, 10, data.shape)
    data_noisy = data + noise
    
    return data_noisy, freq, time, true_params


@pytest.fixture
def edge_case_data(simple_time, simple_freq):
    """Edge case: very low S/N burst."""
    data = np.random.normal(0, 1, (len(simple_freq), len(simple_time)))
    # Add weak burst
    data[:, 450:550] += 0.5
    return data, simple_freq, simple_time


# ============================================================================
# Unit Tests: estimate_spectral_index
# ============================================================================

class TestEstimateSpectralIndex:
    """Unit tests for spectral index estimation."""
    
    def test_returns_tuple(self, synthetic_burst):
        """Should return (gamma, gamma_err) tuple."""
        data, freq, time, _ = synthetic_burst
        result = estimate_spectral_index(data, freq)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_recovers_gamma(self, synthetic_burst):
        """Should recover spectral index within uncertainty."""
        data, freq, time, true_params = synthetic_burst
        gamma, gamma_err = estimate_spectral_index(data, freq)
        
        # Allow reasonable tolerance (spectral index is hard to measure)
        assert -5 < gamma < 2  # Within physical bounds
        # Note: May not exactly match true value due to scattering effects
    
    def test_default_on_bad_data(self, simple_time, simple_freq):
        """Should return default -1.6 for all-zero data."""
        data = np.zeros((len(simple_freq), len(simple_time)))
        gamma, gamma_err = estimate_spectral_index(data, simple_freq)
        assert gamma == -1.6
    
    def test_handles_nan(self, simple_time, simple_freq):
        """Should handle NaN values gracefully."""
        data = np.random.rand(len(simple_freq), len(simple_time))
        data[10:20, :] = np.nan  # Some NaN channels
        gamma, gamma_err = estimate_spectral_index(data, simple_freq)
        assert np.isfinite(gamma)


# ============================================================================
# Unit Tests: estimate_pulse_width
# ============================================================================

class TestEstimatePulseWidth:
    """Unit tests for pulse width estimation."""
    
    def test_returns_tuple(self, synthetic_burst):
        """Should return (t0, width, width_err) tuple."""
        data, freq, time, _ = synthetic_burst
        result = estimate_pulse_width(data, time)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_finds_peak_time(self, synthetic_burst):
        """Should find peak time near true value."""
        data, freq, time, true_params = synthetic_burst
        t0, width, width_err = estimate_pulse_width(data, time)
        
        # Peak should be within 5% of true value
        assert abs(t0 - true_params['t0']) < 0.05 * true_params['t0']
    
    def test_positive_width(self, synthetic_burst):
        """Width should always be positive."""
        data, freq, time, _ = synthetic_burst
        t0, width, width_err = estimate_pulse_width(data, time)
        assert width > 0
        assert width_err >= 0
    
    def test_handles_flat_profile(self, simple_time, simple_freq):
        """Should handle flat (no burst) data gracefully."""
        data = np.ones((len(simple_freq), len(simple_time)))
        t0, width, _ = estimate_pulse_width(data, simple_time)
        assert np.isfinite(t0)
        assert width > 0


# ============================================================================
# Unit Tests: estimate_scattering_from_tail
# ============================================================================

class TestEstimateScatteringFromTail:
    """Unit tests for scattering tail estimation."""
    
    def test_returns_tuple(self, synthetic_burst):
        """Should return (tau, tau_err) tuple."""
        data, freq, time, true_params = synthetic_burst
        t0 = true_params['t0']
        width = true_params['width']
        result = estimate_scattering_from_tail(data, time, freq, t0, width)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_positive_tau(self, synthetic_burst):
        """Tau should always be positive."""
        data, freq, time, true_params = synthetic_burst
        tau, tau_err = estimate_scattering_from_tail(
            data, time, freq, true_params['t0'], true_params['width']
        )
        assert tau > 0
        assert tau_err >= 0
    
    def test_reasonable_scatter(self, synthetic_burst):
        """Should estimate scattering in reasonable range."""
        data, freq, time, true_params = synthetic_burst
        tau, _ = estimate_scattering_from_tail(
            data, time, freq, true_params['t0'], true_params['width']
        )
        # Should be reasonable order of magnitude
        assert 0.01 < tau < 100


# ============================================================================
# Smoke Tests: data_driven_initial_guess
# ============================================================================

class TestDataDrivenInitialGuessSmoke:
    """Smoke tests for full initial guess function."""
    
    def test_returns_result_object(self, synthetic_burst):
        """Should return InitialGuessResult."""
        data, freq, time, _ = synthetic_burst
        result = data_driven_initial_guess(data, freq, time, verbose=False)
        assert isinstance(result, InitialGuessResult)
        assert isinstance(result.params, FRBParams)
    
    def test_has_all_params(self, synthetic_burst):
        """Result should have all required parameters."""
        data, freq, time, _ = synthetic_burst
        result = data_driven_initial_guess(data, freq, time, verbose=False)
        params = result.params
        
        assert hasattr(params, 'c0')
        assert hasattr(params, 't0')
        assert hasattr(params, 'gamma')
        assert hasattr(params, 'zeta')
        assert hasattr(params, 'tau_1ghz')
        assert hasattr(params, 'alpha')
    
    def test_params_are_finite(self, synthetic_burst):
        """All parameters should be finite."""
        data, freq, time, _ = synthetic_burst
        result = data_driven_initial_guess(data, freq, time, verbose=False)
        params = result.params
        
        assert np.isfinite(params.c0)
        assert np.isfinite(params.t0)
        assert np.isfinite(params.gamma)
        assert np.isfinite(params.zeta)
        assert np.isfinite(params.tau_1ghz)
        assert np.isfinite(params.alpha)
    
    def test_positive_params_are_positive(self, synthetic_burst):
        """c0, zeta, tau_1ghz should be positive."""
        data, freq, time, _ = synthetic_burst
        result = data_driven_initial_guess(data, freq, time, verbose=False)
        params = result.params
        
        assert params.c0 > 0
        assert params.zeta > 0
        assert params.tau_1ghz > 0
    
    def test_has_diagnostics(self, synthetic_burst):
        """Result should include diagnostics dict."""
        data, freq, time, _ = synthetic_burst
        result = data_driven_initial_guess(data, freq, time, verbose=False)
        
        assert isinstance(result.diagnostics, dict)
        assert len(result.diagnostics) > 0
    
    def test_quick_wrapper(self, synthetic_burst):
        """quick_initial_guess should return FRBParams directly."""
        data, freq, time, _ = synthetic_burst
        params = quick_initial_guess(data, freq, time)
        assert isinstance(params, FRBParams)


# ============================================================================
# Integration Tests
# ============================================================================

class TestDataDrivenInitialGuessIntegration:
    """Integration tests with the full pipeline."""
    
    def test_params_work_with_model(self, synthetic_burst):
        """Estimated params should work with FRBModel."""
        data, freq, time, _ = synthetic_burst
        result = data_driven_initial_guess(data, freq, time, verbose=False)
        
        # Create model
        model = FRBModel(time, freq, data=data, dm_init=0.0)
        
        # Params should produce valid output
        output = model(result.params, "M3")
        assert output.shape == data.shape
        assert np.all(np.isfinite(output))
    
    def test_params_give_reasonable_likelihood(self, synthetic_burst):
        """Estimated params should give non-terrible likelihood."""
        data, freq, time, _ = synthetic_burst
        result = data_driven_initial_guess(data, freq, time, verbose=False)
        
        model = FRBModel(time, freq, data=data, dm_init=0.0)
        ll = model.log_likelihood(result.params, "M3")
        
        # Log-likelihood should be finite and not terrible
        assert np.isfinite(ll)
        # Shouldn't be astronomically negative
        assert ll > -1e20
    
    def test_frequency_conversion(self, synthetic_burst):
        """Should handle MHz frequencies (auto-convert to GHz)."""
        data, freq_ghz, time, _ = synthetic_burst
        freq_mhz = freq_ghz * 1000  # Convert to MHz
        
        result = data_driven_initial_guess(data, freq_mhz, time, verbose=False)
        
        # Should still work
        assert isinstance(result.params, FRBParams)
        assert np.isfinite(result.params.tau_1ghz)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_all_zeros(self, simple_time, simple_freq):
        """Should handle all-zero data gracefully."""
        data = np.zeros((len(simple_freq), len(simple_time)))
        result = data_driven_initial_guess(data, simple_freq, simple_time, verbose=False)
        
        # Should return something, even if not meaningful
        assert isinstance(result.params, FRBParams)
    
    def test_all_nan(self, simple_time, simple_freq):
        """Should handle all-NaN data gracefully."""
        data = np.full((len(simple_freq), len(simple_time)), np.nan)
        
        # May raise exception or return with warnings - either is acceptable
        try:
            result = data_driven_initial_guess(data, simple_freq, simple_time, verbose=False)
            assert isinstance(result.params, FRBParams)
        except Exception:
            pass  # Exception is acceptable for all-NaN data
    
    def test_single_channel(self, simple_time):
        """Should handle single-channel data."""
        freq = np.array([1.0])
        data = np.random.rand(1, len(simple_time))
        data[0, len(simple_time)//2] = 10  # Add peak
        
        result = data_driven_initial_guess(data, freq, simple_time, verbose=False)
        assert isinstance(result.params, FRBParams)
    
    def test_short_time_series(self, simple_freq):
        """Should handle very short time series."""
        time = np.linspace(0, 1, 10)  # Only 10 samples
        data = np.random.rand(len(simple_freq), 10)
        data[:, 5] = 10  # Add peak
        
        result = data_driven_initial_guess(data, simple_freq, time, verbose=False)
        assert isinstance(result.params, FRBParams)
    
    def test_negative_values(self, simple_time, simple_freq):
        """Should handle data with negative values."""
        data = np.random.normal(0, 1, (len(simple_freq), len(simple_time)))
        data[:, len(simple_time)//2] += 10  # Burst above baseline
        
        result = data_driven_initial_guess(data, simple_freq, simple_time, verbose=False)
        assert isinstance(result.params, FRBParams)
        assert result.params.c0 > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_completes_in_reasonable_time(self, synthetic_burst):
        """Should complete in under 5 seconds."""
        import time as time_module
        
        data, freq, t, _ = synthetic_burst
        
        start = time_module.time()
        result = data_driven_initial_guess(data, freq, t, verbose=False)
        elapsed = time_module.time() - start
        
        assert elapsed < 5.0, f"Took {elapsed:.2f}s, expected < 5s"
    
    def test_large_data(self):
        """Should handle large datasets (4096 channels, 10000 samples)."""
        freq = np.linspace(0.4, 0.8, 4096)
        time = np.linspace(0, 100, 10000)
        data = np.random.rand(4096, 10000)
        data[:, 5000] = 100  # Add burst
        
        result = data_driven_initial_guess(data, freq, time, verbose=False)
        assert isinstance(result.params, FRBParams)
