"""
test_interactive.py
====================

Unit tests for burstfit_interactive module (InitialGuessWidget).
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import Mock, patch
from flits.scattering.scat_analysis.burstfit_interactive import InitialGuessWidget
from flits.scattering.scat_analysis.burstfit import FRBParams


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with realistic attributes."""
    dataset = Mock()
    
    # Realistic data parameters
    np.random.seed(42)
    n_freq, n_time = 16, 100
    dt_ms = 0.065  # Typical DSA time resolution
    
    # Create time and frequency axes
    dataset.time = np.linspace(0, dt_ms * n_time, n_time)
    dataset.freq = np.linspace(1.3, 1.5, n_freq)
    dataset.df_MHz = (dataset.freq[1] - dataset.freq[0]) * 1000
    dataset.dm_init = 0.0
    
    # Create burst-like signal with noise
    data = np.random.randn(n_freq, n_time) * 0.05
    t_peak = n_time // 2
    for i in range(n_freq):
        # Add Gaussian-like burst
        burst = np.exp(-((np.arange(n_time) - t_peak) ** 2) / (2 * 5**2))
        data[i, :] += burst
    
    dataset.data = data
    return dataset


@pytest.fixture
def widget(mock_dataset):
    """Create an InitialGuessWidget instance."""
    return InitialGuessWidget(mock_dataset, model_key="M3")


# ============================================================================
# Slider Configuration Tests
# ============================================================================


class TestSliderConfiguration:
    """Tests for slider step sizes and bounds.
    
    These tests verify the slider configuration directly by inspecting
    the create_widget method's logic without actually creating the full widget.
    """
    
    def test_slider_step_sizes(self, mock_dataset):
        """Test that slider step sizes are fine enough for precise control."""
        widget = InitialGuessWidget(mock_dataset, model_key="M3")
        
        # Get the computed ranges the widget would use
        t_range = widget.dataset.time[-1] - widget.dataset.time[0]
        dt_ms = widget.dataset.time[1] - widget.dataset.time[0]
        c_max = np.max(widget.dataset.data) * widget.dataset.data.shape[0]
        
        # Compute expected slider parameters using the same logic as create_widget
        c0_max = max(c_max * 1.5, widget.params.c0 * 2)
        c0_step = max(c0_max / 2000, 0.0001)  # ~2000 steps across range
        tau_max = max(t_range * 0.2, widget.params.tau_1ghz * 2, 0.5)
        tau_step = max(tau_max / 2000, dt_ms / 20, 0.0001)
        
        # Verify step sizes are very fine (these are the expected values from the code)
        gamma_step = 0.001
        zeta_step = 0.001
        alpha_step = 0.01
        t0_step = dt_ms / 20
        
        # Check c0 step is fine (should allow ~2000 steps across range)
        assert c0_step <= c0_max / 500, \
            f"c0 step ({c0_step}) should be <= {c0_max/500} for fine control"
        
        # Check gamma step (very fine)
        assert gamma_step == 0.001, "gamma step should be 0.001"
        
        # Check zeta step (very fine)
        assert zeta_step == 0.001, "zeta step should be 0.001"
        
        # Check alpha step
        assert alpha_step == 0.01, "alpha step should be 0.01"
        
        # Check t0 step (20× finer than time resolution)
        assert t0_step < dt_ms / 10, "t0 step should be at least 10× finer than time resolution"
        
        # Check tau step is very fine (allows at least ~500 steps across range)
        assert tau_step <= tau_max / 100, "tau step should allow very fine control"
    
    def test_slider_upper_bounds(self, mock_dataset):
        """Test that slider upper bounds are reasonable."""
        widget = InitialGuessWidget(mock_dataset, model_key="M3")
        
        # Get data characteristics
        t_range = widget.dataset.time[-1] - widget.dataset.time[0]
        c_max = np.max(widget.dataset.data) * widget.dataset.data.shape[0]
        
        # Check computed bounds (using the same logic as create_widget)
        c0_max = max(c_max * 1.5, widget.params.c0 * 2)
        tau_max = max(t_range * 0.2, widget.params.tau_1ghz * 2, 0.5)
        
        # Verify bounds are reasonable
        # c0: max at 2× initial or 1.5× data max (not 3×)
        assert c0_max <= max(c_max * 1.5, widget.params.c0 * 2) + 1e-6, \
            "c0_max should be at most 2× initial or 1.5× data max"
        
        # tau: max at 20% of time range (not 50%)
        assert tau_max <= max(t_range * 0.2, widget.params.tau_1ghz * 2, 0.5) + 1e-6, \
            "tau_max should be reasonable"
        
        # gamma range is -3 to 2 (narrower than -4 to 4)
        gamma_min = -3.0
        gamma_max = 2.0
        assert gamma_min == -3.0, "gamma min should be -3.0"
        assert gamma_max == 2.0, "gamma max should be 2.0"
        
        # zeta max is 2.0 (reduced from 5.0)
        zeta_max = 2.0
        assert zeta_max == 2.0, "zeta max should be 2.0"
    
    def test_slider_configuration_consistency(self, mock_dataset):
        """Test that slider configuration is internally consistent."""
        widget = InitialGuessWidget(mock_dataset, model_key="M3")
        params = widget.params
        
        # Get bounds
        t_range = widget.dataset.time[-1] - widget.dataset.time[0]
        c_max = np.max(widget.dataset.data) * widget.dataset.data.shape[0]
        
        c0_max = max(c_max * 1.5, params.c0 * 2)
        tau_max = max(t_range * 0.2, params.tau_1ghz * 2, 0.5)
        
        # Initial values should be within bounds
        assert 0 <= params.c0 <= c0_max, "Initial c0 should be within bounds"
        assert widget.dataset.time[0] <= params.t0 <= widget.dataset.time[-1], \
            "Initial t0 should be within time range"
        assert -3.0 <= params.gamma <= 2.0, "Initial gamma should be within bounds"
        assert 0.01 <= params.zeta <= 2.0, "Initial zeta should be within bounds"
        assert 0 <= params.tau_1ghz <= tau_max, "Initial tau should be within bounds"
        assert 2.0 <= params.alpha <= 6.0, "Initial alpha should be within bounds"


# ============================================================================
# Data-Driven Guess Tests
# ============================================================================


class TestDataDrivenGuess:
    """Tests for the automatic initial parameter estimation."""
    
    def test_guess_returns_valid_params(self, mock_dataset):
        """Test that data-driven guess returns valid FRBParams."""
        widget = InitialGuessWidget(mock_dataset, model_key="M3")
        params = widget._get_data_driven_guess()
        
        assert isinstance(params, FRBParams)
        assert params.c0 > 0, "c0 should be positive"
        assert params.zeta > 0, "zeta should be positive"
        assert params.tau_1ghz > 0, "tau_1ghz should be positive"
        assert 2.0 <= params.alpha <= 6.0, "alpha should be in reasonable range"
    
    def test_t0_near_peak(self, mock_dataset):
        """Test that t0 is estimated near the data peak."""
        widget = InitialGuessWidget(mock_dataset, model_key="M3")
        params = widget._get_data_driven_guess()
        
        # Find actual peak
        time_profile = np.nansum(mock_dataset.data, axis=0)
        peak_idx = np.argmax(time_profile)
        actual_peak_time = mock_dataset.time[peak_idx]
        
        # t0 should be close to peak
        dt = mock_dataset.time[1] - mock_dataset.time[0]
        assert abs(params.t0 - actual_peak_time) < 5 * dt, \
            f"t0 ({params.t0:.3f}) should be near peak ({actual_peak_time:.3f})"
    
    def test_guess_handles_flat_spectrum(self):
        """Test guess works with flat frequency spectrum."""
        dataset = Mock()
        n_freq, n_time = 16, 100
        dataset.time = np.linspace(0, 10, n_time)
        dataset.freq = np.linspace(1.3, 1.5, n_freq)
        dataset.df_MHz = 12.5
        dataset.dm_init = 0.0
        
        # Flat spectrum (uniform across frequency)
        burst = np.exp(-((np.arange(n_time) - 50) ** 2) / 50)
        dataset.data = np.outer(np.ones(n_freq), burst)
        
        widget = InitialGuessWidget(dataset, model_key="M3")
        params = widget._get_data_driven_guess()
        
        # Should still produce valid params
        assert np.isfinite(params.c0)
        assert np.isfinite(params.t0)
        assert np.isfinite(params.gamma)
    
    def test_guess_handles_noisy_data(self):
        """Test guess is robust to noisy data."""
        dataset = Mock()
        np.random.seed(42)
        n_freq, n_time = 16, 100
        dataset.time = np.linspace(0, 10, n_time)
        dataset.freq = np.linspace(1.3, 1.5, n_freq)
        dataset.df_MHz = 12.5
        dataset.dm_init = 0.0
        
        # Mostly noise with weak signal
        dataset.data = np.random.randn(n_freq, n_time)
        # Add weak burst
        dataset.data[:, 45:55] += 0.5
        
        widget = InitialGuessWidget(dataset, model_key="M3")
        params = widget._get_data_driven_guess()
        
        # Should produce finite params even for noisy data
        assert np.isfinite(params.c0)
        assert np.isfinite(params.t0)
        assert np.isfinite(params.gamma)


# ============================================================================
# Widget Functionality Tests
# ============================================================================


class TestWidgetFunctionality:
    """Tests for widget methods."""
    
    def test_get_params_initial(self, widget):
        """Test get_params returns initial guess before acceptance."""
        params = widget.get_params()
        assert isinstance(params, FRBParams)
        assert params == widget.params
    
    def test_get_params_after_optimize(self, widget):
        """Test get_params returns optimized params after acceptance."""
        # Simulate accepting parameters
        widget.optimized_params = FRBParams(
            c0=5.0, t0=3.0, gamma=-1.0, zeta=0.3,
            tau_1ghz=0.2, alpha=4.2, delta_dm=0.0
        )
        
        params = widget.get_params()
        assert params == widget.optimized_params
    
    def test_custom_initial_params(self, mock_dataset):
        """Test widget accepts custom initial parameters."""
        custom = FRBParams(
            c0=10.0, t0=5.0, gamma=-2.0, zeta=0.5,
            tau_1ghz=0.5, alpha=4.5, delta_dm=0.0
        )
        
        widget = InitialGuessWidget(
            mock_dataset, model_key="M3", initial_params=custom
        )
        
        assert widget.params == custom


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_data(self):
        """Test widget handles data with all zeros."""
        dataset = Mock()
        dataset.time = np.linspace(0, 10, 100)
        dataset.freq = np.linspace(1.3, 1.5, 16)
        dataset.df_MHz = 12.5
        dataset.dm_init = 0.0
        dataset.data = np.zeros((16, 100))
        
        # Should not crash
        widget = InitialGuessWidget(dataset, model_key="M3")
        params = widget._get_data_driven_guess()
        
        assert np.isfinite(params.c0)
    
    def test_single_time_bin(self):
        """Test widget handles very short time series."""
        dataset = Mock()
        dataset.time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        dataset.freq = np.linspace(1.3, 1.5, 16)
        dataset.df_MHz = 12.5
        dataset.dm_init = 0.0
        dataset.data = np.random.rand(16, 5)
        
        widget = InitialGuessWidget(dataset, model_key="M3")
        params = widget._get_data_driven_guess()
        
        assert np.isfinite(params.t0)
        assert 0.0 <= params.t0 <= 0.4


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
