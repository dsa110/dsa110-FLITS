"""
test_core.py
============

Unit tests for scint_analysis/core.py - DynamicSpectrum and ACF classes.
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
from numpy.testing import assert_allclose, assert_array_equal

from scint_analysis.core import DynamicSpectrum, ACF


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_power():
    """Simple 2D power array (freq x time)."""
    np.random.seed(42)
    return np.random.rand(64, 256)


@pytest.fixture
def simple_freqs():
    """Simple frequency axis (MHz)."""
    return np.linspace(1280, 1530, 64)


@pytest.fixture
def simple_times():
    """Simple time axis (seconds)."""
    return np.linspace(0, 0.1, 256)


@pytest.fixture
def simple_ds(simple_power, simple_freqs, simple_times):
    """Simple DynamicSpectrum instance."""
    return DynamicSpectrum(simple_power, simple_freqs, simple_times)


@pytest.fixture
def burst_ds():
    """DynamicSpectrum with a simulated burst."""
    np.random.seed(42)
    n_freq, n_time = 64, 512
    freqs = np.linspace(1280, 1530, n_freq)
    times = np.linspace(0, 0.1, n_time)
    
    # Background noise
    power = np.random.normal(0, 0.1, (n_freq, n_time))
    
    # Add burst at center with Gaussian profile
    t_burst = n_time // 2
    burst_width = 10
    time_profile = np.exp(-0.5 * ((np.arange(n_time) - t_burst) / burst_width) ** 2)
    
    # Add frequency structure (simple scintillation-like pattern)
    freq_pattern = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_freq) / 10)
    
    burst = np.outer(freq_pattern, time_profile)
    power += burst
    
    return DynamicSpectrum(power, freqs, times)


# ============================================================================
# DynamicSpectrum Initialization Tests
# ============================================================================

class TestDynamicSpectrumInit:
    """Tests for DynamicSpectrum initialization."""

    def test_basic_init(self, simple_power, simple_freqs, simple_times):
        """Test basic initialization."""
        ds = DynamicSpectrum(simple_power, simple_freqs, simple_times)
        
        assert ds.num_channels == 64
        assert ds.num_timesteps == 256
        assert_allclose(ds.frequencies, simple_freqs)
        assert_allclose(ds.times, simple_times)

    def test_masked_array_creation(self, simple_power, simple_freqs, simple_times):
        """Test that power is converted to masked array."""
        ds = DynamicSpectrum(simple_power, simple_freqs, simple_times)
        
        assert isinstance(ds.power, np.ma.MaskedArray)

    def test_nan_handling(self, simple_freqs, simple_times):
        """Test that NaN values are masked."""
        power = np.random.rand(64, 256)
        power[10, 20] = np.nan
        power[30, 40] = np.nan
        
        ds = DynamicSpectrum(power, simple_freqs, simple_times)
        
        assert ds.power.mask[10, 20] is np.True_
        assert ds.power.mask[30, 40] is np.True_

    def test_frequency_flip(self, simple_power, simple_times):
        """Test that descending frequencies are flipped."""
        freqs_desc = np.linspace(1530, 1280, 64)  # Descending
        
        ds = DynamicSpectrum(simple_power, freqs_desc, simple_times)
        
        # Frequencies should be ascending after init
        assert ds.frequencies[0] < ds.frequencies[-1]

    def test_shape_validation(self, simple_freqs, simple_times):
        """Test that shape mismatches raise errors."""
        bad_power = np.random.rand(32, 256)  # Wrong freq dimension
        
        with pytest.raises(AssertionError):
            DynamicSpectrum(bad_power, simple_freqs, simple_times)


# ============================================================================
# DynamicSpectrum Properties Tests
# ============================================================================

class TestDynamicSpectrumProperties:
    """Tests for DynamicSpectrum properties."""

    def test_channel_width(self, simple_ds):
        """Test channel width calculation."""
        expected = np.abs(np.mean(np.diff(simple_ds.frequencies)))
        assert_allclose(simple_ds.channel_width_mhz, expected)

    def test_time_resolution(self, simple_ds):
        """Test time resolution calculation."""
        expected = np.abs(np.mean(np.diff(simple_ds.times)))
        assert_allclose(simple_ds.time_resolution_s, expected)


# ============================================================================
# DynamicSpectrum Downsample Tests
# ============================================================================

class TestDynamicSpectrumDownsample:
    """Tests for DynamicSpectrum.downsample()."""

    def test_identity_downsample(self, simple_ds):
        """Test that f_factor=1, t_factor=1 returns same object."""
        result = simple_ds.downsample(f_factor=1, t_factor=1)
        
        assert result is simple_ds

    def test_downsample_shape(self, simple_ds):
        """Test output shape after downsampling."""
        result = simple_ds.downsample(f_factor=4, t_factor=2)
        
        assert result.num_channels == 16
        assert result.num_timesteps == 128

    def test_downsample_in_place(self, simple_power, simple_freqs, simple_times):
        """Test in-place downsampling."""
        ds = DynamicSpectrum(simple_power.copy(), simple_freqs.copy(), simple_times.copy())
        original_id = id(ds)
        
        result = ds.downsample(f_factor=2, t_factor=2, in_place=True)
        
        assert id(result) == original_id
        assert ds.num_channels == 32

    def test_downsample_preserves_mean(self, simple_ds):
        """Test that downsampling approximately preserves mean."""
        original_mean = np.ma.mean(simple_ds.power)
        
        result = simple_ds.downsample(f_factor=4, t_factor=4)
        downsampled_mean = np.ma.mean(result.power)
        
        assert_allclose(original_mean, downsampled_mean, rtol=0.05)


# ============================================================================
# DynamicSpectrum Burst Detection Tests
# ============================================================================

class TestBurstEnvelope:
    """Tests for find_burst_envelope()."""

    def test_finds_burst(self, burst_ds):
        """Test that burst is found."""
        lims = burst_ds.find_burst_envelope(thres=3.0)
        
        assert len(lims) == 2
        assert lims[0] < lims[1]
        # Burst is at center (256), should be found
        assert lims[0] < 256 < lims[1]

    def test_no_burst_returns_empty(self, simple_ds):
        """Test that no burst returns [0, 0] with high threshold."""
        lims = simple_ds.find_burst_envelope(thres=100.0)
        
        assert lims == [0, 0]

    def test_padding(self, burst_ds):
        """Test that padding expands envelope."""
        lims_nopad = burst_ds.find_burst_envelope(thres=3.0, padding_factor=0.0)
        lims_padded = burst_ds.find_burst_envelope(thres=3.0, padding_factor=0.5)
        
        # Padded should be wider
        assert lims_padded[0] <= lims_nopad[0]
        assert lims_padded[1] >= lims_nopad[1]


# ============================================================================
# DynamicSpectrum Profile/Spectrum Tests
# ============================================================================

class TestProfileSpectrum:
    """Tests for get_profile() and get_spectrum()."""

    def test_get_profile_full(self, simple_ds):
        """Test full profile extraction."""
        profile = simple_ds.get_profile()
        
        assert len(profile) == simple_ds.num_timesteps

    def test_get_profile_windowed(self, simple_ds):
        """Test windowed profile extraction."""
        profile = simple_ds.get_profile(time_window_bins=(50, 100))
        
        assert len(profile) == 50

    def test_get_spectrum(self, simple_ds):
        """Test spectrum extraction."""
        spectrum = simple_ds.get_spectrum((50, 100))
        
        assert len(spectrum) == simple_ds.num_channels


# ============================================================================
# DynamicSpectrum RFI Masking Tests
# ============================================================================

class TestRFIMasking:
    """Tests for mask_rfi()."""

    def test_basic_masking(self, burst_ds):
        """Test that RFI masking returns new object."""
        config = {
            'analysis': {
                'rfi_masking': {
                    'find_burst_thres': 3.0,
                    'freq_threshold_sigma': 5.0,
                    'time_threshold_sigma': 7.0,
                    'rfi_downsample_factor': 8,
                    'enable_time_domain_flagging': False,
                }
            }
        }
        
        result = burst_ds.mask_rfi(config)
        
        # Should return new object
        assert result is not burst_ds

    def test_manual_window(self, burst_ds):
        """Test that manual window is respected."""
        config = {
            'analysis': {
                'rfi_masking': {
                    'manual_burst_window': [200, 300],
                    'freq_threshold_sigma': 5.0,
                    'time_threshold_sigma': 7.0,
                    'rfi_downsample_factor': 8,
                    'enable_time_domain_flagging': False,
                }
            }
        }
        
        # Should not raise error
        result = burst_ds.mask_rfi(config)
        assert result is not None


# ============================================================================
# ACF Class Tests
# ============================================================================

class TestACF:
    """Tests for ACF class."""

    def test_basic_init(self):
        """Test basic ACF initialization."""
        lags = np.linspace(-10, 10, 21)
        acf_data = np.exp(-np.abs(lags) / 3.0)
        
        acf = ACF(acf_data, lags)
        
        assert len(acf) == 21
        assert_allclose(acf.lags, lags)
        assert_allclose(acf.acf, acf_data)
        assert acf.err is None

    def test_init_with_errors(self):
        """Test ACF initialization with errors."""
        lags = np.linspace(-10, 10, 21)
        acf_data = np.exp(-np.abs(lags) / 3.0)
        errors = np.ones(21) * 0.1
        
        acf = ACF(acf_data, lags, acf_err=errors)
        
        assert acf.err is not None
        assert_allclose(acf.err, errors)

    def test_shape_validation(self):
        """Test that shape mismatches raise errors."""
        lags = np.linspace(-10, 10, 21)
        acf_data = np.zeros(15)  # Wrong length
        
        with pytest.raises(ValueError):
            ACF(acf_data, lags)

    def test_1d_validation(self):
        """Test that 2D arrays raise errors."""
        lags = np.linspace(-10, 10, 21)
        acf_data = np.zeros((21, 2))  # 2D
        
        with pytest.raises(ValueError):
            ACF(acf_data, lags)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_channel(self, simple_times):
        """Test single-channel spectrum."""
        power = np.random.rand(1, 256)
        freqs = np.array([1400.0])
        
        ds = DynamicSpectrum(power, freqs, simple_times)
        
        assert ds.num_channels == 1

    def test_single_timestep(self, simple_freqs):
        """Test single-timestep spectrum."""
        power = np.random.rand(64, 1)
        times = np.array([0.0])
        
        ds = DynamicSpectrum(power, simple_freqs, times)
        
        assert ds.num_timesteps == 1

    def test_all_masked(self, simple_freqs, simple_times):
        """Test handling of fully masked data."""
        power = np.full((64, 256), np.nan)
        
        ds = DynamicSpectrum(power, simple_freqs, simple_times)
        
        # All should be masked
        assert ds.power.mask.all()

