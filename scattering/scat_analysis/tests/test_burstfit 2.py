"""
test_burstfit.py
================

Comprehensive unit tests for the burstfit module.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from flits.scattering.scat_analysis.burstfit import (
    FRBParams,
    FRBModel,
    FRBFitter,
    build_priors,
    compute_bic,
    downsample,
    goodness_of_fit,
    gelman_rubin,
    _POSITIVE,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_time():
    """Simple time axis (ms)."""
    return np.linspace(-5.0, 5.0, 256)


@pytest.fixture
def simple_freq():
    """Simple frequency axis (GHz)."""
    return np.linspace(0.4, 0.8, 64)


@pytest.fixture
def simple_params():
    """Simple FRB parameters for testing."""
    return FRBParams(
        c0=1.0, t0=0.0, gamma=-1.6, zeta=0.1, tau_1ghz=0.5, alpha=4.4, delta_dm=0.0
    )


@pytest.fixture
def simple_model(simple_time, simple_freq):
    """Simple FRBModel instance without data."""
    return FRBModel(
        time=simple_time,
        freq=simple_freq,
        data=None,
        dm_init=0.0,
        df_MHz=0.39,
    )


@pytest.fixture
def synthetic_data(simple_time, simple_freq, simple_params):
    """Generate synthetic burst data with noise."""
    np.random.seed(42)
    model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
    clean = model(simple_params, "M3")
    noise = np.random.normal(0, 0.05, clean.shape)
    return clean + noise


# ============================================================================
# FRBParams Tests
# ============================================================================

class TestFRBParams:
    """Tests for FRBParams dataclass."""

    def test_defaults(self):
        """Test default parameter values."""
        p = FRBParams(c0=1.0, t0=0.0, gamma=-1.6)
        assert p.zeta == 0.0
        assert p.tau_1ghz == 0.0
        assert p.alpha == 4.4
        assert p.delta_dm == 0.0

    def test_to_sequence_m0(self):
        """Test conversion to sequence for M0 model."""
        p = FRBParams(c0=1.0, t0=2.0, gamma=-1.5)
        seq = p.to_sequence("M0")
        assert len(seq) == 3
        assert seq == [1.0, 2.0, -1.5]

    def test_to_sequence_m1(self):
        """Test conversion to sequence for M1 model."""
        p = FRBParams(c0=1.0, t0=2.0, gamma=-1.5, zeta=0.5)
        seq = p.to_sequence("M1")
        assert len(seq) == 4
        assert seq == [1.0, 2.0, -1.5, 0.5]

    def test_to_sequence_m3(self):
        """Test conversion to sequence for M3 model."""
        p = FRBParams(c0=1.0, t0=2.0, gamma=-1.5, zeta=0.5, tau_1ghz=0.3, alpha=4.0, delta_dm=0.1)
        seq = p.to_sequence("M3")
        assert len(seq) == 7
        assert seq == [1.0, 2.0, -1.5, 0.5, 0.3, 4.0, 0.1]

    def test_from_sequence_m3(self):
        """Test reconstruction from sequence for M3 model."""
        seq = [1.0, 2.0, -1.5, 0.5, 0.3, 4.0, 0.1]
        p = FRBParams.from_sequence(seq, "M3")
        assert p.c0 == 1.0
        assert p.t0 == 2.0
        assert p.gamma == -1.5
        assert p.zeta == 0.5
        assert p.tau_1ghz == 0.3
        assert p.alpha == 4.0
        assert p.delta_dm == 0.1

    def test_roundtrip(self):
        """Test that to_sequence and from_sequence are inverses."""
        original = FRBParams(c0=1.5, t0=-0.5, gamma=-2.0, zeta=0.2, tau_1ghz=0.8, alpha=4.2, delta_dm=-0.05)
        for key in ["M0", "M1", "M2", "M3"]:
            seq = original.to_sequence(key)
            reconstructed = FRBParams.from_sequence(seq, key)
            # Check the parameters that are in this model
            seq2 = reconstructed.to_sequence(key)
            assert_allclose(seq, seq2)


# ============================================================================
# FRBModel Tests
# ============================================================================

class TestFRBModel:
    """Tests for FRBModel forward model."""

    def test_output_shape(self, simple_model, simple_params):
        """Test that model output has correct shape."""
        output = simple_model(simple_params, "M3")
        assert output.shape == (64, 256)

    def test_m0_gaussian_peak(self, simple_time, simple_freq):
        """Test M0 model produces a Gaussian centered at t0."""
        model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        # Use a non-zero intrinsic width to ensure the Gaussian is resolved
        p = FRBParams(c0=1.0, t0=0.0, gamma=0.0, zeta=0.2)  # Flat spectrum with width
        output = model(p, "M1")  # M1 includes zeta
        
        # Sum over frequency to get time profile
        profile = np.sum(output, axis=0)
        peak_idx = np.argmax(profile)
        peak_time = simple_time[peak_idx]
        
        # Peak should be near t0=0 (within time resolution)
        dt = simple_time[1] - simple_time[0]
        assert abs(peak_time) < 2 * dt

    def test_m3_scattered_peak_delay(self, simple_time, simple_freq):
        """Test that scattering (M3) causes profile broadening at low frequency."""
        model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        
        # Strong scattering with adequate intrinsic width
        p = FRBParams(c0=1.0, t0=0.0, gamma=0.0, zeta=0.1, tau_1ghz=5.0, alpha=4.0, delta_dm=0.0)
        output = model(p, "M3")
        
        # Note: simple_freq goes from 0.4 to 0.8 GHz, so:
        # output[0] = lowest freq (0.4 GHz) - MORE scattering
        # output[-1] = highest freq (0.8 GHz) - LESS scattering
        low_freq_profile = output[0, :]    # Lowest frequency channel (0.4 GHz)
        high_freq_profile = output[-1, :]  # Highest frequency channel (0.8 GHz)
        
        # Compute second moment (variance) to measure broadening
        # Broader profiles have larger variance
        def profile_variance(prof, time_axis):
            if np.sum(prof) <= 0:
                return 0
            mean_t = np.sum(time_axis * prof) / np.sum(prof)
            var = np.sum((time_axis - mean_t)**2 * prof) / np.sum(prof)
            return var
        
        var_low = profile_variance(low_freq_profile, simple_time)
        var_high = profile_variance(high_freq_profile, simple_time)
        
        # Low frequency should be broader (larger variance due to more scattering)
        assert var_low > var_high * 0.9  # Allow some tolerance

    def test_spectral_index(self, simple_time, simple_freq):
        """Test that spectral index affects amplitude correctly."""
        model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        
        # Use M1 with intrinsic width so the pulse is properly resolved
        p_steep = FRBParams(c0=1.0, t0=0.0, gamma=-2.0, zeta=0.2)
        p_flat = FRBParams(c0=1.0, t0=0.0, gamma=0.0, zeta=0.2)
        
        out_steep = model(p_steep, "M1")
        out_flat = model(p_flat, "M1")
        
        # Integrate over time (mask any numerical noise)
        spec_steep = np.clip(np.sum(out_steep, axis=1), 1e-10, None)
        spec_flat = np.clip(np.sum(out_flat, axis=1), 1e-10, None)
        
        # For steep spectrum (gamma=-2), high freq should be brighter than low freq
        # For flat spectrum (gamma=0), all frequencies should be similar
        # Ratio = high_freq / low_freq
        ratio_steep = spec_steep[0] / spec_steep[-1]
        ratio_flat = spec_flat[0] / spec_flat[-1]
        
        # Steep spectrum should have ratio further from 1 than flat
        assert abs(ratio_steep - 1) > abs(ratio_flat - 1) * 0.5

    def test_no_nans(self, simple_model, simple_params):
        """Test that model output contains no NaN values."""
        output = simple_model(simple_params, "M3")
        assert not np.any(np.isnan(output))

    def test_positive_output(self, simple_model, simple_params):
        """Test that model output is essentially non-negative (allowing numerical noise)."""
        output = simple_model(simple_params, "M3")
        # Allow tiny negative values from FFT numerical noise (< 1e-10)
        assert np.all(output >= -1e-10)


# ============================================================================
# FRBModel Noise Estimation Tests
# ============================================================================

class TestNoiseEstimation:
    """Tests for noise estimation in FRBModel."""

    def test_noise_estimation_from_data(self, simple_time, simple_freq, simple_params):
        """Test that noise is estimated correctly from off-pulse data."""
        np.random.seed(42)
        true_noise = 0.1
        
        # Create model with data
        model_gen = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        clean = model_gen(simple_params, "M3")
        noisy = clean + np.random.normal(0, true_noise, clean.shape)
        
        model = FRBModel(
            time=simple_time, freq=simple_freq, data=noisy, dm_init=0.0, df_MHz=0.39
        )
        
        # Estimated noise should be close to true noise
        assert_allclose(np.median(model.noise_std), true_noise, rtol=0.3)


# ============================================================================
# FRBFitter Tests
# ============================================================================

class TestFRBFitter:
    """Tests for FRBFitter MCMC sampler."""

    def test_log_param_detection_simple(self):
        """Test _is_log_param correctly identifies log-space parameters."""
        model = FRBModel(
            time=np.linspace(-5, 5, 64),
            freq=np.linspace(0.4, 0.8, 16),
            dm_init=0.0,
            df_MHz=0.39,
        )
        priors = {
            "c0": (0.1, 10.0),
            "t0": (-5.0, 5.0),
            "gamma": (-3.0, 0.0),
            "zeta": (0.01, 1.0),
            "tau_1ghz": (0.01, 5.0),
            "alpha": (3.0, 5.0),
            "delta_dm": (-0.5, 0.5),
        }
        fitter = FRBFitter(model, priors, n_steps=10, sample_log_params=True)
        
        # Base log params
        assert fitter._is_log_param("c0") is True
        assert fitter._is_log_param("zeta") is True
        assert fitter._is_log_param("tau_1ghz") is True
        
        # Non-log params
        assert fitter._is_log_param("t0") is False
        assert fitter._is_log_param("gamma") is False
        assert fitter._is_log_param("alpha") is False
        assert fitter._is_log_param("delta_dm") is False

    def test_log_param_detection_multicomp(self):
        """Test _is_log_param handles multi-component parameter names."""
        model = FRBModel(
            time=np.linspace(-5, 5, 64),
            freq=np.linspace(0.4, 0.8, 16),
            dm_init=0.0,
            df_MHz=0.39,
        )
        priors = {
            "gamma": (-3.0, 0.0),
            "tau_1ghz": (0.01, 5.0),
            "alpha": (3.0, 5.0),
            "delta_dm": (-0.5, 0.5),
            "c0_1": (0.1, 10.0),
            "t0_1": (-5.0, 5.0),
            "zeta_1": (0.01, 1.0),
            "c0_2": (0.1, 10.0),
            "t0_2": (-5.0, 5.0),
            "zeta_2": (0.01, 1.0),
        }
        fitter = FRBFitter(model, priors, n_steps=10, sample_log_params=True)
        
        # Multi-component log params
        assert fitter._is_log_param("c0_1") is True
        assert fitter._is_log_param("c0_2") is True
        assert fitter._is_log_param("zeta_1") is True
        assert fitter._is_log_param("zeta_2") is True
        
        # Multi-component non-log params
        assert fitter._is_log_param("t0_1") is False
        assert fitter._is_log_param("t0_2") is False


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestBuildPriors:
    """Tests for build_priors helper function."""

    def test_basic_priors(self, simple_params):
        """Test that priors are built with correct structure."""
        priors, use_logw = build_priors(simple_params, scale=3.0)
        
        # Check all parameters are present
        assert "c0" in priors
        assert "t0" in priors
        assert "gamma" in priors
        assert "zeta" in priors
        assert "tau_1ghz" in priors
        assert "alpha" in priors
        assert "delta_dm" in priors
        
        # Check bounds are tuples
        for name, bounds in priors.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] < bounds[1]

    def test_positive_params_have_positive_lower_bounds(self, simple_params):
        """Test that positive-definite parameters have positive lower bounds."""
        priors, _ = build_priors(simple_params, scale=3.0)
        
        for name in _POSITIVE:
            assert priors[name][0] > 0, f"{name} should have positive lower bound"


class TestComputeBIC:
    """Tests for BIC computation."""

    def test_bic_formula(self):
        """Test BIC formula implementation."""
        logL = -100.0
        k = 5
        n = 1000
        
        expected = -2 * logL + k * np.log(n)
        actual = compute_bic(logL, k, n)
        
        assert_allclose(actual, expected)

    def test_bic_prefers_simpler_model(self):
        """Test that BIC penalizes model complexity appropriately."""
        # Same likelihood but different number of parameters
        logL = -100.0
        n = 1000
        
        bic_simple = compute_bic(logL, k=3, n=n)
        bic_complex = compute_bic(logL, k=7, n=n)
        
        # Simpler model should have lower BIC
        assert bic_simple < bic_complex


class TestDownsample:
    """Tests for downsample helper function."""

    def test_identity_downsample(self):
        """Test that f_factor=1, t_factor=1 returns unchanged array."""
        data = np.random.rand(64, 256)
        result = downsample(data, f_factor=1, t_factor=1)
        assert_allclose(result, data)

    def test_downsample_shape(self):
        """Test output shape after downsampling."""
        data = np.random.rand(64, 256)
        result = downsample(data, f_factor=4, t_factor=2)
        assert result.shape == (16, 128)

    def test_downsample_mean_preserved(self):
        """Test that overall mean is approximately preserved."""
        data = np.random.rand(64, 256)
        result = downsample(data, f_factor=4, t_factor=4)
        assert_allclose(np.mean(data), np.mean(result), rtol=0.1)


class TestGelmanRubin:
    """Tests for Gelman-Rubin convergence diagnostic."""

    def test_converged_chains(self):
        """Test R̂ ≈ 1 for well-converged chains."""
        # Create mock sampler with converged chains
        class MockSampler:
            def get_chain(self, discard=0):
                # 200 steps, 16 walkers, 2 params - all sampling from same distribution
                np.random.seed(42)
                return np.random.normal(0, 1, (200, 16, 2))
        
        result = gelman_rubin(MockSampler(), discard=0)
        
        assert result['converged'] is True
        assert result['max_rhat'] < 1.1

    def test_unconverged_chains(self):
        """Test R̂ > 1.1 for unconverged chains."""
        class MockSampler:
            def get_chain(self, discard=0):
                # Each walker samples from a different mean
                np.random.seed(42)
                chain = np.zeros((200, 16, 2))
                for i in range(16):
                    chain[:, i, :] = np.random.normal(i * 0.5, 0.1, (200, 2))
                return chain
        
        result = gelman_rubin(MockSampler(), discard=0)
        
        assert result['converged'] is False
        assert result['max_rhat'] > 1.1

    def test_too_few_steps(self):
        """Test handling of chains with too few steps."""
        class MockSampler:
            def get_chain(self, discard=0):
                return np.random.rand(5, 8, 2)  # Only 5 steps
        
        result = gelman_rubin(MockSampler(), discard=0)
        
        assert result['converged'] is False
        assert 'warning' in result


class TestGoodnessOfFit:
    """Tests for goodness_of_fit function."""

    def test_perfect_fit(self):
        """Test χ² ≈ 1 for perfect fit with known noise."""
        np.random.seed(42)
        n_freq, n_time = 64, 256
        noise_std = np.ones(n_freq) * 0.1
        
        true_model = np.random.rand(n_freq, n_time)
        data = true_model + np.random.normal(0, 0.1, (n_freq, n_time))
        
        gof = goodness_of_fit(data, true_model, noise_std, n_params=5)
        
        # χ²/dof should be close to 1
        assert 0.8 < gof['chi2_reduced'] < 1.2

    def test_poor_fit(self):
        """Test χ² >> 1 for poor fit."""
        np.random.seed(42)
        n_freq, n_time = 64, 256
        noise_std = np.ones(n_freq) * 0.1
        
        data = np.random.rand(n_freq, n_time) + 1.0  # Signal at ~1
        bad_model = np.zeros((n_freq, n_time))  # Model at 0
        
        gof = goodness_of_fit(data, bad_model, noise_std, n_params=5)
        
        # χ²/dof should be >> 1
        assert gof['chi2_reduced'] > 10


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full fitting pipeline."""

    @pytest.mark.slow
    def test_simple_recovery(self, simple_time, simple_freq, simple_params):
        """Test that we can recover simple parameters from synthetic data."""
        np.random.seed(42)
        
        # Generate data
        model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        clean = model(simple_params, "M3")
        noisy = clean + np.random.normal(0, 0.02, clean.shape)
        
        # Create model with data
        model_fit = FRBModel(
            time=simple_time, freq=simple_freq, data=noisy, dm_init=0.0, df_MHz=0.39
        )
        
        # Build priors
        init_guess = FRBParams(c0=0.8, t0=0.1, gamma=-1.5, zeta=0.15, tau_1ghz=0.4, alpha=4.5, delta_dm=0.0)
        priors, use_logw = build_priors(init_guess, scale=3.0, log_weight_pos=True)
        
        # Fit
        fitter = FRBFitter(
            model_fit, priors, n_steps=100, pool=None,
            log_weight_pos=use_logw, sample_log_params=True
        )
        sampler = fitter.sample(init_guess, model_key="M3")
        
        # Get best fit
        chain = sampler.get_chain(discard=50, flat=True)
        best_idx = np.argmax(sampler.get_log_prob(discard=50, flat=True))
        best_fit = FRBParams.from_sequence(chain[best_idx], "M3")
        
        # Check recovery (loose tolerance due to short chain)
        assert abs(best_fit.t0 - simple_params.t0) < 0.5
        assert abs(best_fit.gamma - simple_params.gamma) < 0.5


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_tau(self, simple_time, simple_freq):
        """Test model works with tau_1ghz=0 (no scattering)."""
        model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        p = FRBParams(c0=1.0, t0=0.0, gamma=-1.6, zeta=0.1, tau_1ghz=0.0)
        output = model(p, "M3")
        
        assert not np.any(np.isnan(output))
        assert np.all(output >= 0)

    def test_very_small_zeta(self, simple_time, simple_freq):
        """Test model works with very small intrinsic width."""
        model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        p = FRBParams(c0=1.0, t0=0.0, gamma=-1.6, zeta=1e-6, tau_1ghz=0.1)
        output = model(p, "M3")
        
        assert not np.any(np.isnan(output))
        assert np.all(output >= 0)

    def test_large_tau(self, simple_time, simple_freq):
        """Test model works with large scattering time."""
        model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        p = FRBParams(c0=1.0, t0=0.0, gamma=-1.6, zeta=0.1, tau_1ghz=10.0, alpha=4.0)
        output = model(p, "M3")
        
        assert not np.any(np.isnan(output))
        # Allow tiny negative values from FFT numerical noise
        assert np.all(output >= -1e-10)

    def test_negative_gamma(self, simple_time, simple_freq):
        """Test model works with negative spectral index."""
        model = FRBModel(time=simple_time, freq=simple_freq, dm_init=0.0, df_MHz=0.39)
        p = FRBParams(c0=1.0, t0=0.0, gamma=-3.0)
        output = model(p, "M0")
        
        assert not np.any(np.isnan(output))
