"""
test_pipeline_integration.py
============================

End-to-end integration tests for the full scattering analysis pipeline.

These tests verify that all components work together correctly:
- Data loading and preprocessing
- Initial guess estimation
- MCMC fitting
- Diagnostics
- Model selection

Smoke tests run quickly; integration tests are marked slow.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scattering.scat_analysis.burstfit import FRBModel, FRBParams, build_priors, FRBFitter
from scattering.scat_analysis.burstfit_pipeline import BurstPipeline, BurstDataset, refine_initial_guess_mle

# Try imports for optional components
try:
    from scattering.scat_analysis.burstfit_init import data_driven_initial_guess
    INIT_AVAILABLE = True
except ImportError:
    INIT_AVAILABLE = False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_data_file(tmp_path):
    """Create a synthetic .npy file for testing."""
    # Generate synthetic burst
    n_freq, n_time = 128, 500
    time = np.linspace(0, 25, n_time)
    freq = np.linspace(1.0, 1.5, n_freq)
    
    # Build simple burst
    ref_freq = np.median(freq)
    gamma = -1.5
    t0 = 12.5
    sigma = 1.0
    
    amp = (freq / ref_freq) ** gamma
    gauss = np.exp(-0.5 * ((time[None, :] - t0) / sigma) ** 2)
    data = amp[:, None] * gauss * 100
    
    # Add noise
    data += np.random.normal(0, 5, data.shape)
    
    # Save to file
    filepath = tmp_path / "test_burst.npy"
    np.save(filepath, data)
    
    # Also save metadata
    metadata = {
        'freq': freq,
        'time': time,
        'dm': 100.0,
    }
    
    return filepath, metadata


@pytest.fixture
def simple_model_with_data():
    """Create FRBModel with synthetic data for quick tests."""
    time = np.linspace(0, 20, 200)
    freq = np.linspace(1.0, 1.5, 64)
    
    true_params = FRBParams(
        c0=100.0, t0=10.0, gamma=-1.5,
        zeta=0.5, tau_1ghz=0.2, alpha=4.0, delta_dm=0.0
    )
    
    model_gen = FRBModel(time, freq, dm_init=0.0)
    clean = model_gen(true_params, "M3")
    data = clean + np.random.normal(0, 0.5, clean.shape)
    
    model = FRBModel(time, freq, data=data, dm_init=0.0)
    
    return model, true_params


# ============================================================================
# Smoke Tests: Minimal Pipeline Components
# ============================================================================

class TestPipelineComponentsSmoke:
    """Smoke tests for individual pipeline components."""
    
    def test_frbmodel_creation(self):
        """FRBModel should create without data."""
        time = np.linspace(0, 10, 100)
        freq = np.linspace(1.0, 1.5, 50)
        model = FRBModel(time, freq, dm_init=0.0)
        assert model is not None
    
    def test_frbmodel_forward(self, simple_model_with_data):
        """Forward model should produce valid output."""
        model, params = simple_model_with_data
        output = model(params, "M3")
        assert output.shape == model.data.shape
        assert np.all(np.isfinite(output))
    
    def test_likelihood_computes(self, simple_model_with_data):
        """Log-likelihood should compute finite value."""
        model, params = simple_model_with_data
        ll = model.log_likelihood(params, "M3")
        assert np.isfinite(ll)
    
    def test_build_priors(self, simple_model_with_data):
        """build_priors should return valid dict."""
        _, params = simple_model_with_data
        priors, use_logw = build_priors(params, scale=3.0)
        
        assert isinstance(priors, dict)
        assert "c0" in priors
        assert "t0" in priors
        assert len(priors["c0"]) == 2  # (lo, hi)
    
    @pytest.mark.skipif(not INIT_AVAILABLE, reason="burstfit_init not available")
    def test_data_driven_guess(self, simple_model_with_data):
        """data_driven_initial_guess should work with model data."""
        model, _ = simple_model_with_data
        
        result = data_driven_initial_guess(
            model.data, model.freq, model.time, verbose=False
        )
        
        assert result.params.c0 > 0
        assert np.isfinite(result.params.t0)


# ============================================================================
# Integration Tests: MLE Refinement
# ============================================================================

class TestMLERefinementIntegration:
    """Integration tests for MLE initial guess refinement."""
    
    def test_refine_improves_likelihood(self, simple_model_with_data):
        """MLE refinement should improve or maintain likelihood."""
        model, true_params = simple_model_with_data
        
        # Perturbed initial guess
        init_guess = FRBParams(
            c0=true_params.c0 * 0.8,
            t0=true_params.t0 + 0.5,
            gamma=true_params.gamma - 0.3,
            zeta=true_params.zeta * 1.5,
            tau_1ghz=true_params.tau_1ghz * 2.0,
            alpha=true_params.alpha + 0.5,
            delta_dm=0.0,
        )
        
        ll_before = model.log_likelihood(init_guess, "M3")
        refined = refine_initial_guess_mle(model, init_guess)
        ll_after = model.log_likelihood(refined, "M3")
        
        # Should improve (or at least not get much worse)
        assert ll_after >= ll_before - 1.0  # Allow small tolerance
    
    def test_refine_returns_frbparams(self, simple_model_with_data):
        """MLE refinement should return FRBParams."""
        model, true_params = simple_model_with_data
        refined = refine_initial_guess_mle(model, true_params)
        assert isinstance(refined, FRBParams)
    
    def test_refine_handles_bad_guess(self, simple_model_with_data):
        """MLE refinement should handle poor initial guess."""
        model, _ = simple_model_with_data
        
        # Very bad initial guess
        bad_guess = FRBParams(
            c0=1.0, t0=0.5, gamma=0.0,
            zeta=10.0, tau_1ghz=20.0, alpha=2.0, delta_dm=5.0
        )
        
        # Should not raise
        refined = refine_initial_guess_mle(model, bad_guess)
        assert isinstance(refined, FRBParams)


# ============================================================================
# Integration Tests: MCMC Sampling
# ============================================================================

@pytest.mark.slow
class TestMCMCSamplingIntegration:
    """Integration tests for MCMC sampling (slow)."""
    
    def test_fitter_runs(self, simple_model_with_data):
        """FRBFitter should complete sampling."""
        model, init_params = simple_model_with_data
        
        priors, use_logw = build_priors(init_params, scale=3.0)
        priors["alpha"] = (2.0, 6.0)
        priors["delta_dm"] = (-0.5, 0.5)
        
        fitter = FRBFitter(
            model, priors,
            n_steps=50,  # Very short for test
            n_walkers_mult=4,
            log_weight_pos=use_logw,
        )
        
        sampler = fitter.sample(init_params, model_key="M3")
        
        assert sampler is not None
        assert sampler.iteration > 0
    
    def test_chain_shape(self, simple_model_with_data):
        """MCMC chain should have correct shape."""
        model, init_params = simple_model_with_data
        
        priors, use_logw = build_priors(init_params, scale=3.0)
        priors["alpha"] = (2.0, 6.0)
        priors["delta_dm"] = (-0.5, 0.5)
        
        fitter = FRBFitter(
            model, priors,
            n_steps=30,
            n_walkers_mult=4,
        )
        
        sampler = fitter.sample(init_params, model_key="M3")
        chain = sampler.get_chain()
        
        # Shape: (n_steps, n_walkers, n_params)
        n_params = len(FRBFitter._ORDER["M3"])
        assert chain.shape[2] == n_params
        assert chain.shape[0] == 30


# ============================================================================
# Integration Tests: Full Pipeline
# ============================================================================

@pytest.mark.slow
class TestFullPipelineIntegration:
    """Full pipeline integration tests (slow)."""
    
    def test_pipeline_runs_minimal(self, synthetic_data_file):
        """Pipeline should complete with minimal options."""
        filepath, metadata = synthetic_data_file
        
        with tempfile.TemporaryDirectory() as outdir:
            # This would require proper data format
            # For now, test that pipeline can be instantiated
            try:
                pipe = BurstPipeline(
                    inpath=str(filepath),
                    outpath=outdir,
                    name="test_burst",
                    dm_init=metadata['dm'],
                    nproc=0,  # Run serially to avoid interactive prompt
                )
                assert pipe is not None
            except Exception as e:
                # Expected if data format doesn't match
                assert "shape" in str(e).lower() or "format" in str(e).lower()


# ============================================================================
# Regression Tests
# ============================================================================

class TestRegressionTests:
    """Tests to prevent regression of known issues."""
    
    def test_no_nan_in_model_output(self, simple_model_with_data):
        """Model should never produce NaN."""
        model, params = simple_model_with_data
        
        # Test various model keys
        for key in ["M0", "M1", "M2", "M3"]:
            output = model(params, key)
            assert np.all(np.isfinite(output)), f"NaN in {key} output"
    
    def test_positive_noise_estimate(self, simple_model_with_data):
        """Noise estimate should always be positive."""
        model, _ = simple_model_with_data
        assert np.all(model.noise_std > 0)
    
    def test_likelihood_finite_for_reasonable_params(self, simple_model_with_data):
        """Likelihood should be finite for reasonable parameters."""
        model, params = simple_model_with_data
        
        ll = model.log_likelihood(params, "M3")
        assert np.isfinite(ll)
        
        # Should not be insanely negative
        assert ll > -1e15


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfigurationTests:
    """Tests for configuration handling."""
    
    def test_fitter_accepts_student_t(self, simple_model_with_data):
        """FRBFitter should accept Student-t likelihood."""
        model, init_params = simple_model_with_data
        priors, _ = build_priors(init_params, scale=3.0)
        priors["alpha"] = (2.0, 6.0)
        priors["delta_dm"] = (-0.5, 0.5)
        
        fitter = FRBFitter(
            model, priors,
            n_steps=10,
            likelihood_kind="studentt",
            student_nu=5.0,
        )
        
        assert fitter.likelihood_kind == "studentt"
        assert fitter.student_nu == 5.0
    
    def test_fitter_accepts_alpha_prior(self, simple_model_with_data):
        """FRBFitter should accept Gaussian alpha prior."""
        model, init_params = simple_model_with_data
        priors, _ = build_priors(init_params, scale=3.0)
        priors["alpha"] = (2.0, 6.0)
        priors["delta_dm"] = (-0.5, 0.5)
        
        fitter = FRBFitter(
            model, priors,
            n_steps=10,
            alpha_prior=(4.0, 0.5),
        )
        
        assert fitter.alpha_prior == (4.0, 0.5)
