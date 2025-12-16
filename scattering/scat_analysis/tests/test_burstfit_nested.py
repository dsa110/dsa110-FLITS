"""
test_burstfit_nested.py
=======================

Unit and integration tests for the nested sampling module.

Test Categories:
- Unit tests: Individual functions
- Smoke tests: Basic functionality with mock data
- Integration tests: Full evidence calculation (slow)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scattering.scat_analysis.burstfit import FRBModel, FRBParams, build_priors

# Try to import nested sampling module
try:
    from scattering.scat_analysis.burstfit_nested import (
        fit_models_evidence,
        fit_single_model_nested,
        NestedSamplingResult,
        interpret_bayes_factor,
        _build_prior_transform,
        _build_log_likelihood,
    )
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False

# Skip all tests if dynesty not installed
pytestmark = pytest.mark.skipif(
    not DYNESTY_AVAILABLE,
    reason="dynesty not installed"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_time():
    """Time axis in ms."""
    return np.linspace(0, 20, 200)


@pytest.fixture
def simple_freq():
    """Frequency axis in GHz."""
    return np.linspace(1.0, 1.5, 64)


@pytest.fixture
def simple_params():
    """Simple FRB parameters."""
    return FRBParams(
        c0=100.0,
        t0=10.0,
        gamma=-1.5,
        zeta=0.5,
        tau_1ghz=0.3,
        alpha=4.0,
        delta_dm=0.0,
    )


@pytest.fixture
def synthetic_model(simple_time, simple_freq, simple_params):
    """FRBModel with synthetic data."""
    model = FRBModel(simple_time, simple_freq, dm_init=0.0)
    
    # Generate synthetic data
    clean_data = model(simple_params, "M3")
    noise = np.random.normal(0, 0.1, clean_data.shape)
    data = clean_data + noise
    
    # Create model with data
    model_with_data = FRBModel(
        simple_time, simple_freq,
        data=data,
        dm_init=0.0,
    )
    
    return model_with_data, simple_params


# ============================================================================
# Unit Tests: interpret_bayes_factor
# ============================================================================

class TestInterpretBayesFactor:
    """Unit tests for Bayes factor interpretation."""
    
    def test_inconclusive(self):
        """ln(BF) < 1 should be inconclusive."""
        result = interpret_bayes_factor(0.5)
        assert "Inconclusive" in result
    
    def test_weak_positive(self):
        """1 < ln(BF) < 2.5 should be weak."""
        result = interpret_bayes_factor(1.5)
        assert "Weak" in result
        assert "favors first model" in result
    
    def test_weak_negative(self):
        """-2.5 < ln(BF) < -1 should be weak."""
        result = interpret_bayes_factor(-1.5)
        assert "Weak" in result
        assert "favors second model" in result
    
    def test_moderate(self):
        """2.5 < |ln(BF)| < 5 should be moderate."""
        result = interpret_bayes_factor(3.5)
        assert "Moderate" in result
    
    def test_strong(self):
        """|ln(BF)| > 5 should be strong."""
        result = interpret_bayes_factor(7.0)
        assert "Strong" in result
    
    def test_custom_model_names(self):
        """Should use custom model names."""
        result = interpret_bayes_factor(3.0, model1="M3", model2="M2")
        assert "M3" in result or "M2" in result


# ============================================================================
# Unit Tests: _build_prior_transform
# ============================================================================

class TestBuildPriorTransform:
    """Unit tests for prior transform builder."""
    
    def test_returns_callable(self):
        """Should return a callable function."""
        priors = {"c0": (0.1, 100), "t0": (0, 20)}
        transform = _build_prior_transform(priors, ("c0", "t0"))
        assert callable(transform)
    
    def test_transforms_unit_cube(self):
        """Should map [0,1] to prior bounds."""
        priors = {"c0": (0.0, 100.0), "t0": (0.0, 20.0)}
        transform = _build_prior_transform(priors, ("c0", "t0"), log_params=())
        
        # u = 0 should give lower bound
        result_lo = transform(np.array([0.0, 0.0]))
        assert_allclose(result_lo, [0.0, 0.0], atol=1e-10)
        
        # u = 1 should give upper bound
        result_hi = transform(np.array([1.0, 1.0]))
        assert_allclose(result_hi, [100.0, 20.0], atol=1e-10)
        
        # u = 0.5 should give midpoint (for uniform)
        result_mid = transform(np.array([0.5, 0.5]))
        assert_allclose(result_mid, [50.0, 10.0], atol=1e-10)
    
    def test_log_uniform_for_positive_params(self):
        """Should use log-uniform for positive params."""
        priors = {"tau_1ghz": (0.01, 10.0)}
        transform = _build_prior_transform(
            priors, ("tau_1ghz",), log_params=("tau_1ghz",)
        )
        
        # u = 0.5 should give geometric mean
        result = transform(np.array([0.5]))
        expected = np.sqrt(0.01 * 10.0)  # Geometric mean
        assert_allclose(result, [expected], rtol=0.01)


# ============================================================================
# Unit Tests: _build_log_likelihood
# ============================================================================

class TestBuildLogLikelihood:
    """Unit tests for log-likelihood builder."""
    
    def test_returns_callable(self, synthetic_model):
        """Should return a callable function."""
        model, params = synthetic_model
        loglik = _build_log_likelihood(
            model, "M3", ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm")
        )
        assert callable(loglik)
    
    def test_returns_finite_value(self, synthetic_model):
        """Should return finite log-likelihood."""
        model, params = synthetic_model
        loglik = _build_log_likelihood(
            model, "M3", ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm")
        )
        
        theta = params.to_sequence("M3")
        ll = loglik(np.array(theta))
        assert np.isfinite(ll)
    
    def test_handles_bad_params(self, synthetic_model):
        """Should return very negative value for bad params."""
        model, _ = synthetic_model
        loglik = _build_log_likelihood(
            model, "M3", ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm")
        )
        
        # Very bad parameters
        theta = [0, 0, 0, 0, 0, 0, 0]
        ll = loglik(np.array(theta))
        assert ll < -1e50 or not np.isfinite(ll)


# ============================================================================
# Smoke Tests: NestedSamplingResult
# ============================================================================

class TestNestedSamplingResult:
    """Smoke tests for result container."""
    
    def test_creation(self):
        """Should create result object."""
        result = NestedSamplingResult(
            log_evidence=-100.0,
            log_evidence_err=0.5,
            samples=np.random.rand(100, 3),
            weights=np.ones(100) / 100,
            param_names=("c0", "t0", "gamma"),
            model_key="M0",
            nlive=100,
            ncall=1000,
        )
        assert result.log_evidence == -100.0
        assert result.model_key == "M0"
    
    def test_computes_percentiles(self):
        """Should compute parameter percentiles."""
        samples = np.random.normal(10, 1, (1000, 1))
        result = NestedSamplingResult(
            log_evidence=-100.0,
            log_evidence_err=0.5,
            samples=samples,
            weights=np.ones(1000) / 1000,
            param_names=("c0",),
            model_key="M0",
            nlive=100,
            ncall=1000,
        )
        
        # Check percentiles exist
        assert "c0" in result.percentiles
        assert "median" in result.percentiles["c0"]
        
        # Median should be close to 10
        assert abs(result.percentiles["c0"]["median"] - 10) < 1
    
    def test_repr(self):
        """Should have readable repr."""
        result = NestedSamplingResult(
            log_evidence=-100.0,
            log_evidence_err=0.5,
            samples=np.random.rand(100, 3),
            weights=np.ones(100) / 100,
            param_names=("c0", "t0", "gamma"),
            model_key="M0",
            nlive=100,
            ncall=1000,
        )
        repr_str = repr(result)
        assert "M0" in repr_str
        assert "-100" in repr_str


# ============================================================================
# Integration Tests (Slow - mark for CI skip)
# ============================================================================

@pytest.mark.slow
class TestNestedSamplingIntegration:
    """Integration tests for full nested sampling runs."""
    
    def test_fit_single_model(self, synthetic_model):
        """Should complete single model fit."""
        model, init_params = synthetic_model
        
        result = fit_single_model_nested(
            model=model,
            init=init_params,
            model_key="M0",  # Simplest model
            nlive=50,  # Small for speed
            dlogz=1.0,  # Loose tolerance
            verbose=False,
        )
        
        assert isinstance(result, NestedSamplingResult)
        assert np.isfinite(result.log_evidence)
        assert result.model_key == "M0"
    
    def test_fit_models_evidence(self, synthetic_model):
        """Should compare models and return best."""
        model, init_params = synthetic_model
        
        best_key, results = fit_models_evidence(
            model=model,
            init=init_params,
            model_keys=("M0", "M1"),  # Just 2 models for speed
            nlive=50,
            dlogz=1.0,
            verbose=False,
        )
        
        assert best_key in ("M0", "M1")
        assert "M0" in results
        assert "M1" in results
        assert "bayes_factors" in results
    
    def test_bayes_factors_computed(self, synthetic_model):
        """Should compute Bayes factors."""
        model, init_params = synthetic_model
        
        _, results = fit_models_evidence(
            model=model,
            init=init_params,
            model_keys=("M0", "M1"),
            nlive=50,
            dlogz=1.0,
            verbose=False,
        )
        
        bf = results["bayes_factors"]
        assert "ln_BF_M0_vs_M1" in bf or "ln_BF_M1_vs_M0" in bf


# ============================================================================
# Edge Cases
# ============================================================================

class TestNestedSamplingEdgeCases:
    """Edge case tests."""
    
    def test_with_tau_prior(self, synthetic_model):
        """Should accept tau_prior parameter."""
        model, init_params = synthetic_model
        
        # This should not raise
        loglik = _build_log_likelihood(
            model, "M3",
            ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm"),
            tau_prior=(-1.0, 0.5),  # Log-normal prior
        )
        
        theta = init_params.to_sequence("M3")
        ll = loglik(np.array(theta))
        assert np.isfinite(ll)
    
    def test_with_alpha_prior(self, synthetic_model):
        """Should accept alpha_prior parameter."""
        model, init_params = synthetic_model
        
        loglik = _build_log_likelihood(
            model, "M3",
            ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm"),
            alpha_prior=(4.0, 0.5),  # Gaussian prior
        )
        
        theta = init_params.to_sequence("M3")
        ll = loglik(np.array(theta))
        assert np.isfinite(ll)
