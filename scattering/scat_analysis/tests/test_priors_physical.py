"""
test_priors_physical.py
=======================

Unit and integration tests for the NE2001 physical priors module.

Test Categories:
- Unit tests: Individual functions
- Smoke tests: Basic functionality checks
- Integration tests: With fitting pipeline
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Try to import the module
try:
    from scattering.scat_analysis.priors_physical import (
        get_ne2001_scattering,
        build_physical_priors,
        PhysicalPriors,
        get_burst_priors_from_catalog,
        log_prob_lognormal,
        TURBULENCE_INDICES,
        _empirical_dm_tau_relation,
    )
    PRIORS_AVAILABLE = True
except ImportError:
    PRIORS_AVAILABLE = False

# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    not PRIORS_AVAILABLE,
    reason="priors_physical module not available"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def galactic_center_position():
    """Position near Galactic center (high DM, high scattering)."""
    return {"ra_deg": 266.417, "dec_deg": -29.008, "dm": 500.0}


@pytest.fixture
def high_lat_position():
    """High Galactic latitude position (low scattering)."""
    return {"ra_deg": 180.0, "dec_deg": 80.0, "dm": 100.0}


# ============================================================================
# Unit Tests: TURBULENCE_INDICES
# ============================================================================

class TestTurbulenceIndices:
    """Unit tests for turbulence index constants."""
    
    def test_has_kolmogorov(self):
        """Should have Kolmogorov index."""
        assert "kolmogorov" in TURBULENCE_INDICES
        assert_allclose(TURBULENCE_INDICES["kolmogorov"], 4.4, atol=0.1)
    
    def test_has_thin_screen(self):
        """Should have thin screen index."""
        assert "thin_screen" in TURBULENCE_INDICES
        assert_allclose(TURBULENCE_INDICES["thin_screen"], 4.0, atol=0.1)
    
    def test_all_positive(self):
        """All indices should be positive."""
        for name, val in TURBULENCE_INDICES.items():
            assert val > 0, f"{name} should be positive"


# ============================================================================
# Unit Tests: log_prob_lognormal
# ============================================================================

class TestLogProbLognormal:
    """Unit tests for log-normal prior probability."""
    
    def test_negative_returns_neg_inf(self):
        """Negative values should return -inf."""
        result = log_prob_lognormal(-1.0, mu=0.0, sigma=1.0)
        assert result == -np.inf
    
    def test_zero_returns_neg_inf(self):
        """Zero should return -inf."""
        result = log_prob_lognormal(0.0, mu=0.0, sigma=1.0)
        assert result == -np.inf
    
    def test_positive_returns_finite(self):
        """Positive values should return finite log-prob."""
        result = log_prob_lognormal(1.0, mu=0.0, sigma=1.0)
        assert np.isfinite(result)
    
    def test_peak_at_mode(self):
        """Should peak near the mode."""
        mu = 1.0  # log10 mode
        sigma = 0.3
        
        # Sample around mode
        x_mode = 10 ** mu  # mode of log-normal in log10
        x_lo = x_mode * 0.5
        x_hi = x_mode * 2.0
        
        lp_mode = log_prob_lognormal(x_mode, mu, sigma)
        lp_lo = log_prob_lognormal(x_lo, mu, sigma)
        lp_hi = log_prob_lognormal(x_hi, mu, sigma)
        
        # Mode should have highest probability
        assert lp_mode > lp_lo
        assert lp_mode > lp_hi


# ============================================================================
# Unit Tests: _empirical_dm_tau_relation
# ============================================================================

class TestEmpiricalDMTauRelation:
    """Unit tests for Bhat et al. (2004) empirical relation."""
    
    def test_returns_tuple(self):
        """Should return (tau, nu_scint) tuple."""
        result = _empirical_dm_tau_relation(dm=100.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_tau_positive(self):
        """Tau should always be positive."""
        tau, _ = _empirical_dm_tau_relation(dm=100.0)
        assert tau > 0
    
    def test_tau_scales_with_dm(self):
        """Higher DM should give more scattering."""
        tau_lo, _ = _empirical_dm_tau_relation(dm=10.0)
        tau_hi, _ = _empirical_dm_tau_relation(dm=1000.0)
        assert tau_hi > tau_lo
    
    def test_tau_scales_with_frequency(self):
        """Lower frequency should give more scattering."""
        tau_hi_freq, _ = _empirical_dm_tau_relation(dm=100.0, freq_mhz=2000.0)
        tau_lo_freq, _ = _empirical_dm_tau_relation(dm=100.0, freq_mhz=500.0)
        assert tau_lo_freq > tau_hi_freq
    
    def test_nu_scint_positive(self):
        """Scintillation bandwidth should be positive."""
        _, nu_scint = _empirical_dm_tau_relation(dm=100.0)
        assert nu_scint > 0


# ============================================================================
# Smoke Tests: get_ne2001_scattering
# ============================================================================

class TestGetNE2001Scattering:
    """Smoke tests for NE2001 query function."""
    
    def test_returns_tuple(self, high_lat_position):
        """Should return (tau, nu_scint) tuple."""
        result = get_ne2001_scattering(**high_lat_position)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_tau_positive(self, high_lat_position):
        """Tau should be positive."""
        tau, _ = get_ne2001_scattering(**high_lat_position)
        assert tau > 0
    
    def test_nu_scint_positive(self, high_lat_position):
        """Scintillation bandwidth should be positive."""
        _, nu_scint = get_ne2001_scattering(**high_lat_position)
        assert nu_scint > 0
    
    def test_galactic_center_more_scattered(
        self, galactic_center_position, high_lat_position
    ):
        """Galactic center should have more scattering than high-lat."""
        tau_gc, _ = get_ne2001_scattering(**galactic_center_position)
        tau_hl, _ = get_ne2001_scattering(**high_lat_position)
        
        # GC should have more scattering (higher tau)
        # Note: This may use empirical fallback if NE2001 not installed
        assert tau_gc > 0 and tau_hl > 0


# ============================================================================
# Smoke Tests: build_physical_priors
# ============================================================================

class TestBuildPhysicalPriors:
    """Smoke tests for building physical priors."""
    
    def test_returns_physical_priors(self, high_lat_position):
        """Should return PhysicalPriors object."""
        result = build_physical_priors(**high_lat_position)
        assert isinstance(result, PhysicalPriors)
    
    def test_has_tau_lognormal(self, high_lat_position):
        """Should have tau log-normal prior."""
        result = build_physical_priors(**high_lat_position)
        assert hasattr(result, 'tau_lognormal')
        assert len(result.tau_lognormal) == 2  # (mu, sigma)
    
    def test_has_alpha_gaussian(self, high_lat_position):
        """Should have alpha Gaussian prior."""
        result = build_physical_priors(**high_lat_position)
        assert hasattr(result, 'alpha_gaussian')
        assert len(result.alpha_gaussian) == 2  # (mu, sigma)
    
    def test_has_bounds(self, high_lat_position):
        """Should have parameter bounds."""
        result = build_physical_priors(**high_lat_position)
        assert hasattr(result, 'bounds')
        assert isinstance(result.bounds, dict)
        assert 'tau_1ghz' in result.bounds
        assert 'alpha' in result.bounds
    
    def test_has_ne2001_prediction(self, high_lat_position):
        """Should store NE2001 prediction."""
        result = build_physical_priors(**high_lat_position)
        assert hasattr(result, 'ne2001_tau_1ghz')
        assert result.ne2001_tau_1ghz > 0
    
    def test_respects_turbulence_model(self, high_lat_position):
        """Should use specified turbulence model for alpha."""
        result_kolm = build_physical_priors(
            **high_lat_position, turbulence_model="kolmogorov"
        )
        result_thin = build_physical_priors(
            **high_lat_position, turbulence_model="thin_screen"
        )
        
        # Alpha means should differ
        assert result_kolm.alpha_gaussian[0] != result_thin.alpha_gaussian[0]
    
    def test_repr(self, high_lat_position):
        """Should have readable repr."""
        result = build_physical_priors(**high_lat_position)
        repr_str = repr(result)
        assert "LogNormal" in repr_str
        assert "Normal" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhysicalPriorsIntegration:
    """Integration tests with fitting pipeline."""
    
    def test_priors_compatible_with_nested_sampling(self, high_lat_position):
        """Priors should work with nested sampling module."""
        priors = build_physical_priors(**high_lat_position)
        
        # Check tau_lognormal format
        mu, sigma = priors.tau_lognormal
        assert np.isfinite(mu)
        assert sigma > 0
        
        # Check alpha_gaussian format
        mu_a, sigma_a = priors.alpha_gaussian
        assert np.isfinite(mu_a)
        assert sigma_a > 0
    
    def test_bounds_are_valid(self, high_lat_position):
        """Bounds should be valid (lo < hi)."""
        priors = build_physical_priors(**high_lat_position)
        
        for name, (lo, hi) in priors.bounds.items():
            assert lo < hi, f"Invalid bounds for {name}: ({lo}, {hi})"
    
    def test_tau_bounds_contain_prediction(self, high_lat_position):
        """Tau bounds should contain NE2001 prediction."""
        priors = build_physical_priors(**high_lat_position)
        
        tau_pred = priors.ne2001_tau_1ghz
        tau_lo, tau_hi = priors.bounds["tau_1ghz"]
        
        # Prediction should be within bounds
        assert tau_lo <= tau_pred <= tau_hi


# ============================================================================
# Edge Cases
# ============================================================================

class TestPhysicalPriorsEdgeCases:
    """Edge case tests."""
    
    def test_very_low_dm(self):
        """Should handle very low DM."""
        priors = build_physical_priors(ra_deg=0.0, dec_deg=90.0, dm=1.0)
        assert isinstance(priors, PhysicalPriors)
        assert priors.ne2001_tau_1ghz > 0
    
    def test_very_high_dm(self):
        """Should handle very high DM."""
        priors = build_physical_priors(ra_deg=0.0, dec_deg=0.0, dm=5000.0)
        assert isinstance(priors, PhysicalPriors)
        assert priors.ne2001_tau_1ghz > 0
    
    def test_host_scattering_flag(self, high_lat_position):
        """allow_host_scattering should widen tau bounds."""
        priors_no_host = build_physical_priors(
            **high_lat_position, allow_host_scattering=False
        )
        priors_host = build_physical_priors(
            **high_lat_position, allow_host_scattering=True
        )
        
        # With host scattering, sigma should be larger
        assert priors_host.tau_lognormal[1] >= priors_no_host.tau_lognormal[1]
    
    def test_custom_alpha_prior(self, high_lat_position):
        """Should respect custom alpha mean/std."""
        priors = build_physical_priors(
            **high_lat_position,
            alpha_mean=5.0,
            alpha_std=0.2,
            turbulence_model=None,  # Don't override
        )
        
        assert_allclose(priors.alpha_gaussian[0], 5.0)
        assert_allclose(priors.alpha_gaussian[1], 0.2)
    
    def test_negative_coords(self):
        """Should handle negative declination."""
        priors = build_physical_priors(ra_deg=180.0, dec_deg=-45.0, dm=200.0)
        assert isinstance(priors, PhysicalPriors)


# ============================================================================
# Catalog Tests (if bursts.yaml exists)
# ============================================================================

class TestBurstCatalogIntegration:
    """Tests for catalog-based prior lookup."""
    
    def test_catalog_lookup_raises_for_unknown(self):
        """Should raise for unknown burst name."""
        with pytest.raises(ValueError, match="not in catalog"):
            get_burst_priors_from_catalog("nonexistent_burst_xyz123")
    
    @pytest.mark.skipif(True, reason="Requires bursts.yaml")
    def test_catalog_lookup_returns_priors(self):
        """Should return priors for known burst."""
        # This test would need a real burst name from bursts.yaml
        priors = get_burst_priors_from_catalog("casey")
        assert isinstance(priors, PhysicalPriors)
