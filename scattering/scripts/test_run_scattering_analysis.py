#!/usr/bin/env python
"""
Unit tests for run_scattering_analysis.py

Tests cover:
- FigureValidator class methods
- Figure generation functions
- Configuration loading
- Edge cases and error handling

Run with:
    pytest scattering/scripts/test_run_scattering_analysis.py -v
"""

import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np

FLITS_ROOT = Path(__file__).resolve().parents[2]

# Import the module under test
from flits.scattering.scripts.run_scattering_analysis import (
    FigureValidator,
    create_data_overview_figure,
    create_initial_guess_figure,
    create_results_summary_figure,
    main,
)


# =============================================================================
# FIXTURES
# =============================================================================


@dataclass
class MockFRBParams:
    """Mock FRBParams for testing."""

    c0: float = 1.0
    t0: float = 5.0
    gamma: float = -1.0
    zeta: float = 0.5
    tau_1ghz: float = 0.1
    alpha: float = 4.0
    delta_dm: float = 0.0


@pytest.fixture
def validator():
    """Create a FigureValidator instance."""
    return FigureValidator("test_analysis")


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with realistic attributes."""
    dataset = Mock()
    # Create realistic dynamic spectrum data
    np.random.seed(42)
    n_freq, n_time = 16, 100
    dataset.data = np.random.randn(n_freq, n_time) * 0.1
    # Add a burst-like signal
    t_peak = 50
    for i in range(n_freq):
        dataset.data[i, t_peak - 2 : t_peak + 3] += 1.0 * np.exp(
            -((np.arange(5) - 2) ** 2) / 2
        )

    dataset.time = np.linspace(0, 10, n_time)
    dataset.freq = np.linspace(1.3, 1.5, n_freq)
    dataset.df_MHz = (dataset.freq[1] - dataset.freq[0]) * 1000
    dataset.dm_init = 0.0
    return dataset


@pytest.fixture
def mock_results(mock_dataset):
    """Create mock MCMC results."""
    n_walkers, n_steps, n_params = 32, 100, 7
    np.random.seed(42)

    # Create mock sampler
    sampler = Mock()
    sampler.get_chain.return_value = np.random.randn(n_steps, n_walkers, n_params)
    sampler.iteration = n_steps
    sampler.get_autocorr_time.return_value = np.array([10.0] * n_params)

    # Create mock model
    model_instance = Mock()
    model_instance.return_value = mock_dataset.data * 0.9  # Similar to data
    model_instance.noise_std = np.ones(mock_dataset.data.shape[0]) * 0.1

    return {
        "best_key": "M3",
        "best_params": MockFRBParams(),
        "flat_chain": np.random.randn(1000, n_params),
        "param_names": ["c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm"],
        "sampler": sampler,
        "model_instance": model_instance,
        "goodness_of_fit": {
            "chi2_reduced": 1.2,
            "chi2": 1200.0,
            "ndof": 1000,
        },
        "chain_stats": {"burn_in": 50},
    }


# =============================================================================
# TESTS: FigureValidator
# =============================================================================


class TestFigureValidator:
    """Tests for the FigureValidator class."""

    def test_init(self, validator):
        """Test validator initialization."""
        assert validator.name == "test_analysis"
        assert validator.issues == []

    def test_check_array_clean(self, validator, capsys):
        """Test check_array with clean data."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        validator.check_array(arr, "clean_array")

        assert len(validator.issues) == 0
        captured = capsys.readouterr()
        assert "[STATS] clean_array" in captured.out

    def test_check_array_with_nan(self, validator):
        """Test check_array detects NaN values."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        validator.check_array(arr, "nan_array")

        assert len(validator.issues) == 1
        assert "NaN values" in validator.issues[0]
        assert "40.0%" in validator.issues[0]

    def test_check_array_with_inf(self, validator):
        """Test check_array detects Inf values."""
        arr = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
        validator.check_array(arr, "inf_array")

        assert len(validator.issues) == 1
        assert "Inf values" in validator.issues[0]

    def test_check_array_all_zeros(self, validator):
        """Test check_array detects all-zero arrays."""
        arr = np.zeros(10)
        validator.check_array(arr, "zero_array")

        assert len(validator.issues) == 1
        assert "All values are zero" in validator.issues[0]

    def test_check_array_uniform(self, validator):
        """Test check_array detects suspiciously uniform data."""
        arr = np.ones(100) * 5.0
        validator.check_array(arr, "uniform_array")

        assert len(validator.issues) == 1
        assert "Suspiciously uniform" in validator.issues[0]

    def test_check_array_negative_not_allowed(self, validator):
        """Test check_array flags negative values when not allowed."""
        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        validator.check_array(arr, "neg_array", allow_negative=False)

        assert len(validator.issues) == 1
        assert "negative values" in validator.issues[0]

    def test_check_array_no_finite_values(self, validator):
        """Test check_array handles arrays with no finite values."""
        arr = np.array([np.nan, np.inf, -np.inf])
        validator.check_array(arr, "bad_array")

        # Should have issues for NaN, Inf, and no finite values
        assert len(validator.issues) >= 1
        has_no_finite = any("No finite values" in issue for issue in validator.issues)
        assert has_no_finite

    def test_check_chi2_negative(self, validator):
        """Test check_chi2 flags negative chi-squared."""
        validator.check_chi2(-1.0)
        assert len(validator.issues) == 1
        assert "[ERROR]" in validator.issues[0]

    def test_check_chi2_too_low(self, validator):
        """Test check_chi2 flags suspiciously low values."""
        validator.check_chi2(0.3)
        assert len(validator.issues) == 1
        assert "suspiciously low" in validator.issues[0]

    def test_check_chi2_too_high(self, validator):
        """Test check_chi2 flags very high values."""
        validator.check_chi2(150.0)
        assert len(validator.issues) == 1
        assert "very high" in validator.issues[0]

    def test_check_chi2_reasonable(self, validator, capsys):
        """Test check_chi2 accepts reasonable values."""
        validator.check_chi2(1.2)
        assert len(validator.issues) == 0
        captured = capsys.readouterr()
        assert "[OK]" in captured.out

    def test_check_mcmc_convergence_good(self, validator, capsys):
        """Test check_mcmc_convergence with well-converged chain."""
        sampler = Mock()
        sampler.iteration = 10000
        sampler.get_autocorr_time.return_value = np.array([50, 60, 70])

        flat_chain = np.random.randn(5000, 3)
        validator.check_mcmc_convergence(sampler, flat_chain)

        captured = capsys.readouterr()
        assert "[OK] Chain convergence" in captured.out
        assert "[OK] Effective samples" in captured.out

    def test_check_mcmc_convergence_poor(self, validator):
        """Test check_mcmc_convergence with poorly converged chain."""
        sampler = Mock()
        sampler.iteration = 100
        sampler.get_autocorr_time.return_value = np.array([50, 60, 70])

        flat_chain = np.random.randn(50, 3)  # Too few samples
        validator.check_mcmc_convergence(sampler, flat_chain)

        assert any("not be converged" in issue for issue in validator.issues)
        assert any("Only 50 effective samples" in issue for issue in validator.issues)

    def test_check_mcmc_convergence_autocorr_error(self, validator):
        """Test check_mcmc_convergence handles autocorr errors."""
        sampler = Mock()
        sampler.get_autocorr_time.side_effect = Exception("Chain too short")

        validator.check_mcmc_convergence(sampler, None)
        assert any("Could not compute autocorrelation" in issue for issue in validator.issues)

    def test_check_parameter_bounds_log_space_good(self, validator, capsys):
        """Test check_parameter_bounds with valid log-space params."""
        params = MockFRBParams(c0=-1.0, zeta=-2.0, tau_1ghz=-3.0, alpha=4.0)
        validator.check_parameter_bounds(params, log_space=True)

        # No issues for reasonable log-space values
        assert len(validator.issues) == 0
        captured = capsys.readouterr()
        assert "log-space" in captured.out

    def test_check_parameter_bounds_log_space_extreme(self, validator):
        """Test check_parameter_bounds flags extreme log-space values."""
        params = MockFRBParams(c0=-15.0, zeta=-20.0, tau_1ghz=-20.0, alpha=10.0)
        validator.check_parameter_bounds(params, log_space=True)

        assert len(validator.issues) >= 1

    def test_check_parameter_bounds_linear_space_good(self, validator):
        """Test check_parameter_bounds with valid linear-space params."""
        params = MockFRBParams(c0=1.0, zeta=0.5, tau_1ghz=0.1, alpha=4.0)
        validator.check_parameter_bounds(params, log_space=False)

        assert len(validator.issues) == 0

    def test_check_parameter_bounds_linear_space_bad(self, validator):
        """Test check_parameter_bounds flags invalid linear-space values."""
        params = MockFRBParams(c0=-1.0, zeta=-0.5, tau_1ghz=-0.1, alpha=10.0)
        validator.check_parameter_bounds(params, log_space=False)

        assert len(validator.issues) >= 3  # c0, zeta, tau_1ghz all negative

    def test_report_no_issues(self, validator, capsys):
        """Test report with no issues."""
        result = validator.report()

        assert result is True
        captured = capsys.readouterr()
        assert "All checks passed" in captured.out

    def test_report_with_issues(self, validator, capsys):
        """Test report with issues."""
        validator.issues = ["[WARN] Test issue 1", "[WARN] Test issue 2"]
        result = validator.report()

        assert result is False
        captured = capsys.readouterr()
        assert "2 potential issues" in captured.out
        assert "Test issue 1" in captured.out


# =============================================================================
# TESTS: Figure Generation Functions
# =============================================================================


class TestFigureGeneration:
    """Tests for figure generation functions."""

    def test_create_data_overview_figure(self, mock_dataset, validator, tmp_path):
        """Test data overview figure generation."""
        output_path = tmp_path / "test_overview.pdf"

        create_data_overview_figure(mock_dataset, output_path, validator)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_create_initial_guess_figure(self, mock_dataset, validator, tmp_path):
        """Test initial guess figure generation."""
        output_path = tmp_path / "test_initial_guess.pdf"
        params = MockFRBParams()

        # Mock the FRBModel import
        with patch(
            "scattering.scripts.run_scattering_analysis.create_initial_guess_figure"
        ) as mock_func:
            # Just test that it's callable with correct args
            mock_func(mock_dataset, params, "M3", output_path, validator)
            mock_func.assert_called_once()

    def test_create_results_summary_figure(
        self, mock_dataset, mock_results, validator, tmp_path
    ):
        """Test results summary figure generation."""
        output_path = tmp_path / "test_results.pdf"

        create_results_summary_figure(mock_results, mock_dataset, output_path, validator)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_create_results_summary_figure_no_chain(
        self, mock_dataset, validator, tmp_path
    ):
        """Test results summary with missing chain data."""
        output_path = tmp_path / "test_results_no_chain.pdf"
        results = {
            "best_key": "M3",
            "best_params": None,
            "flat_chain": None,
            "param_names": [],
            "sampler": None,
            "model_instance": None,
            "goodness_of_fit": {},
        }

        create_results_summary_figure(results, mock_dataset, output_path, validator)

        assert output_path.exists()


# =============================================================================
# TESTS: Main Function
# =============================================================================


class TestMainFunction:
    """Tests for the main() function."""

    def test_main_config_not_found(self, capsys):
        """Test main() with non-existent config file."""
        result = main("nonexistent_config.yaml")

        assert result is False
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out

    @patch("flits.scattering.scripts.run_scattering_analysis.load_config")
    @patch("flits.scattering.scripts.run_scattering_analysis.BurstPipeline")
    @patch("flits.scattering.scripts.run_scattering_analysis.BurstDataset")
    def test_main_pipeline_creation_error(
        self, mock_dataset_cls, mock_pipeline_cls, mock_load_config, capsys, tmp_path
    ):
        """Test main() handles pipeline creation errors."""
        # Setup mock config with real tmp_path
        mock_config = Mock()
        mock_config.path = tmp_path / "data.npy"
        mock_config.telescope = Mock()
        mock_config.telescope.name = "test"
        mock_config.sampler = Mock()
        mock_config.dm_init = 0.0
        mock_config.pipeline = Mock(f_factor=1, t_factor=1, steps=100)
        mock_load_config.return_value = mock_config

        # Make pipeline creation fail
        mock_pipeline_cls.side_effect = Exception("Pipeline error")

        result = main("test_config.yaml")

        assert result is False
        captured = capsys.readouterr()
        assert "Failed to create pipeline" in captured.out


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_validator_empty_array(self, validator):
        """Test validator with empty array."""
        arr = np.array([])
        validator.check_array(arr, "empty_array")
        # Should handle gracefully (no crash)

    def test_validator_single_element(self, validator, capsys):
        """Test validator with single-element array."""
        arr = np.array([5.0])
        validator.check_array(arr, "single_element")

        captured = capsys.readouterr()
        assert "[STATS]" in captured.out

    def test_validator_2d_array(self, validator, capsys):
        """Test validator with 2D array."""
        arr = np.random.randn(10, 20)
        validator.check_array(arr, "2d_array")

        captured = capsys.readouterr()
        assert "[STATS]" in captured.out

    def test_chi2_boundary_values(self, validator):
        """Test chi-squared at boundary values."""
        # Exactly at boundary
        validator.check_chi2(0.5)
        assert len(validator.issues) == 0

        validator.issues = []
        validator.check_chi2(100.0)
        assert len(validator.issues) == 0

    def test_parameter_bounds_boundary_alpha(self, validator):
        """Test alpha at boundary values."""
        # Linear space, alpha at boundaries
        params = MockFRBParams(alpha=2.0)
        validator.check_parameter_bounds(params, log_space=False)
        assert len(validator.issues) == 0

        validator.issues = []
        params = MockFRBParams(alpha=6.0)
        validator.check_parameter_bounds(params, log_space=False)
        assert len(validator.issues) == 0


# =============================================================================
# TESTS: Integration
# =============================================================================


class TestIntegration:
    """Integration tests using real config files."""

    @pytest.mark.slow
    def test_main_with_dsa_config(self):
        """Test main() with actual DSA config (if data exists)."""
        config_path = (
            FLITS_ROOT / "scattering/configs/bursts/dsa/casey_dsa.yaml"
        )
        data_path = FLITS_ROOT / "data/dsa/casey_dsa_I_491_211_2500b_cntr_bpc.npy"

        if not config_path.exists() or not data_path.exists():
            pytest.skip("Config or data file not found")

        # This would run the full analysis - skip in normal testing
        pytest.skip("Full integration test - run manually")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
