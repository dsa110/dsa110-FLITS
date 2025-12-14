"""
Unit tests for the FLITS batch processing module.
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ..config_generator import BurstInfo, ConfigGenerator, TELESCOPE_DEFAULTS
from ..results_db import ResultsDatabase, ScatteringResult, ScintillationResult
from ..joint_analysis import JointAnalysis, ConsistencyResult, FrequencyScalingResult


# =============================================================================
# Config Generator Tests
# =============================================================================

class TestBurstInfo:
    """Tests for BurstInfo parsing."""
    
    def test_parse_chime_filename(self):
        """Parse standard CHIME filename."""
        filepath = Path("casey_chime_I_491_2085_32000b_cntr_bpc.npy")
        info = BurstInfo.from_filename(filepath)
        
        assert info is not None
        assert info.burst_name == "casey"
        assert info.telescope == "chime"
        assert info.dm == 491.2085
        assert info.samples == 32000
    
    def test_parse_dsa_filename(self):
        """Parse standard DSA filename."""
        filepath = Path("hamilton_dsa_I_518_799_2500b_cntr_bpc.npy")
        info = BurstInfo.from_filename(filepath)
        
        assert info is not None
        assert info.burst_name == "hamilton"
        assert info.telescope == "dsa"
        assert info.dm == 518.799
        assert info.samples == 2500
    
    def test_parse_invalid_filename(self):
        """Invalid filename returns None."""
        filepath = Path("invalid_filename.npy")
        info = BurstInfo.from_filename(filepath)
        assert info is None
    
    def test_parse_wrong_extension(self):
        """Wrong extension returns None."""
        filepath = Path("casey_chime_I_491_2085_32000b_cntr_bpc.fits")
        info = BurstInfo.from_filename(filepath)
        assert info is None


class TestConfigGenerator:
    """Tests for ConfigGenerator."""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory structure."""
        # Create telescope directories
        (tmp_path / "chime").mkdir()
        (tmp_path / "dsa").mkdir()
        
        # Create dummy data files
        (tmp_path / "chime" / "casey_chime_I_491_2085_32000b_cntr_bpc.npy").write_bytes(b"")
        (tmp_path / "chime" / "hamilton_chime_I_518_8007_32000b_cntr_bpc.npy").write_bytes(b"")
        (tmp_path / "dsa" / "casey_dsa_I_491_211_2500b_cntr_bpc.npy").write_bytes(b"")
        (tmp_path / "dsa" / "hamilton_dsa_I_518_799_2500b_cntr_bpc.npy").write_bytes(b"")
        
        return tmp_path
    
    def test_discover_bursts(self, temp_data_dir):
        """Test burst discovery."""
        gen = ConfigGenerator(temp_data_dir)
        bursts = gen.discover_bursts()
        
        assert "casey" in bursts
        assert "hamilton" in bursts
        assert len(bursts["casey"]) == 2  # CHIME + DSA
        assert len(bursts["hamilton"]) == 2
    
    def test_discover_single_telescope(self, temp_data_dir):
        """Test discovery with single telescope."""
        gen = ConfigGenerator(temp_data_dir)
        bursts = gen.discover_bursts(telescopes=["chime"])
        
        assert len(bursts["casey"]) == 1
        assert bursts["casey"][0].telescope == "chime"
    
    def test_generate_config(self, temp_data_dir):
        """Test config generation."""
        gen = ConfigGenerator(temp_data_dir)
        bursts = gen.discover_bursts()
        
        info = bursts["casey"][0]  # CHIME observation
        config = gen.generate_config(info, steps=5000, nproc=4)
        
        assert config["burst_name"] == "casey"
        assert config["telescope"] == info.telescope
        assert config["steps"] == 5000
        assert config["nproc"] == 4
        assert "f_factor" in config
        assert "t_factor" in config
    
    def test_generate_all_configs(self, temp_data_dir):
        """Test batch config generation."""
        gen = ConfigGenerator(temp_data_dir, temp_data_dir / "configs")
        configs = gen.generate_all_configs()
        
        assert "casey" in configs
        assert "chime" in configs["casey"]
        assert "dsa" in configs["casey"]
        
        # Check files were created
        assert configs["casey"]["chime"].exists()
        assert configs["casey"]["dsa"].exists()


# =============================================================================
# Results Database Tests
# =============================================================================

class TestResultsDatabase:
    """Tests for ResultsDatabase."""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database."""
        db_path = tmp_path / "test_results.db"
        db = ResultsDatabase(db_path)
        yield db
        db.close()
    
    @pytest.fixture
    def sample_scattering_result(self):
        """Create sample scattering result."""
        return ScatteringResult(
            burst_name="casey",
            telescope="chime",
            tau_1ghz=0.5,
            tau_1ghz_err=0.05,
            alpha=4.2,
            alpha_err=0.3,
            chi2_reduced=1.1,
            best_model="M3",
            quality_flag="good",
        )
    
    @pytest.fixture
    def sample_scintillation_result(self):
        """Create sample scintillation result."""
        return ScintillationResult(
            burst_name="casey",
            telescope="chime",
            delta_nu_dc=0.5,
            delta_nu_dc_err=0.1,
            scaling_alpha=3.8,
            scaling_alpha_err=0.5,
            best_model="lorentzian",
            quality_flag="good",
        )
    
    def test_add_scattering_result(self, temp_db, sample_scattering_result):
        """Test adding scattering result."""
        row_id = temp_db.add_scattering_result(sample_scattering_result)
        assert row_id is not None
        
        # Retrieve and verify
        results = temp_db.get_scattering_results(burst_name="casey")
        assert len(results) == 1
        assert results[0].tau_1ghz == 0.5
    
    def test_add_scintillation_result(self, temp_db, sample_scintillation_result):
        """Test adding scintillation result."""
        row_id = temp_db.add_scintillation_result(sample_scintillation_result)
        assert row_id is not None
        
        results = temp_db.get_scintillation_results(burst_name="casey")
        assert len(results) == 1
        assert results[0].delta_nu_dc == 0.5
    
    def test_filter_by_telescope(self, temp_db, sample_scattering_result):
        """Test filtering by telescope."""
        # Add CHIME result
        temp_db.add_scattering_result(sample_scattering_result)
        
        # Add DSA result
        dsa_result = ScatteringResult(
            burst_name="casey",
            telescope="dsa",
            tau_1ghz=0.3,
        )
        temp_db.add_scattering_result(dsa_result)
        
        # Filter
        chime_results = temp_db.get_scattering_results(telescope="chime")
        assert len(chime_results) == 1
        assert chime_results[0].telescope == "chime"
    
    def test_to_dataframe(self, temp_db, sample_scattering_result):
        """Test DataFrame export."""
        temp_db.add_scattering_result(sample_scattering_result)
        
        df = temp_db.to_dataframe("scattering")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "tau_1ghz" in df.columns
    
    def test_get_comparison_table(self, temp_db, sample_scattering_result, sample_scintillation_result):
        """Test comparison table generation."""
        temp_db.add_scattering_result(sample_scattering_result)
        temp_db.add_scintillation_result(sample_scintillation_result)
        
        df = temp_db.get_comparison_table()
        assert len(df) == 1
        assert "tau_1ghz" in df.columns
        assert "delta_nu_dc" in df.columns


# =============================================================================
# Joint Analysis Tests
# =============================================================================

class TestJointAnalysis:
    """Tests for JointAnalysis."""
    
    @pytest.fixture
    def populated_db(self, tmp_path):
        """Create database with sample data."""
        db_path = tmp_path / "test_joint.db"
        db = ResultsDatabase(db_path)
        
        # Add CHIME scattering
        db.add_scattering_result(ScatteringResult(
            burst_name="casey",
            telescope="chime",
            tau_1ghz=0.5,
            tau_1ghz_err=0.05,
            alpha=4.0,
            alpha_err=0.3,
        ))
        
        # Add DSA scattering
        db.add_scattering_result(ScatteringResult(
            burst_name="casey",
            telescope="dsa",
            tau_1ghz=0.48,
            tau_1ghz_err=0.04,
            alpha=4.1,
            alpha_err=0.2,
        ))
        
        # Add CHIME scintillation
        db.add_scintillation_result(ScintillationResult(
            burst_name="casey",
            telescope="chime",
            delta_nu_dc=0.3,
            delta_nu_dc_err=0.05,
        ))
        
        # Add DSA scintillation
        db.add_scintillation_result(ScintillationResult(
            burst_name="casey",
            telescope="dsa",
            delta_nu_dc=2.0,
            delta_nu_dc_err=0.2,
        ))
        
        yield db
        db.close()
    
    def test_tau_deltanu_consistency(self, populated_db):
        """Test τ-Δν consistency check."""
        joint = JointAnalysis(populated_db)
        results = joint.check_tau_deltanu_consistency()
        
        assert len(results) > 0
        
        # Find CHIME result
        chime_result = next((r for r in results if r.telescope == "chime"), None)
        assert chime_result is not None
        assert chime_result.tau_delta_nu_product is not None
    
    def test_frequency_scaling(self, populated_db):
        """Test frequency scaling analysis."""
        joint = JointAnalysis(populated_db)
        results = joint.analyze_frequency_scaling()
        
        # Should find casey as co-detected
        casey = next((r for r in results if r.burst_name == "casey"), None)
        assert casey is not None
        assert casey.tau_chime_ms is not None
        assert casey.tau_dsa_ms is not None
    
    def test_generate_report(self, populated_db, tmp_path):
        """Test report generation."""
        joint = JointAnalysis(populated_db)
        joint.check_tau_deltanu_consistency()
        joint.analyze_frequency_scaling()
        
        report_path = tmp_path / "report.txt"
        report = joint.generate_report(report_path)
        
        assert "JOINT ANALYSIS REPORT" in report
        assert report_path.exists()


# =============================================================================
# Consistency Result Tests
# =============================================================================

class TestConsistencyResult:
    """Tests for ConsistencyResult dataclass."""
    
    def test_within_expected_range(self):
        """Test product within expected range."""
        result = ConsistencyResult(
            burst_name="test",
            telescope="chime",
            tau_1ghz_ms=0.5,
            delta_nu_mhz=0.5,
            tau_delta_nu_product=0.25,  # Within 0.1-2.0 range
        )
        # Note: is_consistent is computed in the analysis, not here
        assert result.tau_delta_nu_product == 0.25
    
    def test_dataclass_fields(self):
        """Test all expected fields exist."""
        result = ConsistencyResult(burst_name="test", telescope="dsa")
        
        assert hasattr(result, "burst_name")
        assert hasattr(result, "telescope")
        assert hasattr(result, "tau_1ghz_ms")
        assert hasattr(result, "delta_nu_mhz")
        assert hasattr(result, "tau_delta_nu_product")
        assert hasattr(result, "is_consistent")
        assert hasattr(result, "interpretation")


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the batch module."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow from config generation to analysis."""
        # 1. Setup mock data directory
        (tmp_path / "data" / "chime").mkdir(parents=True)
        (tmp_path / "data" / "dsa").mkdir(parents=True)
        
        # Create mock data files
        np.save(tmp_path / "data" / "chime" / "test_chime_I_500_0000_1000b_cntr_bpc.npy", np.zeros((64, 100)))
        np.save(tmp_path / "data" / "dsa" / "test_dsa_I_500_0000_1000b_cntr_bpc.npy", np.zeros((64, 100)))
        
        # 2. Generate configs
        gen = ConfigGenerator(tmp_path / "data", tmp_path / "configs")
        bursts = gen.discover_bursts()
        
        assert "test" in bursts
        assert len(bursts["test"]) == 2
        
        # 3. Create database and add mock results
        db = ResultsDatabase(tmp_path / "results.db")
        
        db.add_scattering_result(ScatteringResult(
            burst_name="test",
            telescope="chime",
            tau_1ghz=1.0,
            tau_1ghz_err=0.1,
        ))
        
        db.add_scintillation_result(ScintillationResult(
            burst_name="test",
            telescope="chime",
            delta_nu_dc=0.2,
            delta_nu_dc_err=0.02,
        ))
        
        # 4. Run joint analysis
        joint = JointAnalysis(db)
        consistency = joint.check_tau_deltanu_consistency()
        
        assert len(consistency) > 0
        
        # 5. Export results
        df = db.get_comparison_table()
        assert len(df) >= 1
        
        db.close()


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_database(self, tmp_path):
        """Test with empty database."""
        db = ResultsDatabase(tmp_path / "empty.db")
        
        assert len(db.get_scattering_results()) == 0
        assert len(db.get_scintillation_results()) == 0
        
        df = db.get_comparison_table()
        assert df.empty
        
        db.close()
    
    def test_missing_counterpart(self, tmp_path):
        """Test joint analysis with missing counterpart."""
        db = ResultsDatabase(tmp_path / "partial.db")
        
        # Only add scattering, no scintillation
        db.add_scattering_result(ScatteringResult(
            burst_name="lonely",
            telescope="chime",
            tau_1ghz=0.5,
        ))
        
        joint = JointAnalysis(db)
        results = joint.check_tau_deltanu_consistency()
        
        # Should still work, but product will be None
        lonely = next((r for r in results if r.burst_name == "lonely"), None)
        assert lonely is not None
        assert lonely.tau_delta_nu_product is None
        
        db.close()
    
    def test_nan_handling(self, tmp_path):
        """Test handling of NaN values."""
        db = ResultsDatabase(tmp_path / "nan.db")
        
        db.add_scattering_result(ScatteringResult(
            burst_name="nan_burst",
            telescope="chime",
            tau_1ghz=float("nan"),
            tau_1ghz_err=float("nan"),
        ))
        
        results = db.get_scattering_results()
        assert len(results) == 1
        # SQLite stores NaN as None, so check for None or NaN
        tau = results[0].tau_1ghz
        assert tau is None or (isinstance(tau, float) and np.isnan(tau))
        
        db.close()

