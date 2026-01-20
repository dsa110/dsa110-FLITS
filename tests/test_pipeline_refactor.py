
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch

# Import new modules
from scattering.scat_analysis.pipeline.core import BurstPipeline
from scattering.scat_analysis.pipeline.io import BurstDataset
from scattering.scat_analysis.pipeline.optimization import refine_initial_guess_mle
from scattering.scat_analysis.pipeline.diagnostics import BurstDiagnostics
from scattering.scat_analysis.burstfit import FRBParams
from scattering.scat_analysis.config_utils import TelescopeConfig

def test_pipeline_imports():
    """Test that all pipeline modules import correctly."""
    assert True

def test_burstdataset_instantiation():
    """Test that BurstDataset can be instantiated."""
    # Create dummy data
    time = np.linspace(0, 1, 100)
    freq = np.linspace(1, 1.5, 32)
    data = np.zeros((32, 100))
    
    # Needs telescope config
    tel = TelescopeConfig(name="test", n_ch_raw=32, df_MHz_raw=1.0, dt_ms_raw=0.1, f_min_GHz=1.0, f_max_GHz=1.5)
    
    with tempfile.NamedTemporaryFile(suffix=".npy") as f:
        # Create a dummy file so exists() check passes if lazy=False, 
        # but we use lazy=True here to skip loading.
        ds = BurstDataset(f.name, "out_dir", telescope=tel, lazy=True)
        assert ds.inpath.name == os.path.basename(f.name)
        assert ds.telescope.name == "test"

def test_burstpipeline_instantiation():
    """Test that BurstPipeline can be instantiated."""
    with patch("scattering.scat_analysis.pipeline.core.build_pool") as mock_pool:
        mock_pool.return_value = None
        pipeline = BurstPipeline("in.npy", "out_dir", "FRB1234")
        assert pipeline.inpath == "in.npy"
        assert pipeline.name == "FRB1234"

def test_optimization_function_exists():
    """Test that optimization functions are available."""
    assert callable(refine_initial_guess_mle)

def test_diagnostics_instantiation():
    """Test that BurstDiagnostics can be instantiated."""
    # Create dummy dataset mock
    dataset = MagicMock()
    dataset.data = np.zeros((32, 100))
    dataset.freq = np.linspace(1, 1.5, 32)
    dataset.time = np.linspace(0, 1, 100)
    
    # Create dummy results dict
    results = {"best_params": None, "best_key": "M3"}
    
    diag = BurstDiagnostics(dataset, results)
    assert diag.dataset == dataset
    assert diag.results_in == results
