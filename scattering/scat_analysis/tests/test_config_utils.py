"""
test_config_utils.py
====================

Unit tests for config_utils.py - Configuration loading and path resolution.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

import sys
from pathlib import Path

# Add the parent directories to path for imports
_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scattering dir

from scat_analysis.config_utils import (
    resolve_path,
    TelescopeConfig,
    SamplerConfig,
    PipelineOptions,
    Config,
    load_telescope_block,
    load_sampler_block,
    load_config,
    clear_config_cache,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory with config files."""
    # Create telescopes.yaml
    telescopes = {
        "telescopes": {
            "dsa": {
                "df_MHz_raw": 0.122,
                "dt_ms_raw": 0.262,
                "f_min_GHz": 1.28,
                "f_max_GHz": 1.53,
                "n_ch_raw": 2048,
            },
            "chime": {
                "df_MHz_raw": 0.024,
                "dt_ms_raw": 0.983,
                "f_min_GHz": 0.4,
                "f_max_GHz": 0.8,
                "n_ch_raw": 16384,
            },
        },
        "default_telescope": "dsa",
    }
    
    with open(tmp_path / "telescopes.yaml", "w") as f:
        yaml.dump(telescopes, f)
    
    # Create sampler.yaml
    samplers = {
        "samplers": {
            "emcee": {
                "n_walkers": 32,
                "n_steps": 1000,
            },
            "nested": {
                "nlive": 500,
            },
        },
        "default_sampler": "emcee",
    }
    
    with open(tmp_path / "sampler.yaml", "w") as f:
        yaml.dump(samplers, f)
    
    # Create a run config
    run_config = {
        "path": "data/test_burst.npy",
        "telescope": "dsa",
        "dm_init": 100.0,
        "steps": 5000,
        "f_factor": 2,
        "t_factor": 1,
    }
    
    with open(tmp_path / "test_run.yaml", "w") as f:
        yaml.dump(run_config, f)
    
    # Create data directory
    (tmp_path / "data").mkdir()
    
    return tmp_path


# ============================================================================
# resolve_path Tests
# ============================================================================

class TestResolvePath:
    """Tests for resolve_path function."""

    def test_absolute_path(self):
        """Test that absolute paths are returned unchanged."""
        abs_path = "/usr/local/data/burst.npy"
        result = resolve_path(abs_path)
        
        assert result == Path(abs_path)

    def test_relative_path_with_base(self, tmp_path):
        """Test relative path resolution with base directory."""
        result = resolve_path("subdir/file.txt", base_dir=tmp_path)
        
        assert result == (tmp_path / "subdir" / "file.txt").resolve()

    def test_relative_path_without_base(self):
        """Test relative path resolution without base (uses CWD)."""
        result = resolve_path("data/file.txt")
        
        expected = (Path.cwd() / "data" / "file.txt").resolve()
        assert result == expected

    def test_home_expansion(self):
        """Test ~ expansion."""
        result = resolve_path("~/data/file.txt")
        
        assert str(result).startswith(str(Path.home()))
        assert "~" not in str(result)

    def test_env_var_expansion(self, monkeypatch):
        """Test environment variable expansion."""
        monkeypatch.setenv("TEST_DATA_DIR", "/tmp/test_data")
        
        result = resolve_path("$TEST_DATA_DIR/file.txt")
        
        assert str(result) == "/tmp/test_data/file.txt"

    def test_env_var_braces(self, monkeypatch):
        """Test ${VAR} style expansion."""
        monkeypatch.setenv("MY_PATH", "/opt/data")
        
        result = resolve_path("${MY_PATH}/burst.npy")
        
        assert str(result) == "/opt/data/burst.npy"

    def test_dotdot_resolution(self, tmp_path):
        """Test .. path component resolution."""
        subdir = tmp_path / "configs" / "bursts"
        subdir.mkdir(parents=True)
        
        result = resolve_path("../telescopes.yaml", base_dir=subdir)
        
        expected = (tmp_path / "configs" / "telescopes.yaml").resolve()
        assert result == expected


# ============================================================================
# TelescopeConfig Tests
# ============================================================================

class TestTelescopeConfig:
    """Tests for TelescopeConfig loading."""

    def test_load_telescope(self, temp_config_dir):
        """Test loading telescope configuration."""
        clear_config_cache()
        
        config = load_telescope_block(
            temp_config_dir / "telescopes.yaml", 
            telescope="dsa"
        )
        
        assert isinstance(config, TelescopeConfig)
        assert config.name == "dsa"
        assert config.df_MHz_raw == 0.122
        assert config.f_min_GHz == 1.28
        assert config.f_max_GHz == 1.53

    def test_load_default_telescope(self, temp_config_dir):
        """Test loading default telescope when none specified."""
        clear_config_cache()
        
        config = load_telescope_block(temp_config_dir / "telescopes.yaml")
        
        assert config.name == "dsa"  # default

    def test_missing_telescope(self, temp_config_dir):
        """Test error for non-existent telescope."""
        clear_config_cache()
        
        with pytest.raises(KeyError, match="not present"):
            load_telescope_block(
                temp_config_dir / "telescopes.yaml",
                telescope="vla"
            )


# ============================================================================
# SamplerConfig Tests
# ============================================================================

class TestSamplerConfig:
    """Tests for SamplerConfig loading."""

    def test_load_sampler(self, temp_config_dir):
        """Test loading sampler configuration."""
        clear_config_cache()
        
        config = load_sampler_block(temp_config_dir / "sampler.yaml", name="emcee")
        
        assert isinstance(config, SamplerConfig)
        assert config.name == "emcee"
        assert config.params["n_walkers"] == 32

    def test_sampler_getattr(self, temp_config_dir):
        """Test attribute-style access to sampler params."""
        clear_config_cache()
        
        config = load_sampler_block(temp_config_dir / "sampler.yaml", name="emcee")
        
        assert config.n_walkers == 32
        assert config.n_steps == 1000


# ============================================================================
# Full Config Loading Tests
# ============================================================================

class TestConfigLoading:
    """Tests for full configuration loading."""

    def test_load_full_config(self, temp_config_dir):
        """Test loading complete run configuration."""
        clear_config_cache()
        
        # Need to create a minimal data file
        (temp_config_dir / "data" / "test_burst.npy").touch()
        
        config = load_config(
            temp_config_dir / "test_run.yaml",
            workspace_root=temp_config_dir
        )
        
        assert isinstance(config, Config)
        assert config.dm_init == 100.0
        assert config.telescope.name == "dsa"
        assert config.pipeline.steps == 5000

    def test_config_path_resolution(self, temp_config_dir):
        """Test that data path is resolved correctly."""
        clear_config_cache()
        
        (temp_config_dir / "data" / "test_burst.npy").touch()
        
        config = load_config(
            temp_config_dir / "test_run.yaml",
            workspace_root=temp_config_dir
        )
        
        # Path should be absolute and resolved
        assert config.path.is_absolute()

    def test_missing_path_error(self, temp_config_dir):
        """Test error when 'path' is missing from config."""
        clear_config_cache()
        
        # Create config without path
        bad_config = {"telescope": "dsa", "dm_init": 100.0}
        with open(temp_config_dir / "bad_config.yaml", "w") as f:
            yaml.dump(bad_config, f)
        
        with pytest.raises(ValueError, match="path"):
            load_config(temp_config_dir / "bad_config.yaml")

    def test_missing_telescope_error(self, temp_config_dir):
        """Test error when 'telescope' is missing from config."""
        clear_config_cache()
        
        bad_config = {"path": "data/test.npy", "dm_init": 100.0}
        with open(temp_config_dir / "bad_config.yaml", "w") as f:
            yaml.dump(bad_config, f)
        
        with pytest.raises(ValueError, match="telescope"):
            load_config(temp_config_dir / "bad_config.yaml")


# ============================================================================
# PipelineOptions Tests
# ============================================================================

class TestPipelineOptions:
    """Tests for PipelineOptions defaults and parsing."""

    def test_default_values(self):
        """Test PipelineOptions default values."""
        opts = PipelineOptions()
        
        assert opts.steps == 2000
        assert opts.f_factor == 1
        assert opts.t_factor == 1
        assert opts.extend_chain is False
        assert opts.model_scan is True
        assert opts.diagnostics is True
        assert opts.plot is True

    def test_custom_values(self):
        """Test PipelineOptions with custom values."""
        opts = PipelineOptions(
            steps=5000,
            f_factor=4,
            t_factor=2,
            nproc=8,
            extend_chain=True,
        )
        
        assert opts.steps == 5000
        assert opts.f_factor == 4
        assert opts.t_factor == 2
        assert opts.nproc == 8
        assert opts.extend_chain is True


# ============================================================================
# Cache Tests
# ============================================================================

class TestConfigCache:
    """Tests for configuration caching."""

    def test_cache_clear(self, temp_config_dir):
        """Test that cache can be cleared."""
        # Load something to populate cache
        load_telescope_block(temp_config_dir / "telescopes.yaml")
        
        # Clear should not raise
        clear_config_cache()

    def test_reload_after_clear(self, temp_config_dir):
        """Test that configs can be reloaded after cache clear."""
        config1 = load_telescope_block(temp_config_dir / "telescopes.yaml")
        clear_config_cache()
        config2 = load_telescope_block(temp_config_dir / "telescopes.yaml")
        
        assert config1.name == config2.name

