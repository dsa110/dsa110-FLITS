"""
config_generator.py
===================

Auto-generate configuration files from data file naming conventions.

Naming Convention:
    {burst_name}_{telescope}_I_{dm_int}_{dm_frac}_{samples}b_cntr_bpc.npy
    
Example:
    casey_chime_I_491_2085_32000b_cntr_bpc.npy
    → burst_name: casey
    → telescope: chime
    → dm: 491.2085
    → samples: 32000
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class BurstInfo:
    """Parsed information from a burst data filename."""
    burst_name: str
    telescope: str
    dm: float
    samples: int
    filepath: Path
    
    @classmethod
    def from_filename(cls, filepath: Path) -> Optional["BurstInfo"]:
        """Parse burst info from filename.
        
        Expected format: {name}_{telescope}_I_{dm_int}_{dm_frac}_{samples}b_cntr_bpc.npy
        """
        pattern = r"^(\w+)_(chime|dsa)_I_(\d+)_(\d+)_(\d+)b_cntr_bpc\.npy$"
        match = re.match(pattern, filepath.name, re.IGNORECASE)
        
        if not match:
            log.warning(f"Could not parse filename: {filepath.name}")
            return None
            
        burst_name, telescope, dm_int, dm_frac, samples = match.groups()
        
        # Reconstruct DM: dm_int.dm_frac
        dm = float(f"{dm_int}.{dm_frac}")
        
        return cls(
            burst_name=burst_name.lower(),
            telescope=telescope.lower(),
            dm=dm,
            samples=int(samples),
            filepath=filepath
        )


@dataclass
class TelescopeDefaults:
    """Default configuration values per telescope."""
    name: str
    freq_range: tuple[float, float]  # GHz
    channel_width: float  # MHz
    f_factor: int  # frequency downsampling
    t_factor: int  # time downsampling
    
    
TELESCOPE_DEFAULTS = {
    "chime": TelescopeDefaults(
        name="chime",
        freq_range=(0.4, 0.8),
        channel_width=0.024414,  # MHz
        f_factor=64,
        t_factor=24,
    ),
    "dsa": TelescopeDefaults(
        name="dsa",
        freq_range=(1.28, 1.53),
        channel_width=0.122,  # MHz
        f_factor=384,
        t_factor=2,
    ),
}


class ConfigGenerator:
    """Generate configuration files for batch processing."""
    
    def __init__(
        self,
        data_root: Path,
        output_dir: Optional[Path] = None,
        template_path: Optional[Path] = None,
    ):
        """
        Initialize config generator.
        
        Args:
            data_root: Root directory containing telescope subdirectories (chime/, dsa/)
            output_dir: Directory to write generated configs (default: data_root/configs/generated)
            template_path: Optional YAML template to use as base config
        """
        self.data_root = Path(data_root)
        self.output_dir = output_dir or (self.data_root.parent / "configs" / "generated")
        self.template_path = template_path
        self._base_config: Dict[str, Any] = {}
        
        if template_path and Path(template_path).exists():
            with open(template_path) as f:
                self._base_config = yaml.safe_load(f) or {}
                
    def discover_bursts(self, telescopes: Optional[List[str]] = None) -> Dict[str, List[BurstInfo]]:
        """
        Discover all burst data files organized by burst name.
        
        Args:
            telescopes: List of telescopes to search (default: ["chime", "dsa"])
            
        Returns:
            Dictionary mapping burst_name -> list of BurstInfo (one per telescope)
        """
        if telescopes is None:
            telescopes = ["chime", "dsa"]
            
        bursts_by_name: Dict[str, List[BurstInfo]] = {}
        
        for telescope in telescopes:
            telescope_dir = self.data_root / telescope
            if not telescope_dir.exists():
                log.warning(f"Telescope directory not found: {telescope_dir}")
                continue
                
            for npy_file in telescope_dir.glob("*.npy"):
                info = BurstInfo.from_filename(npy_file)
                if info:
                    if info.burst_name not in bursts_by_name:
                        bursts_by_name[info.burst_name] = []
                    bursts_by_name[info.burst_name].append(info)
                    
        # Sort each burst's observations by telescope for consistency
        for burst_name in bursts_by_name:
            bursts_by_name[burst_name].sort(key=lambda x: x.telescope)
            
        log.info(f"Discovered {len(bursts_by_name)} unique bursts across {telescopes}")
        return bursts_by_name
    
    def generate_config(
        self,
        burst_info: BurstInfo,
        *,
        steps: int = 10000,
        nproc: int = 8,
        model_scan: bool = True,
        diagnostics: bool = True,
        plot: bool = True,
        dm_init: Optional[float] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a configuration dictionary for a single burst.
        
        Args:
            burst_info: Parsed burst information
            steps: Number of MCMC steps
            nproc: Number of parallel processes
            model_scan: Whether to run model comparison
            diagnostics: Whether to run diagnostics
            plot: Whether to generate plots
            dm_init: Initial DM guess (default: use parsed DM)
            overrides: Additional config overrides
            
        Returns:
            Configuration dictionary ready for YAML serialization
        """
        defaults = TELESCOPE_DEFAULTS.get(burst_info.telescope)
        if defaults is None:
            raise ValueError(f"Unknown telescope: {burst_info.telescope}")
            
        # Build relative path from expected config location
        # Configs go in configs/generated/{telescope}/
        # Data is in data/{telescope}/
        rel_data_path = f"../../../data/{burst_info.telescope}/{burst_info.filepath.name}"
        
        config = {
            "burst_name": burst_info.burst_name,
            "path": rel_data_path,
            "telcfg_path": "../../telescopes.yaml",
            "sampcfg_path": "../../sampler.yaml",
            "dm_init": dm_init if dm_init is not None else 0.0,  # Often want to refine DM
            "telescope": burst_info.telescope,
            "steps": steps,
            "f_factor": defaults.f_factor,
            "t_factor": defaults.t_factor,
            "nproc": nproc,
            "extend_chain": True,
            "chunk_size": 2000,
            "max_chunks": 5,
            "model_scan": model_scan,
            "diagnostics": diagnostics,
            "plot": plot,
        }
        
        # Merge with base template if provided
        merged = {**self._base_config, **config}
        
        # Apply any user overrides
        if overrides:
            merged.update(overrides)
            
        return merged
    
    def generate_all_configs(
        self,
        telescopes: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Path]]:
        """
        Generate configs for all discovered bursts.
        
        Args:
            telescopes: Telescopes to include
            **kwargs: Passed to generate_config()
            
        Returns:
            Nested dict: {burst_name: {telescope: config_path}}
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        bursts = self.discover_bursts(telescopes)
        generated_paths: Dict[str, Dict[str, Path]] = {}
        
        for burst_name, burst_list in bursts.items():
            generated_paths[burst_name] = {}
            
            for burst_info in burst_list:
                # Create telescope subdirectory
                telescope_dir = self.output_dir / burst_info.telescope
                telescope_dir.mkdir(exist_ok=True)
                
                config = self.generate_config(burst_info, **kwargs)
                
                config_path = telescope_dir / f"{burst_name}_{burst_info.telescope}.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    
                generated_paths[burst_name][burst_info.telescope] = config_path
                log.info(f"Generated config: {config_path}")
                
        return generated_paths
    
    def generate_batch_manifest(
        self,
        telescopes: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate a manifest file listing all configs for batch processing.
        
        Args:
            telescopes: Telescopes to include
            output_path: Where to write manifest (default: output_dir/manifest.yaml)
            
        Returns:
            Path to generated manifest file
        """
        configs = self.generate_all_configs(telescopes)
        
        manifest = {
            "version": "1.0",
            "description": "FLITS batch processing manifest",
            "data_root": str(self.data_root),
            "bursts": {},
        }
        
        for burst_name, telescope_configs in configs.items():
            manifest["bursts"][burst_name] = {
                telescope: str(path) 
                for telescope, path in telescope_configs.items()
            }
            
        output_path = output_path or (self.output_dir / "manifest.yaml")
        with open(output_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            
        log.info(f"Generated manifest: {output_path}")
        return output_path


def main():
    """CLI entry point for config generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate FLITS batch configs")
    parser.add_argument("data_root", type=Path, help="Root data directory")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--steps", type=int, default=10000, help="MCMC steps")
    parser.add_argument("--nproc", type=int, default=8, help="Parallel processes")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    generator = ConfigGenerator(args.data_root, args.output)
    manifest = generator.generate_batch_manifest(steps=args.steps, nproc=args.nproc)
    
    print(f"\n✅ Generated configs and manifest: {manifest}")


if __name__ == "__main__":
    main()

