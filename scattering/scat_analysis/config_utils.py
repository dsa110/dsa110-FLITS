"""
config_utils.py
===============

Utility for reading telescope-specific raw-data parameters from
*telescopes.yaml*.  Keeping this separate avoids a hard dependency on
`pyyaml` in the core physics modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os

import functools
import yaml


def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Resolve a path, supporting relative paths and environment variables.
    
    Parameters
    ----------
    path : str or Path
        The path to resolve. Can be:
        - Absolute path: returned as-is
        - Relative path: resolved relative to base_dir or CWD
        - Contains $VAR or ${VAR}: environment variables are expanded
        - Starts with ~: user home directory is expanded
    base_dir : Path, optional
        Base directory for resolving relative paths. If None, uses current
        working directory.
        
    Returns
    -------
    Path
        The resolved, absolute path.
        
    Examples
    --------
    >>> resolve_path("data/burst.npy", base_dir=Path("/project/configs"))
    PosixPath('/project/data/burst.npy')
    >>> resolve_path("$HOME/data/burst.npy")
    PosixPath('/home/user/data/burst.npy')
    """
    # Convert to string for expansion
    path_str = str(path)
    
    # Expand environment variables (handles both $VAR and ${VAR})
    path_str = os.path.expandvars(path_str)
    
    # Expand user home directory (~)
    path_str = os.path.expanduser(path_str)
    
    # Convert to Path
    resolved = Path(path_str)
    
    # If already absolute, return as-is
    if resolved.is_absolute():
        return resolved
    
    # Resolve relative to base_dir or CWD
    if base_dir is not None:
        resolved = base_dir / resolved
    else:
        resolved = Path.cwd() / resolved
    
    # Normalize the path (resolve .. and . components)
    return resolved.resolve()

__all__ = [
    "TelescopeConfig",
    "SamplerConfig",
    "PipelineOptions",
    "Config",
    "load_telescope_block",
    "load_sampler_block",
    "load_sampler_choice",
    "load_config",
    "clear_config_cache",
    "resolve_path",
]


@dataclass
class TelescopeConfig:
    """Container for raw telescope parameters."""
    name: str
    df_MHz_raw: float
    dt_ms_raw: float
    f_min_GHz: float
    f_max_GHz: float
    n_ch_raw: Optional[int] = None
    freq_descending: bool = False  # True if data[0] = f_max (high freq at row 0)


@dataclass
class SamplerConfig:
    """Container for sampler settings.  Arbitrary keys are stored in ``params``."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        try:
            return self.params[item]
        except KeyError as exc:  # pragma: no cover - error path
            raise AttributeError(f"Sampler setting '{item}' not found") from exc


@dataclass
class PipelineOptions:
    """General options controlling the BurstFit pipeline."""
    steps: int = 2000
    f_factor: int = 1
    t_factor: int = 1
    nproc: Optional[int] = None
    extend_chain: bool = False
    chunk_size: int = 0
    max_chunks: int = 0
    model_scan: bool = True
    diagnostics: bool = True
    plot: bool = True


@dataclass
class Config:
    """Top-level configuration object returned by :func:`load_config`."""
    path: Path
    dm_init: float
    telescope: TelescopeConfig
    sampler: SamplerConfig
    pipeline: PipelineOptions


@functools.lru_cache(maxsize=None)
def _read_yaml(path: str | Path) -> dict:
    """Low-level cache shared by all YAML helpers."""
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_telescope_block(
    telcfg_path: str | Path = "telescopes.yaml",
    telescope: str | None = None,
) -> TelescopeConfig:
    """Return a :class:`TelescopeConfig` for the requested telescope."""

    cfg = _read_yaml(telcfg_path)

    blocks = cfg.get("telescopes", cfg)  # legacy files may be flat
    if not isinstance(blocks, dict) or not blocks:
        raise KeyError(f"No 'telescopes' block found in {telcfg_path}")

    if telescope is None:
        telescope = cfg.get("default_telescope") or next(iter(blocks))

    if telescope not in blocks:
        raise KeyError(
            f"Telescope '{telescope}' not present in '{telcfg_path}'. "
            f"Available: {list(blocks)}"
        )

    entry = blocks[telescope]
    required = ("df_MHz_raw", "dt_ms_raw", "f_min_GHz", "f_max_GHz")
    missing = [k for k in required if k not in entry or entry[k] is None]
    if missing:
        raise ValueError(
            f"Telescope '{telescope}' in '{telcfg_path}' is missing fields {missing}"
        )

    params = {k: float(entry[k]) for k in required}
    if "n_ch_raw" in entry and entry["n_ch_raw"] is not None:
        params["n_ch_raw"] = int(entry["n_ch_raw"])

    return TelescopeConfig(name=telescope, **params)


def load_sampler_block(
    path: str | Path = "sampler.yaml", name: str | None = None
) -> SamplerConfig:
    """Return a :class:`SamplerConfig` representing the chosen sampler."""

    cfg = _read_yaml(path)

    samplers = cfg.get("samplers", {})
    if not samplers:
        raise KeyError("No 'samplers' section found in sampler YAML")

    target = (name or cfg.get("default_sampler") or "emcee").lower()
    if target not in samplers:
        raise KeyError(
            f"Sampler '{target}' not found in YAML. Available: {list(samplers)}"
        )

    return SamplerConfig(name=target, params=samplers[target])


def load_sampler_choice(path: str | Path = "sampler.yaml") -> str:
    """Return only the default sampler name (helper for CLI autocompletion)."""
    return load_sampler_block(path).name


def load_config(path: str | Path, workspace_root: Optional[Path] = None) -> Config:
    """Load the full analysis configuration from ``path``.

    The file specified by *path* is expected to contain run-specific options
    such as the data ``path``, ``dm_init`` and ``telescope`` choice.  The
    corresponding ``telescopes.yaml`` and ``sampler.yaml`` files are assumed to
    live in the same directory unless explicit ``telcfg_path`` or
    ``sampcfg_path`` entries are provided.
    
    Parameters
    ----------
    path : str or Path
        Path to the run configuration YAML file.
    workspace_root : Path, optional
        Root directory of the workspace/project. Used for resolving relative
        data paths. If None, defaults to the parent of the config file's directory.
        
    Notes
    -----
    Relative paths in the config are resolved in the following order:
    1. Relative to config file directory (for telcfg_path, sampcfg_path)
    2. Relative to workspace_root (for data paths)
    
    Environment variables ($VAR) and home directory (~) are expanded in all paths.
    """

    run_path = Path(path).expanduser().resolve()
    with run_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    config_dir = run_path.parent
    
    # Workspace root defaults to grandparent of config (e.g., configs/bursts/x.yaml -> workspace)
    if workspace_root is None:
        # Try to find workspace root by looking for common markers
        workspace_root = config_dir
        for _ in range(5):  # Look up to 5 levels
            if (workspace_root / "pyproject.toml").exists() or \
               (workspace_root / "setup.py").exists() or \
               (workspace_root / ".git").exists():
                break
            if workspace_root.parent == workspace_root:
                break
            workspace_root = workspace_root.parent
    
    # Resolve telescope and sampler config paths relative to config directory
    telcfg_raw = cfg.get("telcfg_path", "telescopes.yaml")
    sampcfg_raw = cfg.get("sampcfg_path", "sampler.yaml")
    telcfg_path = resolve_path(telcfg_raw, base_dir=config_dir)
    sampcfg_path = resolve_path(sampcfg_raw, base_dir=config_dir)

    if "telescope" not in cfg:
        raise ValueError("Run config is missing required field 'telescope'")

    telescope = load_telescope_block(telcfg_path, cfg["telescope"])
    sampler = load_sampler_block(sampcfg_path, cfg.get("sampler"))

    data_path_raw = cfg.get("path")
    if data_path_raw is None:
        raise ValueError("Run config must specify 'path' to the data file")
    
    # Resolve data path relative to workspace root
    data_path = resolve_path(data_path_raw, base_dir=workspace_root)

    dm_init = float(cfg.get("dm_init", 0.0))

    pipe = PipelineOptions(
        steps=int(cfg.get("steps", 2000)),
        f_factor=int(cfg.get("f_factor", 1)),
        t_factor=int(cfg.get("t_factor", 1)),
        nproc=cfg.get("nproc"),
        extend_chain=bool(cfg.get("extend_chain", False)),
        chunk_size=int(cfg.get("chunk_size", 0)),
        max_chunks=int(cfg.get("max_chunks", 0)),
        model_scan=bool(cfg.get("model_scan", True)),
        diagnostics=bool(cfg.get("diagnostics", True)),
        plot=bool(cfg.get("plot", True)),
    )

    return Config(
        path=data_path,
        dm_init=dm_init,
        telescope=telescope,
        sampler=sampler,
        pipeline=pipe,
    )


def clear_config_cache() -> None:
    """Clear the in-memory YAML cache for telescope & sampler loaders."""
    _read_yaml.cache_clear()

