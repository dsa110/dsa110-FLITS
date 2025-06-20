"""
config_utils.py
===============

Utility for reading telescope-specific raw-data parameters from
*telescopes.yaml*.  Keeping this separate avoids a hard dependency on
`pyyaml` in the core physics modules.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import yaml
import functools

__all__ = [
    "load_telescope_block",
    "load_sampler_block",
    "load_sampler_choice",
    "clear_config_cache",
]

@functools.lru_cache(maxsize=None)
def _read_yaml(path: str | Path) -> dict:
    """Low-level cache shared by all YAML helpers."""
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def load_telescope_block(
    telcfg_path: str | Path = "telescopes.yaml",
    telescope: str | None = None,
) -> tuple[str, dict]:
    """
    Return ``(name, params)`` for the requested telescope.

    Parameters
    ----------
    telcfg_path
        Path to the YAML file that contains a ``telescopes:`` section.
    telescope
        Which telescope to load.  If *None*, use ``default_telescope`` from
        the YAML file, falling back to the *first* telescope encountered.

    The function validates and converts the four required fields::

        df_MHz_raw, dt_ms_raw, f_min_GHz, f_max_GHz
    """
    cfg = _read_yaml(telcfg_path)

    blocks = cfg.get("telescopes", cfg)   # legacy files may be flat
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
    missing  = [k for k in required if k not in entry or entry[k] is None]
    if missing:
        raise ValueError(
            f"Telescope '{telescope}' in '{telcfg_path}' "
            f"is missing fields {missing}"
        )

    params = {k: float(entry[k]) for k in required}
    return telescope, params

def load_sampler_block(path: str | Path = "sampler.yaml",
                       name: str | None = None) -> tuple[str, dict]:
    """
    Return ``(sampler_name, params_dict)``.

    * If *name* is given, use that sampler.
    * Otherwise use ``default_sampler`` from YAML (falls back to 'emcee').
    """
    cfg = _read_yaml(path)

    samplers = cfg.get("samplers", {})
    if not samplers:
        raise KeyError("No 'samplers' section found in sampler YAML")

    # decide which block we want
    target = (name or cfg.get("default_sampler") or "emcee").lower()
    if target not in samplers:
        raise KeyError(f"Sampler '{target}' not found in YAML. "
                       f"Available: {list(samplers)}")

    return target, samplers[target]

def load_sampler_choice(path: str | Path = "sampler.yaml") -> str:
    """Return only the default sampler name (helper for CLI autocompletion)."""
    return load_sampler_block(path)[0]

def clear_config_cache() -> None:
    """Clear the in-memory YAML cache for telescope & sampler loaders."""
    _read_yaml.cache_clear()

