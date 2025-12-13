"""
Bridge module that exposes the legacy `scattering` code under the
installable `flits.scattering` namespace.

This lets downstream code use standard imports such as
`from flits.scattering.scat_analysis import burstfit` without relying on
ad-hoc `sys.path` manipulation.

Also exports broadening and scattering utilities from `.broaden`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .broaden import scatter_broaden, tau_per_freq, log_normal_prior, gaussian_prior

_ROOT = Path(__file__).resolve().parents[2] / "scattering"

if not _ROOT.exists():
    raise ImportError(
        "Could not locate the 'scattering' module directory. "
        "Ensure the FLITS repository is checked out with the 'scattering' "
        "subdirectory intact."
    )

# Tell Python where to find submodules such as scat_analysis.*
__path__: List[str] = [str(_ROOT)]

# Re-export the absolute path for tooling/tests that need it.
SCATTERING_ROOT = _ROOT

__all__ = [
    "scatter_broaden",
    "tau_per_freq",
    "log_normal_prior",
    "gaussian_prior",
]
