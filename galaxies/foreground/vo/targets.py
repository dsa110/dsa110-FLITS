"""Target loading (legacy shim) and cosmology selection."""

from __future__ import annotations

from astropy.cosmology import Cosmology, Planck18

from .domain import Sightline
from .io import read_targets_yaml as load_targets

Target = Sightline  # legacy alias; the old z_host attribute is now .redshift


def get_cosmology(name: str | None = None) -> Cosmology:
    if name is None or name == "Planck18":
        return Planck18
    raise ValueError(f"Unsupported cosmology: {name}")


__all__ = ["Sightline", "Target", "get_cosmology", "load_targets"]
