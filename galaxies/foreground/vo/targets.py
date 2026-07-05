"""FRB target definitions and cosmology selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from astropy.cosmology import Planck18
from astropy.cosmology.core import Cosmology


@dataclass
class Target:
    name: str
    ra: float  # deg
    dec: float  # deg
    z_host: float | None = None  # host spectroscopic redshift, if known


def load_targets(yaml_path: str | Path) -> list[Target]:
    with Path(yaml_path).open() as f:
        data = yaml.safe_load(f) or {}
    return [Target(**t) for t in data.get("targets", [])]


def get_cosmology(name: str | None = None) -> Cosmology:
    if name is None or name == "Planck18":
        return Planck18
    raise ValueError(f"Unsupported cosmology: {name}")
