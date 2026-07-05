"""Domain models for FRB foreground candidate searches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord


@dataclass(frozen=True)
class Sightline:
    """A background FRB or other source used to search for foreground systems."""

    ra: float
    dec: float
    name: str | None = None
    source_id: str | None = None
    redshift: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skycoord(self) -> SkyCoord:
        """Return the sightline as an ICRS coordinate."""
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs")


@dataclass
class ForegroundCandidate:
    """A foreground galaxy, group, cluster, or halo candidate near a sightline."""

    ra: float
    dec: float
    name: str | None = None
    source_id: str | None = None
    z: float | None = None
    z_type: str = "unknown"
    service: str | None = None
    table: str | None = None
    richness: float | None = None
    m_delta: float | None = None
    r_delta: float | None = None
    delta_def: float | None = None
    theta_arcmin: float | None = None
    impact_parameter_kpc: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def skycoord(self) -> SkyCoord:
        """Return the candidate as an ICRS coordinate."""
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs")

    def has_redshift(self) -> bool:
        """Return whether the candidate has a usable redshift value."""
        return self.z is not None


@dataclass
class QueryResult:
    """Results from one service/table query around one sightline."""

    sightline: Sightline
    service: str
    table: str
    candidates: list[ForegroundCandidate]
    adql: str | None = None
    success: bool = True
    error_message: str | None = None

    def __len__(self) -> int:
        return len(self.candidates)
