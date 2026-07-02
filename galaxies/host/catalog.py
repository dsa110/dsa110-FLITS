"""Host-galaxy metadata for DM predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from galaxies.foreground import config

# Must match galaxies.foreground.sightline_budget.PLACEHOLDER_Z (unknown host z).
PLACEHOLDER_Z = 1.0

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_HOSTS_PATH = PACKAGE_DIR / "data" / "hosts.yaml"


@dataclass(frozen=True)
class HostRecord:
    nickname: str
    z: float
    z_is_placeholder: bool = False
    log10_mstar: float | None = None
    halpha_flux_erg_s: float | None = None
    reff_arcsec: float | None = None
    ssfr_msun_yr_kpc2: float | None = None
    offset_kpc: float = 0.0
    av: float | None = None
    ism_path_kpc: float = 1.0
    ism_path_pc: float = 100.0
    source: str | None = None


def _is_placeholder_z(z: float) -> bool:
    return math.isfinite(float(z)) and abs(float(z) - PLACEHOLDER_Z) < 1.0e-6


def _target_z(name: str) -> tuple[float, bool]:
    key = name.strip().lower()
    for nick, _ra, _dec, z in config.TARGETS:
        if nick.lower() == key:
            z_val = float(z)
            return z_val, _is_placeholder_z(z_val)
    raise KeyError(f"Unknown sightline nickname: {name!r}")


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_host_catalog(path: Path | str | None = None) -> dict[str, HostRecord]:
    """Load optional per-burst host metadata merged with ``config.TARGETS`` redshifts."""
    catalog_path = Path(path) if path is not None else DEFAULT_HOSTS_PATH
    overrides: dict[str, dict[str, Any]] = {}
    if catalog_path.is_file():
        raw = yaml.safe_load(catalog_path.read_text()) or {}
        for nick, payload in (raw.get("hosts") or {}).items():
            if isinstance(payload, dict):
                overrides[str(nick).lower()] = payload

    out: dict[str, HostRecord] = {}
    for nick, _ra, _dec, z in config.TARGETS:
        key = nick.lower()
        z_val = float(z)
        payload = overrides.get(key, {})
        z_override = _coerce_optional_float(payload.get("z"))
        if z_override is not None:
            z_val = z_override
        out[key] = HostRecord(
            nickname=nick,
            z=z_val,
            z_is_placeholder=_is_placeholder_z(z_val),
            log10_mstar=_coerce_optional_float(payload.get("log10_mstar")),
            halpha_flux_erg_s=_coerce_optional_float(payload.get("halpha_flux_erg_s")),
            reff_arcsec=_coerce_optional_float(payload.get("reff_arcsec")),
            ssfr_msun_yr_kpc2=_coerce_optional_float(payload.get("ssfr_msun_yr_kpc2")),
            offset_kpc=_coerce_optional_float(payload.get("offset_kpc")) or 0.0,
            av=_coerce_optional_float(payload.get("av")),
            ism_path_kpc=_coerce_optional_float(payload.get("ism_path_kpc")) or 1.0,
            ism_path_pc=_coerce_optional_float(payload.get("ism_path_pc")) or 100.0,
            source=str(payload.get("source")) if payload.get("source") is not None else None,
        )
    return out


def host_record_for_target(name: str, catalog: dict[str, HostRecord] | None = None) -> HostRecord:
    """Return the host record for a sightline nickname."""
    key = name.strip().lower()
    if catalog is None:
        catalog = load_host_catalog()
    if key in catalog:
        return catalog[key]
    z_val, z_ph = _target_z(name)
    return HostRecord(
        nickname=name,
        z=z_val,
        z_is_placeholder=z_ph,
    )
