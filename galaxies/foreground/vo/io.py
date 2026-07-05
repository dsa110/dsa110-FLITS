"""Input/output helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .domain import Sightline

PLACEHOLDER_Z_THRESHOLD = 0.999


def normalize_host_redshift(
    value: Any,
    *,
    placeholder_z_threshold: float = PLACEHOLDER_Z_THRESHOLD,
) -> float | None:
    """Normalize YAML/CSV host redshift values for foreground science.

    Returns ``None`` when the host spectroscopic redshift is unknown or a
    documented placeholder (``z >= placeholder_z_threshold``, or explicit
    ``host_z_placeholder: true`` in target metadata). Placeholder values such
    as ``1.0`` must not be treated as real hosts — they falsely classify
    every ``z < 1`` galaxy as foreground.
    """
    if value is None:
        return None
    try:
        z = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(z) or z <= 0:
        return None
    if z >= placeholder_z_threshold:
        return None
    return z


def read_targets_yaml(path: str | Path) -> list[Sightline]:
    """Read a `targets: [...]` YAML file into sightlines.

    Accepts ``redshift`` or the legacy ``z_host`` key (los_halos-era target
    files) for the host redshift.
    """
    payload: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    targets = payload.get("targets", [])
    sightlines: list[Sightline] = []
    for row in targets:
        if "z_host" in row and "redshift" not in row:
            row = {**row, "redshift": row["z_host"]}
        metadata = {
            k: v for k, v in row.items() if k not in {"name", "ra", "dec", "redshift", "z_host"}
        }
        raw_z = row.get("redshift")
        if metadata.pop("host_z_placeholder", False):
            z_norm = None
        else:
            z_norm = normalize_host_redshift(raw_z)
        sightlines.append(
            Sightline(
                name=row.get("name"),
                source_id=row.get("source_id"),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                redshift=z_norm,
                metadata=metadata,
            )
        )
    return sightlines


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or parquet tabular data."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write CSV or parquet tabular data based on suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
