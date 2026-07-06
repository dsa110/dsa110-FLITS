"""Normalize heterogeneous catalog rows into the foreground-candidate schema."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from .provenance import make_provenance

SCHEMA_COLUMNS = [
    "name",
    "id",
    "ra",
    "dec",
    "z",
    "z_type",
    "richness",
    "m_delta",
    "r_delta",
    "delta_def",
    "service",
    "table",
    "provenance_json",
]


@dataclass(frozen=True)
class ColumnMapping:
    """Map normalized schema fields to source DataFrame columns."""

    name: str | None = None
    id: str | None = None
    ra: str | None = None
    dec: str | None = None
    z: str | None = None
    z_type: str | None = None
    richness: str | None = None
    m_delta: str | None = None
    r_delta: str | None = None
    delta_def: str | None = None
    z_source: str | None = None

    def as_provenance_dict(self) -> dict[str, str | None]:
        """Return mapping fields as a deterministic provenance payload."""
        return {
            "name": self.name,
            "id": self.id,
            "ra": self.ra,
            "dec": self.dec,
            "z": self.z,
            "z_type": self.z_type,
            "richness": self.richness,
            "m_delta": self.m_delta,
            "r_delta": self.r_delta,
            "delta_def": self.delta_def,
            "z_source": self.z_source,
        }


def infer_z_type(z_value: object, z_column: str | None) -> str:
    """Infer redshift type from value presence and source-column name."""
    if z_value is None or pd.isna(z_value):
        return "none"
    z_name = (z_column or "").lower()
    if "phot" in z_name or "photo" in z_name:
        return "photo"
    if "spec" in z_name or "zsp" in z_name:
        return "spec"
    return "unknown"


def _series_or_none(df: pd.DataFrame, source: str | None) -> pd.Series:
    if source and source in df.columns:
        return df[source]
    return pd.Series([None] * len(df), index=df.index)


def normalize(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    service: str,
    table: str,
    adql: str | None = None,
    extra_provenance: dict | None = None,
) -> pd.DataFrame:
    """Normalize arbitrary catalog rows into the stable candidate schema."""
    data = {
        "name": _series_or_none(df, mapping.name),
        "id": _series_or_none(df, mapping.id),
        "ra": _series_or_none(df, mapping.ra),
        "dec": _series_or_none(df, mapping.dec),
        "z": _series_or_none(df, mapping.z),
        "z_type": _series_or_none(df, mapping.z_type),
        "richness": _series_or_none(df, mapping.richness),
        "m_delta": _series_or_none(df, mapping.m_delta),
        "r_delta": _series_or_none(df, mapping.r_delta),
        "delta_def": _series_or_none(df, mapping.delta_def),
    }

    out = pd.DataFrame(data, index=df.index)
    if mapping.z_type is None:
        out["z_type"] = out["z"].map(lambda value: infer_z_type(value, mapping.z))

    out["service"] = service
    out["table"] = table

    provenance = make_provenance(
        adql,
        service=service,
        table=table,
        column_mapping=mapping.as_provenance_dict(),
        extra=extra_provenance,
    )
    out["provenance_json"] = provenance

    is_photo = out["z_type"].astype(str) == "photo"
    if is_photo.any():
        photo_provenance = json.loads(provenance)
        photo_provenance["z_prior"] = True
        out.loc[is_photo, "provenance_json"] = json.dumps(photo_provenance, sort_keys=True)

    return out[SCHEMA_COLUMNS].reset_index(drop=True)


def to_common_schema(
    df: pd.DataFrame,
    *,
    ra_col: str,
    dec_col: str,
    z_col: str | None,
    service: str,
    table: str,
    id_col: str | None = None,
    name_col: str | None = None,
    adql: str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for the common RA/Dec/redshift case."""
    return normalize(
        df,
        ColumnMapping(
            name=name_col,
            id=id_col,
            ra=ra_col,
            dec=dec_col,
            z=z_col,
        ),
        service=service,
        table=table,
        adql=adql,
    )
