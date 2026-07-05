"""Normalize heterogeneous catalog tables into the common candidate schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

import pandas as pd

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


@dataclass
class ColumnMapping:
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
    z_source: str | None = None  # optional metadata: 'velocity' if cz/v_rad detected


def _infer_z_type(z_value, z_source_col: str | None) -> str:
    if z_value is None or (isinstance(z_value, float) and pd.isna(z_value)):
        return "none"
    if z_source_col:
        s = z_source_col.lower()
        if "phot" in s:
            return "photo"
        if "spec" in s:
            return "spec"
    return "unknown"


def normalize(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    service: str,
    table: str,
    extra_provenance: dict | None = None,
) -> pd.DataFrame:
    """Map arbitrary columns into the common schema.

    Returns a DataFrame with stable column order and a ``provenance_json``
    column (sorted keys) recording service, table, and the column mapping.
    """
    data = {}
    for field in SCHEMA_COLUMNS[:10]:  # all mapped fields, excluding service/table/provenance
        src = getattr(mapping, field)
        data[field] = df[src] if src in df.columns else pd.Series([None] * len(df))

    if mapping.z_type is None and mapping.z is not None:
        data["z_type"] = df[mapping.z].map(lambda v: _infer_z_type(v, mapping.z))

    out = pd.DataFrame(data)
    out["service"] = service
    out["table"] = table

    prov_base = {
        "service": service,
        "table": table,
        "column_mapping": {f.name: getattr(mapping, f.name) for f in fields(mapping)},
    }
    if extra_provenance:
        prov_base.update(extra_provenance)
    out["provenance_json"] = json.dumps(prov_base, sort_keys=True)

    return out[SCHEMA_COLUMNS]


def to_common_schema(
    df: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    z_col: str,
    service: str,
    table: str,
    id_col: str | None = None,
) -> pd.DataFrame:
    """Map a raw DataFrame to the common schema with z_type inference.

    Classifies z_type from the column header (spec vs photo, else per-value).
    Photo-z rows are preserved and marked ``z_prior=true`` in provenance.
    """
    mapping = ColumnMapping(id=id_col, ra=ra_col, dec=dec_col, z=z_col)
    z_header = (z_col or "").lower()
    if "phot" in z_header:
        explicit_zt = "photo"
    elif "spec" in z_header:
        explicit_zt = "spec"
    else:
        explicit_zt = None

    out = normalize(df, mapping, service=service, table=table)
    if explicit_zt is not None:
        out.loc[:, "z_type"] = explicit_zt

    is_photo = out["z_type"].astype(str) == "photo"
    if is_photo.any():
        prov = json.loads(out.loc[is_photo.index[0], "provenance_json"]) if len(out) else {}
        prov["z_prior"] = True
        out.loc[is_photo, "provenance_json"] = json.dumps(prov, sort_keys=True)
    return out
