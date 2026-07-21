"""Refresh normalized, auditable VizieR snapshots for the expanded catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from galaxies.foreground.census_registry import load_intervening_census_registry

DATA_DIR = Path(__file__).resolve().parent / "data"
SNAPSHOT_DIR = DATA_DIR / "catalog_crossmatch_snapshots"
SEARCH_RADIUS_ARCSEC = 3.0

CATALOGS = {
    "gsc242": {
        "table": "I/353/gsc242",
        "release": "GSC 2.4.2",
        "id": "GSC2",
        "ra": "RA_ICRS",
        "dec": "DE_ICRS",
        "fields": ["GSC2", "objID", "Class"],
    },
    "allwise": {
        "table": "II/328/allwise",
        "release": "AllWISE",
        "id": "AllWISE",
        "ra": "RAJ2000",
        "dec": "DEJ2000",
        "fields": ["AllWISE", "W1mag", "e_W1mag", "W2mag", "e_W2mag", "qph", "ccf", "ex"],
    },
    "catwise2020": {
        "table": "II/365/catwise",
        "release": "CatWISE2020",
        "id": "Name",
        "ra": "RA_ICRS",
        "dec": "DE_ICRS",
        "fields": [
            "Name", "W1mproPM", "W2mproPM", "FW1pm", "e_FW1pm", "FW2pm",
            "e_FW2pm", "pmQual", "abf",
        ],
    },
    "unwise": {
        "table": "II/363/unwise",
        "release": "unWISE Catalog",
        "id": "objID",
        "ra": "RAJ2000",
        "dec": "DEJ2000",
        "fields": ["objID", "FW1", "e_FW1", "FW2", "e_FW2", "q_W1", "q_W2", "fFW1", "fFW2"],
    },
}


def _plain(value: Any) -> Any:
    if np.ma.is_masked(value) or value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    text = str(value).strip()
    return text or None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _query_key(row: pd.Series) -> str:
    return "|".join((str(row.nickname).lower(), str(row.type), str(row.obj)))


def query_catalog(registry: pd.DataFrame, name: str, retrieved_at: str) -> dict[str, Any]:
    spec = CATALOGS[name]
    targets = SkyCoord(
        registry.ra_deg.to_numpy(float) * u.deg,
        registry.dec_deg.to_numpy(float) * u.deg,
        frame="icrs",
    )
    vizier = Vizier(columns=["*", "_r"], row_limit=-1)
    result = vizier.query_region(
        targets, radius=SEARCH_RADIUS_ARCSEC * u.arcsec, catalog=spec["table"]
    )
    table = result[0] if result else None
    grouped: dict[int, list[Any]] = {}
    if table is not None:
        for row in table:
            grouped.setdefault(int(row["_q"]) - 1, []).append(row)

    queries: list[dict[str, Any]] = []
    for index, target in registry.reset_index(drop=True).iterrows():
        rows: list[dict[str, Any]] = []
        for raw in grouped.get(index, []):
            ra = float(raw[spec["ra"]])
            dec = float(raw[spec["dec"]])
            separation = SkyCoord(target.ra_deg * u.deg, target.dec_deg * u.deg).separation(
                SkyCoord(ra * u.deg, dec * u.deg)
            ).arcsec
            normalized = {
                "catalog_id": str(_plain(raw[spec["id"]])),
                "match_ra_deg": ra,
                "match_dec_deg": dec,
                "separation_arcsec": float(separation),
            }
            for field in spec["fields"]:
                normalized[field] = _plain(raw[field]) if field in raw.colnames else None
            rows.append(normalized)
        rows.sort(key=lambda r: (r["separation_arcsec"], r["catalog_id"]))
        response = {
            "key": _query_key(target),
            "target_ra_deg": float(target.ra_deg),
            "target_dec_deg": float(target.dec_deg),
            "query_status": "ok",
            "rows": rows,
        }
        response["response_sha256"] = hashlib.sha256(_canonical_bytes(response)).hexdigest()
        queries.append(response)
    return {
        "schema_version": 1,
        "catalog": name,
        "table": spec["table"],
        "release": spec["release"],
        "retrieved_at_utc": retrieved_at,
        "search_radius_arcsec": SEARCH_RADIUS_ARCSEC,
        "queries": queries,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def refresh(*, allow_partial_refresh: bool = False) -> list[Path]:
    registry = load_intervening_census_registry()
    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    snapshots: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for name in CATALOGS:
        try:
            snapshots[name] = query_catalog(registry, name, retrieved_at)
        except Exception as exc:  # network failures are evidence, never unmatched rows
            errors[name] = f"{type(exc).__name__}: {exc}"
    if errors:
        if allow_partial_refresh:
            path = SNAPSHOT_DIR / f"partial-refresh-{retrieved_at.replace(':', '')}.json"
            _atomic_json(path, {"retrieved_at_utc": retrieved_at, "errors": errors, "snapshots": snapshots})
            return [path]
        raise RuntimeError(f"catalog refresh failed without overwriting snapshots: {errors}")
    paths = []
    for name, payload in snapshots.items():
        path = SNAPSHOT_DIR / f"{name}.json"
        _atomic_json(path, payload)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial-refresh", action="store_true")
    args = parser.parse_args()
    for path in refresh(allow_partial_refresh=args.allow_partial_refresh):
        print(path)


if __name__ == "__main__":
    main()
