"""Freeze candidate-redshift source rows for the foreground census."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.ma as ma
import pandas as pd
import pyvo
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier

DATA = Path(__file__).parent / "data"
FROZEN = DATA / "frozen_census"
REGISTRY = DATA / "intervening_census_registry.csv"
VALIDATED = FROZEN / "foreground_validated.csv"
STRM = FROZEN / "strm_catalog_rows.csv"
OUT = DATA / "candidate_redshift_provenance.csv"

TAP = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")
RETRIEVED_AT_UTC = datetime.now(UTC).replace(microsecond=0).isoformat()

FIELDNAMES = [
    "nickname",
    "type",
    "obj",
    "source_family",
    "source_release",
    "retrieved_at_utc",
    "stable_source_id",
    "source_row_sha256",
    "query_response_sha256",
    "adopted_z",
    "adopted_z_err",
    "measurement_kind",
    "source_disposition",
    "final_verdict",
    "budget_eligible",
]


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if value is ma.masked:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _table_rows(table) -> list[dict[str, Any]]:
    rows = []
    for row in table:
        rows.append({name: _clean(row[name]) for name in table.colnames})
    return rows


def _bbox(ra: float, dec: float, rad_as: float) -> tuple[float, float, float, float]:
    ddec = rad_as / 3600.0
    dra = ddec / max(np.cos(np.deg2rad(dec)), 0.02)
    return ra - dra, ra + dra, dec - ddec, dec + ddec


def _nearest(rows: list[dict[str, Any]], ra: float, dec: float, racol: str, deccol: str):
    c0 = SkyCoord(ra * u.deg, dec * u.deg)
    coords = SkyCoord(
        [float(r[racol]) for r in rows] * u.deg,
        [float(r[deccol]) for r in rows] * u.deg,
    )
    sep = c0.separation(coords).arcsec
    idx = int(np.argmin(sep))
    row = dict(rows[idx])
    row["matched_separation_arcsec"] = float(sep[idx])
    return row


def _source_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "matched_separation_arcsec"}


def _ls_source(ra: float, dec: float, expected_z: float) -> dict[str, str]:
    rad_as = 5.0
    r0, r1, d0, d1 = _bbox(ra, dec, rad_as)
    query = (
        "SELECT t.ls_id, t.ra, t.dec, t.type, p.z_phot_mean, p.z_phot_std, "
        "p.z_phot_median, p.z_spec "
        "FROM ls_dr9.tractor t JOIN ls_dr9.photo_z p ON t.ls_id = p.ls_id "
        f"WHERE t.dec BETWEEN {d0} AND {d1} AND t.ra BETWEEN {r0} AND {r1}"
    )
    rows = _table_rows(TAP.search(query).to_table())
    if not rows:
        raise RuntimeError(f"Legacy Survey query returned no rows at {ra}, {dec}")
    row = _nearest(rows, ra, dec, "ra", "dec")
    if abs(float(row["z_phot_mean"]) - expected_z) > 5e-4:
        raise RuntimeError(f"Legacy Survey redshift mismatch for {ra}, {dec}: {row}")
    response = {"service": "NOIRLab Data Lab TAP", "query": query, "rows": rows}
    return {
        "source_family": "Legacy Survey/Zhou21",
        "source_release": "NOIRLab Data Lab ls_dr9.tractor + ls_dr9.photo_z",
        "stable_source_id": f"ls_id:{row['ls_id']}",
        "source_row_sha256": _sha256(_source_row(row)),
        "query_response_sha256": _sha256(response),
        "measurement_kind": "photometric",
    }


def _desi_source(ra: float, dec: float, expected_z: float, row_type: str) -> dict[str, str]:
    rad_as = 90.0 if row_type == "cluster" else 5.0
    r0, r1, d0, d1 = _bbox(ra, dec, rad_as)
    query = (
        "SELECT targetid, mean_fiber_ra, mean_fiber_dec, z, zerr, zwarn, "
        "spectype, survey, program "
        "FROM desi_dr1.zpix "
        f"WHERE mean_fiber_dec BETWEEN {d0} AND {d1} "
        f"AND mean_fiber_ra BETWEEN {r0} AND {r1} AND zwarn = 0"
    )
    rows = _table_rows(TAP.search(query).to_table())
    if not rows:
        raise RuntimeError(f"DESI query returned no rows at {ra}, {dec}")
    row = _nearest(rows, ra, dec, "mean_fiber_ra", "mean_fiber_dec")
    if abs(float(row["z"]) - expected_z) > 5e-4:
        raise RuntimeError(f"DESI redshift mismatch for {ra}, {dec}: {row}")
    response = {"service": "NOIRLab Data Lab TAP", "query": query, "rows": rows}
    return {
        "source_family": "DESI",
        "source_release": "NOIRLab Data Lab desi_dr1.zpix",
        "stable_source_id": f"targetid:{row['targetid']}",
        "source_row_sha256": _sha256(_source_row(row)),
        "query_response_sha256": _sha256(response),
        "measurement_kind": "spectroscopic",
    }


def _strm_source(obj: str, row_by_obj: dict[str, dict[str, Any]]) -> dict[str, str]:
    row = row_by_obj.get(str(obj))
    if row is None:
        raise RuntimeError(f"Missing PS1-STRM row for {obj}")
    return {
        "source_family": "PS1-STRM",
        "source_release": "PS1-STRM HLSP declination strips p69-p77",
        "stable_source_id": f"objID:{row['objID']}",
        "source_row_sha256": _sha256(_source_row(row)),
        "query_response_sha256": "not_applicable",
        "measurement_kind": (
            "photometric" if _clean(row.get("z_phot")) not in (None, -999.0) else "no_trustworthy_redshift"
        ),
    }


def _ned_source(ra: float, dec: float, expected_z: float) -> dict[str, str]:
    table = Ned.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=5.0 * u.arcsec)
    rows = _table_rows(table)
    if not rows:
        raise RuntimeError(f"NED query returned no rows at {ra}, {dec}")
    row = _nearest(rows, ra, dec, "RA", "DEC")
    if abs(float(row["Redshift"]) - expected_z) > 5e-4:
        raise RuntimeError(f"NED redshift mismatch for {ra}, {dec}: {row}")
    response = {
        "service": "NASA/IPAC Extragalactic Database",
        "query": {"ra_deg": ra, "dec_deg": dec, "radius_arcsec": 5.0},
        "rows": rows,
    }
    return {
        "source_family": "NED",
        "source_release": "NED object search result",
        "stable_source_id": f"ned_name:{row['Object Name']}",
        "source_row_sha256": _sha256(_source_row(row)),
        "query_response_sha256": _sha256(response),
        "measurement_kind": "spectroscopic",
    }


def _whl_source(ra: float, dec: float, expected_z: float) -> dict[str, str]:
    Vizier.ROW_LIMIT = -1
    table = Vizier(columns=["**"]).query_region(
        SkyCoord(ra * u.deg, dec * u.deg),
        radius=2.0 * u.arcmin,
        catalog="J/ApJS/199/34/table1",
    )[0]
    rows = _table_rows(table)
    row = _nearest(rows, ra, dec, "RAJ2000", "DEJ2000")
    if str(row["WHL"]) != "J115048.0+714428":
        raise RuntimeError(f"WHL12 identity mismatch: {row}")
    if abs(float(row["zph"]) - expected_z) > 5e-4:
        raise RuntimeError(f"WHL12 redshift mismatch: {row}")
    response = {
        "service": "VizieR",
        "catalog": "J/ApJS/199/34/table1",
        "query": {"ra_deg": ra, "dec_deg": dec, "radius_arcmin": 2.0},
        "rows": rows,
    }
    return {
        "source_family": "WHL12",
        "source_release": "VizieR J/ApJS/199/34/table1",
        "stable_source_id": f"recno:{row['recno']};WHL:{row['WHL']}",
        "source_row_sha256": _sha256(row),
        "query_response_sha256": _sha256(response),
        "measurement_kind": "catalog_cluster",
    }


def _blank_source() -> dict[str, str]:
    return {
        "source_family": "extension_no_adopted_candidate_redshift",
        "source_release": "not_applicable",
        "stable_source_id": "not_applicable",
        "source_row_sha256": "not_applicable",
        "query_response_sha256": "not_applicable",
        "measurement_kind": "no_trustworthy_redshift",
    }


def _source_for(row, validated, strm_by_obj: dict[str, dict[str, Any]]) -> dict[str, str]:
    expected_z = _clean(row.best_z)
    source = str(row.best_z_source)
    if expected_z is None:
        return _blank_source()
    if "LS/Zhou" in source:
        return _ls_source(float(row.ra_deg), float(row.dec_deg), float(expected_z))
    if "DESI" in source:
        return _desi_source(float(row.ra_deg), float(row.dec_deg), float(expected_z), str(row.type))
    if "PS1-STRM" in source:
        return _strm_source(str(row.obj), strm_by_obj)
    if source == "NED":
        return _ned_source(float(row.ra_deg), float(row.dec_deg), float(expected_z))
    if "WHL12" in source:
        return _whl_source(float(row.ra_deg), float(row.dec_deg), float(expected_z))
    raise RuntimeError(f"Unsupported candidate redshift source: {source}")


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    validated = pd.read_csv(VALIDATED)
    strm_rows = pd.read_csv(STRM).to_dict("records")
    strm_by_obj = {str(r["objID"]): r for r in strm_rows}

    records = []
    for row in registry.itertuples(index=False):
        source = _source_for(row, validated, strm_by_obj)
        records.append(
            {
                "nickname": row.nickname,
                "type": row.type,
                "obj": str(row.obj),
                **source,
                "retrieved_at_utc": RETRIEVED_AT_UTC,
                "adopted_z": "" if _clean(row.best_z) is None else row.best_z,
                "adopted_z_err": "" if _clean(row.best_z_err) is None else row.best_z_err,
                "source_disposition": "frozen" if _clean(row.best_z) is not None else "no adopted redshift",
                "final_verdict": row.final_verdict,
                "budget_eligible": str(row.budget_eligible),
            }
        )

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
