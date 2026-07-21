"""Build the expanded foreground audit catalog from committed snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18

from galaxies.foreground.census_registry import (
    load_adjudicated_masses,
    load_intervening_census_registry,
    load_mass_overrides,
)
from galaxies.foreground.expanded_catalog import (
    dutton_maccio14_c200c,
    m200c_to_r200c,
    select_match,
    stern12_status,
)
from galaxies.foreground.vo.halos import mstar_to_mhalo

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "catalog_crossmatch_snapshots"
OUTPUT_CSV = DATA_DIR / "expanded_catalog_cross_references.csv"
BUILD_JSON = DATA_DIR / "expanded_catalog_build.json"
AMBIGUITY_ARCSEC = 0.3

CATALOG_META = {
    "gsc242": ("gsc242", "GSC2"),
    "allwise": ("allwise", "AllWISE"),
    "catwise2020": ("catwise2020", "Name"),
    "unwise": ("unwise", "objID"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(row: pd.Series) -> str:
    return "|".join((str(row.nickname).lower(), str(row.type), str(row.obj)))


def _load_snapshots() -> dict[str, dict[str, Any]]:
    snapshots = {}
    for name in CATALOG_META:
        path = SNAPSHOT_DIR / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing committed catalog snapshot: {path}")
        payload = json.loads(path.read_text())
        if len(payload.get("queries", [])) != 52:
            raise ValueError(f"{path} does not contain 52 query records")
        snapshots[name] = payload
    return snapshots


def _query_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(q["key"]): q for q in payload["queries"]}


def _match_columns(name: str, payload: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    prefix, native_id = CATALOG_META[name]
    match = select_match(
        query.get("rows"), None, float(payload["search_radius_arcsec"]), AMBIGUITY_ARCSEC
    )
    selected = dict(match.selected_row or {})
    base = {
        f"{prefix}_status": match.status,
        f"{prefix}_id": selected.get(native_id) or selected.get("catalog_id"),
        f"{prefix}_separation_arcsec": match.separation_arcsec,
        f"{prefix}_candidate_count": match.candidate_count,
        f"{prefix}_second_separation_arcsec": match.second_separation_arcsec,
        f"{prefix}_release": payload["release"],
        f"{prefix}_retrieved_at_utc": payload["retrieved_at_utc"],
        f"{prefix}_snapshot_sha256": query["response_sha256"],
    }
    if name == "gsc242":
        base["gsc242_class"] = selected.get("Class")
    elif name == "allwise":
        base.update(
            {
                "allwise_w1_mag": selected.get("W1mag"),
                "allwise_w1_err_mag": selected.get("e_W1mag"),
                "allwise_w2_mag": selected.get("W2mag"),
                "allwise_w2_err_mag": selected.get("e_W2mag"),
                "allwise_qph": selected.get("qph"),
                "allwise_ccf": selected.get("ccf"),
                "allwise_ex": selected.get("ex"),
            }
        )
    elif name == "catwise2020":
        fw1, efw1 = selected.get("FW1pm"), selected.get("e_FW1pm")
        fw2, efw2 = selected.get("FW2pm"), selected.get("e_FW2pm")
        e1 = 2.5 / np.log(10.0) * efw1 / fw1 if fw1 and efw1 else None
        e2 = 2.5 / np.log(10.0) * efw2 / fw2 if fw2 and efw2 else None
        base.update(
            {
                "catwise2020_w1_mag": selected.get("W1mproPM"),
                "catwise2020_w1_err_mag": e1,
                "catwise2020_w2_mag": selected.get("W2mproPM"),
                "catwise2020_w2_err_mag": e2,
                "catwise2020_pmQual": selected.get("pmQual"),
                "catwise2020_abf": selected.get("abf"),
                "catwise2020_ccf": None,
                "catwise2020_ccf_status": "not_published_in_vizier_table",
            }
        )
    else:
        base.update(
            {
                "unwise_w1_flux": selected.get("FW1"),
                "unwise_w1_flux_err": selected.get("e_FW1"),
                "unwise_w2_flux": selected.get("FW2"),
                "unwise_w2_flux_err": selected.get("e_FW2"),
                "unwise_q_w1": selected.get("q_W1"),
                "unwise_q_w2": selected.get("q_W2"),
                "unwise_ff_w1": selected.get("fFW1"),
                "unwise_ff_w2": selected.get("fFW2"),
            }
        )
    return base


def _mass_authority() -> tuple[dict[tuple[str, str], tuple[float, str]], set[tuple[str, str]]]:
    masses: dict[tuple[str, str], tuple[float, str]] = {}
    adjudicated = load_adjudicated_masses()
    for _, row in adjudicated.iterrows():
        value = pd.to_numeric(row.get("logM_adj"), errors="coerce")
        if pd.notna(value) and np.isfinite(float(value)):
            masses[(str(row.nickname).lower(), str(row.obj))] = (float(value), str(row.mass_source))
    for _, row in load_mass_overrides().iterrows():
        masses[(str(row.nickname).lower(), str(row.obj))] = (float(row.logM_adj), str(row.mass_source))
    return masses, set(masses)


def build_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = load_intervening_census_registry().copy()
    registry["obj"] = registry["obj"].astype(str)
    keys = registry[["nickname", "type", "obj"]].astype(str).agg(tuple, axis=1)
    if len(registry) != 52 or not keys.is_unique:
        raise ValueError("expanded catalog requires exactly 52 unique registry rows")
    snapshots = _load_snapshots()
    indices = {name: _query_index(payload) for name, payload in snapshots.items()}
    masses, _ = _mass_authority()
    h = float(Planck18.H0.value / 100.0)
    records: list[dict[str, Any]] = []
    for idx, row in registry.reset_index(drop=True).iterrows():
        record = {
            "registry_idx": idx + 1,
            "nickname": row.nickname,
            "type": row.type,
            "object_id": row.obj,
            "tns": row.tns,
            "ra_deg": row.ra_deg,
            "dec_deg": row.dec_deg,
            "host_z_spec": row.host_z_spec,
            "best_z": row.best_z,
            "best_z_err": row.best_z_err,
            "best_z_source": row.best_z_source,
            "classification": row.classification,
            "final_verdict": row.final_verdict,
            "registry_tier": row.registry_tier,
            "budget_eligible": row.budget_eligible,
            "impact_kpc": row.impact_kpc,
            "m500_1e14msun": row.m500_1e14msun,
            "r500_mpc": row.r500_mpc,
        }
        query_key = _key(row)
        for name, payload in snapshots.items():
            query = indices[name].get(query_key)
            if query is None:
                raise ValueError(f"snapshot {name} lacks {query_key}")
            record.update(_match_columns(name, payload, query))

        w1, w2 = record.get("allwise_w1_mag"), record.get("allwise_w2_mag")
        e1, e2 = record.get("allwise_w1_err_mag"), record.get("allwise_w2_err_mag")
        numeric_valid = all(v is not None and np.isfinite(float(v)) for v in (w1, w2, e1, e2))
        qph = str(record.get("allwise_qph") or "")
        ccf = str(record.get("allwise_ccf") or "")
        if record["allwise_status"] != "matched":
            photometry_status = f"match_{record['allwise_status']}"
        elif not numeric_valid:
            photometry_status = "missing_or_upper_limit"
        elif len(qph) < 2 or "U" in qph[:2] or "X" in qph[:2]:
            photometry_status = "invalid_photometric_quality"
        elif len(ccf) < 2 or ccf[:2] != "00":
            photometry_status = "contaminated_or_artifact_flag"
        else:
            photometry_status = "pass"
        record["allwise_photometry_status"] = photometry_status
        record["allwise_extended_source_flag"] = bool(
            record.get("allwise_ex") is not None and int(record["allwise_ex"]) > 0
        )
        color_valid = numeric_valid and photometry_status == "pass"
        record["w1_w2_color_mag"] = float(w1) - float(w2) if color_valid else None
        record["w1_w2_color_err_mag"] = float(np.hypot(e1, e2)) if color_valid else None
        record["stern12_status"] = stern12_status(
            record["w1_w2_color_mag"] if color_valid else np.nan,
            float(w2) if w2 is not None else np.nan,
            color_valid=color_valid,
        )
        record["cluver14_log_mstar"] = None
        record["cluver14_log_mstar_err"] = None
        record["cluver14_method"] = "Cluver14_Eq1"
        record["cluver14_units"] = "dex(log10_Msun)"
        record["cluver14_authority"] = "diagnostic_only"
        record["cluver14_status"] = "not_rest_frame"

        if row.type == "cluster":
            record.update(
                {
                    "adopted_log_mstar": None,
                    "adopted_log_mstar_err": None,
                    "adopted_mass_authority": "cluster_catalog_M500_R500",
                    "adopted_mass_status": "not_applicable_cluster",
                    "m200c_msun": None,
                    "m200c_err_msun": None,
                    "m200c_method": "not_computed_without_cluster_conversion_model",
                    "m200c_units": "Msun",
                    "m200c_status": "not_applicable_cluster",
                    "r200c_kpc": None,
                    "r200c_err_kpc": None,
                    "r200c_method": "not_computed_without_cluster_conversion_model",
                    "r200c_units": "proper_kpc",
                    "r200c_status": "not_applicable_cluster",
                    "c200c": None,
                    "c200c_status": "not_applicable_cluster",
                    "scale_radius_kpc": None,
                    "scale_radius_status": "not_applicable_cluster",
                    "b_over_r200c": None,
                }
            )
        else:
            authority = masses.get((str(row.nickname).lower(), str(row.obj)))
            if authority is None or pd.isna(row.best_z):
                record.update(
                    {
                        "adopted_log_mstar": None,
                        "adopted_log_mstar_err": None,
                        "adopted_mass_authority": None,
                        "adopted_mass_status": "unavailable",
                        "m200c_msun": None,
                        "m200c_err_msun": None,
                        "m200c_method": "Moster13_Table1_redshift_dependent",
                        "m200c_units": "Msun",
                        "m200c_status": "unavailable",
                        "r200c_kpc": None,
                        "r200c_err_kpc": None,
                        "r200c_method": "200_times_critical_density_Planck18",
                        "r200c_units": "proper_kpc",
                        "r200c_status": "unavailable",
                        "c200c": None,
                        "c200c_status": "unavailable",
                        "scale_radius_kpc": None,
                        "scale_radius_status": "unavailable",
                        "b_over_r200c": None,
                    }
                )
            else:
                log_mstar, source = authority
                m200 = mstar_to_mhalo(10.0**log_mstar, float(row.best_z))
                r200 = m200c_to_r200c(m200, float(row.best_z))
                c200 = dutton_maccio14_c200c(m200, float(row.best_z), h)
                record.update(
                    {
                        "adopted_log_mstar": log_mstar,
                        "adopted_log_mstar_err": None,
                        "adopted_mass_authority": source,
                        "adopted_mass_status": "pass_uncertainty_unavailable",
                        "m200c_msun": m200,
                        "m200c_err_msun": None,
                        "m200c_method": "Moster13_Table1_redshift_dependent",
                        "m200c_units": "Msun",
                        "m200c_status": "pass_uncertainty_unavailable",
                        "r200c_kpc": r200,
                        "r200c_err_kpc": None,
                        "r200c_method": "200_times_critical_density_Planck18",
                        "r200c_units": "proper_kpc",
                        "r200c_status": "pass_uncertainty_unavailable",
                        "c200c": c200,
                        "c200c_status": "pass_model_relation",
                        "scale_radius_kpc": r200 / c200,
                        "scale_radius_status": "pass_model_relation",
                        "b_over_r200c": float(row.impact_kpc) / r200 if pd.notna(row.impact_kpc) else None,
                    }
                )
        records.append(record)
    frame = pd.DataFrame(records)
    manifest = {
        "schema_version": 1,
        "builder": "galaxies.foreground.build_expanded_catalog",
        "row_count": len(frame),
        "registry_sha256": _sha256(DATA_DIR / "intervening_census_registry.csv"),
        "snapshot_sha256": {
            name: _sha256(SNAPSHOT_DIR / f"{name}.json") for name in snapshots
        },
        "match_policy": {
            "search_radius_arcsec": 3.0,
            "sort": ["exact_spherical_separation", "catalog_identifier"],
            "ambiguity_rule": "second_separation_minus_nearest_lte_0.3_arcsec",
        },
        "physics": {
            "stellar_mass_authority": "adjudicated census masses plus overrides",
            "halo_mass": "Moster et al. 2013 Table 1 redshift-dependent M200c",
            "radius": "R200c at 200 times Planck18 critical density",
            "concentration": "Dutton and Maccio 2014 Planck c200c",
        },
    }
    return frame, manifest


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build() -> tuple[Path, Path]:
    frame, manifest = build_frame()
    _atomic_csv(OUTPUT_CSV, frame)
    manifest["output_sha256"] = _sha256(OUTPUT_CSV)
    _atomic_json(BUILD_JSON, manifest)
    return OUTPUT_CSV, BUILD_JSON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="required; build never performs network I/O")
    args = parser.parse_args()
    if not args.offline:
        parser.error("--offline is required; refresh snapshots explicitly")
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
