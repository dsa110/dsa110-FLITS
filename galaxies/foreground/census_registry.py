"""Build the canonical intervening census registry from validated scratch outputs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from scattering.scat_analysis.burst_metadata import load_tns_name

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_SCRATCH_CODETECTION = PACKAGE_DIR.parents[1] / "scratch" / "codetection"
CENSUS_EXTENSIONS_CSV = DATA_DIR / "census_extensions" / "v4_extension.csv"

SURVEY_SHORT = {
    "WISE, PS1, STRM": "WISE/PS1/STRM",
    "Legacy DR8 (Zhou et al. 2021)": "Legacy/Zhou21",
}


def scratch_codetection_dir(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("FLITS_FOREGROUND_SCRATCH")
    if env:
        return Path(env)
    return DEFAULT_SCRATCH_CODETECTION


def _survey_short(survey: str) -> str:
    s = str(survey)
    if "Wen" in s or "DESI, WISE" in s:
        return "DESI/WISE (Wen+)"
    return SURVEY_SHORT.get(s, s)


def _best_redshift(
    row: pd.Series,
    val_row: pd.Series | None,
) -> tuple[float, float, str, str]:
    if pd.notna(row.get("strm_class")):
        cls = str(row.strm_class)
        if pd.notna(row.get("strm_zphot")):
            return (
                float(row.strm_zphot),
                float(row.strm_zphoterr) if pd.notna(row.strm_zphoterr) else np.nan,
                "PS1-STRM phot",
                cls,
            )
        return np.nan, np.nan, "PS1-STRM", cls
    if row.type == "cluster":
        src = row.best_z_source
        if src == "desi_specz":
            zerr = float(val_row.desi_zerr) if val_row is not None and pd.notna(val_row.desi_zerr) else np.nan
            return float(row.best_z), zerr, "DESI spec", "cluster"
        if src == "ned_z":
            return float(row.best_z), np.nan, "NED", "cluster"
        return float(row.best_z), np.nan, "phot", "cluster"
    src = row.best_z_source
    cls = "galaxy"
    if val_row is not None and pd.notna(val_row.get("lsdr9_type")):
        cls = {"PSF": "point src"}.get(str(val_row.lsdr9_type), str(val_row.lsdr9_type))
    if src == "desi_specz":
        zerr = float(val_row.desi_zerr) if val_row is not None and pd.notna(val_row.desi_zerr) else np.nan
        return float(row.best_z), zerr, "DESI spec", cls
    if src == "lsdr9_zspec":
        return float(row.best_z), 0.0, "LS DR9 spec", cls
    if src == "lsdr9_zphot":
        zerr = float(val_row.lsdr9_zphot_std) if val_row is not None and pd.notna(val_row.lsdr9_zphot_std) else np.nan
        return float(row.best_z), zerr, "LS/Zhou phot", cls
    if src == "ned_z":
        return float(row.best_z), np.nan, "NED", cls
    return np.nan, np.nan, "none", cls


def budget_eligible(final_verdict: str, obj_type: str, b_over_r500: float) -> bool:
    """Registry-tier vs budget-tier gate (see pipeline/CONTEXT.md)."""
    if final_verdict != "confirmed":
        return False
    if obj_type == "cluster":
        return np.isfinite(b_over_r500) and float(b_over_r500) <= 1.0
    return True


def load_census_extensions(path: Path | str | None = None) -> pd.DataFrame:
    """Load manually adjudicated rows omitted by the frozen validation handoff.

    The extension is append-only provenance, not a second validation engine.
    Registry and budget booleans are recomputed here so the CSV cannot promote
    a row by carrying a stale hand-entered flag.
    """
    csv_path = Path(path) if path is not None else CENSUS_EXTENSIONS_CSV
    ext = pd.read_csv(csv_path, dtype={"obj": str})
    keys = ext[["nickname", "type", "obj"]].astype(str).agg(tuple, axis=1)
    if not keys.is_unique:
        raise ValueError(f"duplicate census-extension key in {csv_path}")
    ext["registry_tier"] = ext.final_verdict == "confirmed"
    ext["budget_eligible"] = [
        budget_eligible(verdict, obj_type, b_over_r500)
        for verdict, obj_type, b_over_r500 in zip(
            ext.final_verdict,
            ext.type,
            pd.to_numeric(ext.b_over_r500, errors="coerce"),
            strict=True,
        )
    ]
    return ext


def _append_census_extensions(registry: pd.DataFrame) -> pd.DataFrame:
    """Append the V4 extension idempotently, with the extension as authority."""
    ext = load_census_extensions()
    missing = set(registry.columns) - set(ext.columns)
    extra = set(ext.columns) - set(registry.columns)
    if missing or extra:
        raise ValueError(
            f"census-extension schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    ext_keys = set(ext[["nickname", "type", "obj"]].astype(str).agg(tuple, axis=1))
    base_keys = registry[["nickname", "type", "obj"]].astype(str).agg(tuple, axis=1)
    combined = pd.concat(
        [registry.loc[~base_keys.isin(ext_keys)], ext[registry.columns]],
        ignore_index=True,
    )
    keys = combined[["nickname", "type", "obj"]].astype(str).agg(tuple, axis=1)
    if not keys.is_unique:
        raise ValueError("duplicate stable key after appending census extension")
    return combined


def build_intervening_census_registry(scratch_dir: Path | str | None = None) -> pd.DataFrame:
    """Assemble the 49-object registry from validated scratch/codetection CSVs."""
    here = scratch_codetection_dir(scratch_dir)
    required = [
        here / "foreground_final.csv",
        here / "foreground.csv",
        here / "foreground_validated.csv",
        here / "bursts.csv",
    ]
    if any(not path.exists() for path in required):
        checked_in = DATA_DIR / "intervening_census_registry.csv"
        if scratch_dir is None and checked_in.exists():
            return _append_census_extensions(pd.read_csv(checked_in, dtype={"obj": str}))
    fin = pd.read_csv(here / "foreground_final.csv")
    fgr = pd.read_csv(here / "foreground.csv")
    val = pd.read_csv(here / "foreground_validated.csv")
    bur = pd.read_csv(here / "bursts.csv")

    for frame in (fin, fgr, val):
        frame["obj"] = frame["obj"].astype(str)

    tns = {nickname: load_tns_name(nickname) for nickname in bur.nickname}
    fgr_i = fgr.set_index(["nickname", "type", "obj"])
    val_i = val.set_index(["nickname", "type", "obj"])

    rows: list[dict] = []
    for _, r in fin.iterrows():
        key = (r.nickname, r.type, r.obj)
        fg = fgr_i.loc[key] if key in fgr_i.index else None
        v = val_i.loc[key] if key in val_i.index else None
        z, zerr, zsrc, cls = _best_redshift(r, v)
        impact = (
            float(fg.impact_kpc_listed)
            if fg is not None and pd.notna(fg.impact_kpc_listed)
            else np.nan
        )
        b_over_r500 = (
            float(fg.b_over_r500) if fg is not None and pd.notna(fg.b_over_r500) else np.nan
        )
        m500 = (
            float(fg.m500_1e14msun)
            if fg is not None and pd.notna(getattr(fg, "m500_1e14msun", np.nan))
            else np.nan
        )
        r500 = (
            float(fg.r500_mpc)
            if fg is not None and pd.notna(getattr(fg, "r500_mpc", np.nan))
            else np.nan
        )
        verdict = str(r.final_verdict)
        rows.append(
            {
                "nickname": r.nickname,
                "type": r.type,
                "obj": r.obj,
                "tns": tns.get(r.nickname, ""),
                "host_z_spec": float(r.host_z_spec),
                "survey": _survey_short(r.survey),
                "ra_deg": float(r.ra_deg),
                "dec_deg": float(r.dec_deg),
                "impact_kpc": round(impact, 1) if np.isfinite(impact) else np.nan,
                "b_over_r500": round(b_over_r500, 2) if np.isfinite(b_over_r500) else np.nan,
                "m500_1e14msun": round(m500, 3) if np.isfinite(m500) else np.nan,
                "r500_mpc": round(r500, 3) if np.isfinite(r500) else np.nan,
                "best_z": round(z, 4) if np.isfinite(z) else np.nan,
                "best_z_err": round(zerr, 4) if np.isfinite(zerr) else np.nan,
                "best_z_source": zsrc,
                "classification": cls,
                "final_verdict": verdict,
                "final_reason": str(r.final_reason),
                "registry_tier": verdict == "confirmed",
                "budget_eligible": budget_eligible(verdict, r.type, b_over_r500),
                "provenance_scratch_final": "foreground_final.csv",
                "provenance_scratch_geometry": "foreground.csv",
                "provenance_scratch_validation": "foreground_validated.csv",
            }
        )

    return _append_census_extensions(pd.DataFrame(rows))


def load_intervening_census_registry(path: Path | str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DATA_DIR / "intervening_census_registry.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    return build_intervening_census_registry()


ADJUDICATED_MASSES_CSV = DATA_DIR / "census_masses" / "halo_rvir_ADJUDICATED.csv"
CENSUS_DUPLICATES_CSV = DATA_DIR / "census_masses" / "census_duplicates.csv"
MASS_OVERRIDES_CSV = DATA_DIR / "census_masses" / "mass_overrides.csv"


def load_census_duplicates(path: Path | str | None = None) -> dict[tuple[str, str], str]:
    """(nickname, duplicate_obj) -> canonical_obj for cross-listed census rows.

    Owner adjudication 2026-07-15: seven halo pairs (five confirmed, two
    refuted) sit at <0.2 arcsec separation with identical redshifts -- the
    same physical galaxy carried under two catalog identifiers. The physical
    census is deduplicated by dropping the duplicate member; the canonical
    member is the catalog-resolvable LS DR9 objid.
    """
    csv_path = Path(path) if path is not None else CENSUS_DUPLICATES_CSV
    df = pd.read_csv(csv_path, dtype={"duplicate_obj": str, "canonical_obj": str})
    return {
        (str(r.nickname).lower(), str(r.duplicate_obj)): str(r.canonical_obj)
        for _, r in df.iterrows()
    }


def load_mass_overrides(path: Path | str | None = None) -> pd.DataFrame:
    """Owner mass adjudications that supersede the B7 adjudicated table.

    2026-07-15: whitney obj 1473's B7 wise_w1 mass (logM*=11.279) is a WISE
    blend (LS DR9 forced photometry shows the on-source galaxy at W1=22.9);
    the adopted mass is the optical Zibetti-2009 g-z estimate. See the CSV's
    evidence column for the photometry.
    """
    csv_path = Path(path) if path is not None else MASS_OVERRIDES_CSV
    df = pd.read_csv(csv_path, dtype={"obj": str})
    df["nickname"] = df["nickname"].str.lower()
    return df


def recompute_impact_kpc(
    sight_ra_deg: float,
    sight_dec_deg: float,
    obj_ra_deg: float,
    obj_dec_deg: float,
    z: float,
) -> float:
    """Proper impact parameter (kpc) from the V6 burst position and object coordinates.

    Owner adjudication 2026-07-15: the registry's listed impact parameters are
    provenance-heterogeneous (duplicate pair members carried inconsistent
    values, up to 48 kpc apart); the budget uses this uniform recomputation
    (fiducial cosmology of galaxies.foreground.config) instead.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from galaxies.foreground import config

    sep = SkyCoord(sight_ra_deg * u.deg, sight_dec_deg * u.deg).separation(
        SkyCoord(obj_ra_deg * u.deg, obj_dec_deg * u.deg)
    )
    kpc_per_arcmin = config.COSMO.kpc_proper_per_arcmin(float(z)).to(u.kpc / u.arcmin).value
    return float(sep.arcmin * kpc_per_arcmin)


def load_adjudicated_masses(path: Path | str | None = None) -> pd.DataFrame:
    """The B7 empirical-mass table for the frozen census halos.

    Produced 2026-07-08 (handoff-2026-07-08-18-12-b7-cgm-census-resolved.md):
    per-halo stellar masses from live PS1 g-i (Taylor 2011) / WISE W1
    (Cluver 2014) photometry through the pipeline's own mass ladder
    (select_stellar_mass -> estimate_halo_mass -> get_rvir_and_rs), with the
    eight logM>11.3 suspects human-adjudicated (suspect_vetting_adjudicated.csv;
    owner decision). ``logM_adj`` is the adjudicated log10 stellar mass;
    rows whose mass_status is not a measured/adjudicated pass carry no
    ``logM_adj`` and fall back to the ladder's default behavior downstream.

    Original artifact provenance: Claude Science artifact store, project
    proj_55f9c893cfe1, version d3fd91ff-6bb8-4b94-b185-0a36d0c8fdbe.
    """
    csv_path = Path(path) if path is not None else ADJUDICATED_MASSES_CSV
    df = pd.read_csv(csv_path, dtype={"obj": str})
    df["nickname"] = df["nickname"].str.lower()
    return df


def census_roster_nicknames() -> frozenset[str]:
    """Lowercased nicknames of the census bursts (frozen V4 roster).

    The registry is *authoritative* for these sightlines: a burst on this
    roster with no budget-eligible registry rows has, as a census verdict, no
    confirmed budget-eligible foreground -- consumers must not fall back to
    the legacy pre-V4 candidate lists for it. Nicknames outside the roster
    (synthetic test bursts, future events) are unknown to the census and may
    use their own acquisition paths.
    """
    bursts_csv = DATA_DIR / "frozen_census" / "bursts.csv"
    df = pd.read_csv(bursts_csv)
    return frozenset(str(n).lower() for n in df["nickname"])


def registry_to_matches(
    registry: pd.DataFrame,
    nickname: str,
    z_frb: float,
    *,
    adjudicated_masses: pd.DataFrame | None = None,
    sight_ra_deg: float | None = None,
    sight_dec_deg: float | None = None,
) -> pd.DataFrame:
    """Budget-eligible confirmed registry rows as a ``build_unified_records`` matches table.

    Owner adjudications applied (2026-07-15):

    1. *Dedupe* — rows listed in ``census_duplicates.csv`` (same physical
       galaxy under two catalog identifiers) are dropped in favor of their
       canonical member.
    2. *Empirical masses govern* — halo rows are annotated with the B7
       adjudicated stellar mass (``logM_adj`` + ``mass_source_adj``), as
       superseded by ``mass_overrides.csv`` (e.g. the whitney-1473 WISE-blend
       correction). Rows without an adjudicated mass keep the ladder's
       default behavior.
    3. *Uniform geometry* — when the burst position is provided, the impact
       parameter is recomputed from (position, coordinates, best_z) rather
       than trusting the provenance-heterogeneous listed value.
    """
    # The adjudication inputs are committed repo data; a missing file is a
    # broken checkout and must raise, never silently disable the remediation
    # (review finding on this PR: a swallowed FileNotFoundError here would
    # un-dedupe the census and drop every mass adjudication without warning).
    if adjudicated_masses is None:
        adjudicated_masses = load_adjudicated_masses()
    adj_by_key: dict[tuple[str, str], dict] = {}
    for _, a in adjudicated_masses.iterrows():
        logm = pd.to_numeric(a.get("logM_adj"), errors="coerce")
        if pd.notna(logm) and np.isfinite(float(logm)):
            adj_by_key[(str(a.nickname).lower(), str(a.obj))] = {
                "logM_adj": float(logm),
                "mass_source_adj": str(a.get("mass_source", "census_adjudicated")),
            }
    for _, o in load_mass_overrides().iterrows():
        adj_by_key[(str(o.nickname).lower(), str(o.obj))] = {
            "logM_adj": float(o.logM_adj),
            "mass_source_adj": str(o.mass_source),
        }
    duplicates = load_census_duplicates()

    sub = registry[
        (registry.nickname.str.lower() == nickname.lower())
        & (registry.final_verdict == "confirmed")
        & (registry.budget_eligible)
    ]
    rows: list[dict] = []
    for _, r in sub.iterrows():
        if (str(r.nickname).lower(), str(r.obj)) in duplicates:
            continue  # same physical system as its canonical member
        z = float(r.best_z)
        if not (np.isfinite(z) and z < float(z_frb)):
            continue
        impact = float(r.impact_kpc) if np.isfinite(r.impact_kpc) else np.nan
        if (
            r.type != "cluster"  # cluster keeps its analysis-provenance b
            and sight_ra_deg is not None
            and sight_dec_deg is not None
            and np.isfinite(z)
            and z > 0.0
        ):
            impact = recompute_impact_kpc(
                sight_ra_deg, sight_dec_deg, float(r.ra_deg), float(r.dec_deg), z
            )
        row: dict = {
            "ra": float(r.ra_deg),
            "dec": float(r.dec_deg),
            "z": z,
            "impact_kpc": impact,
            "catalog": f"registry:{r.survey}",
            "classification": "GClstr" if r.type == "cluster" else str(r.classification),
        }
        if r.type == "cluster" and np.isfinite(r.get("m500_1e14msun", np.nan)):
            row["m500_msun"] = float(r.m500_1e14msun) * 1e14
        if np.isfinite(r.get("r500_mpc", np.nan)):
            row["R500_mpc"] = float(r.r500_mpc)
        if r.type != "cluster":
            adj = adj_by_key.get((str(r.nickname).lower(), str(r.obj)))
            if adj is not None:
                row.update(adj)
        rows.append(row)
    return pd.DataFrame(rows)


def write_intervening_census_registry(
    path: Path | str | None = None,
    scratch_dir: Path | str | None = None,
) -> Path:
    out = Path(path) if path is not None else DATA_DIR / "intervening_census_registry.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_intervening_census_registry(scratch_dir=scratch_dir)
    df.to_csv(out, index=False)
    return out
