"""Build the canonical intervening census registry from validated scratch outputs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_SCRATCH_CODETECTION = PACKAGE_DIR.parents[1] / "scratch" / "codetection"

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


def build_intervening_census_registry(scratch_dir: Path | str | None = None) -> pd.DataFrame:
    """Assemble the 49-object registry from validated scratch/codetection CSVs."""
    here = scratch_codetection_dir(scratch_dir)
    fin = pd.read_csv(here / "foreground_final.csv")
    fgr = pd.read_csv(here / "foreground.csv")
    val = pd.read_csv(here / "foreground_validated.csv")
    bur = pd.read_csv(here / "bursts.csv")

    for frame in (fin, fgr, val):
        frame["obj"] = frame["obj"].astype(str)

    tns = dict(zip(bur.nickname, bur.tns, strict=True))
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
