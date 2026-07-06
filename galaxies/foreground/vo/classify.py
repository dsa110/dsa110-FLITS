"""Halo-intersection classification for reduced foreground-candidate frames.

Operates on the output of ``reduce.merge_and_rank`` (which provides ``b_kpc``
and ``r_delta_computed``). Adds three intersection flags at deterministic
brackets — ``strict`` (``b <= 0.8 * r_vir``), ``nominal`` (``b <= r_vir``), and
``loose`` (``b <= 1.2 * r_vir``) — that stand in for the ~15-20% SHMR scatter
on the R_vir estimate without invoking Monte Carlo machinery.

The summary builder aggregates per-FRB counts and reports the closest
candidate (smallest ``b_over_rvir``), which is what the headline question
("how many sightlines have a foreground halo within R_vir?") needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_intersection_flags(
    df: pd.DataFrame,
    *,
    loose: float = 1.2,
    strict: float = 0.8,
) -> pd.DataFrame:
    """Add ``b_over_rvir`` and three ``intersects_*`` boolean columns."""
    out = df.copy()
    b = pd.to_numeric(out.get("b_kpc"), errors="coerce")
    r = pd.to_numeric(out.get("r_delta_computed"), errors="coerce")

    valid = b.notna() & r.notna() & (r > 0)
    ratio = pd.Series(np.full(len(out), np.nan), index=out.index, dtype=float)
    ratio.loc[valid] = b.loc[valid] / r.loc[valid]

    out["b_over_rvir"] = ratio
    out["intersects_nominal"] = np.where(valid, (b <= r).to_numpy(), False)
    out["intersects_loose"] = np.where(valid, (b <= loose * r).to_numpy(), False)
    out["intersects_strict"] = np.where(valid, (b <= strict * r).to_numpy(), False)
    return out


def summarize_sightlines(df: pd.DataFrame) -> pd.DataFrame:
    """Per-FRB rollup of candidate counts, foreground-only intersection counts, and the closest foreground hit.

    The headline ``n_intersects_*`` counts and ``closest_*`` fields are
    restricted to **foreground** candidates only (rows where ``is_foreground``
    is True, or where ``z < frb_z`` if that column-pair is available; all rows
    if neither is present). Background galaxies that happen to lie within
    R_vir of the line of sight are excluded — they would inflate the headline
    answer to the survey's central question without contributing to it.
    """
    columns = [
        "frb_name",
        "n_candidates",
        "n_foreground",
        "n_intersects_nominal",
        "n_intersects_loose",
        "n_intersects_strict",
        "min_b_over_rvir",
        "closest_name",
        "closest_id",
        "status",
    ]
    if df.empty or "frb_name" not in df.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for frb_name, group in df.groupby("frb_name", sort=True):
        if "is_foreground" in group.columns:
            is_fg = group["is_foreground"].fillna(False).astype(bool)
        elif "frb_z" in group.columns and "z" in group.columns:
            cand_z = pd.to_numeric(group["z"], errors="coerce")
            host_z = pd.to_numeric(group["frb_z"], errors="coerce")
            is_fg = (cand_z < host_z).fillna(False)
        else:
            is_fg = pd.Series([True] * len(group), index=group.index)

        fg_group = group[is_fg]

        ratio = pd.to_numeric(fg_group.get("b_over_rvir"), errors="coerce")
        valid_ratio = ratio.dropna()
        if not valid_ratio.empty:
            min_idx = valid_ratio.idxmin()
            min_b_over_rvir = float(valid_ratio.loc[min_idx])
            closest_name = fg_group.loc[min_idx, "name"] if "name" in fg_group.columns else None
            closest_id = fg_group.loc[min_idx, "id"] if "id" in fg_group.columns else None
        else:
            min_b_over_rvir = float("nan")
            closest_name = None
            closest_id = None

        def _count_fg(col: str, fg_group: pd.DataFrame = fg_group) -> int:
            if col not in fg_group.columns:
                return 0
            return int(fg_group[col].fillna(False).astype(bool).sum())

        rows.append(
            {
                "frb_name": frb_name,
                "n_candidates": len(group),
                "n_foreground": int(is_fg.sum()),
                "n_intersects_nominal": _count_fg("intersects_nominal"),
                "n_intersects_loose": _count_fg("intersects_loose"),
                "n_intersects_strict": _count_fg("intersects_strict"),
                "min_b_over_rvir": min_b_over_rvir,
                "closest_name": closest_name,
                "closest_id": closest_id,
                "status": "ok",
            }
        )

    return pd.DataFrame(rows, columns=columns)


def empty_sightline_row(frb_name: str, status: str) -> dict:
    """Return a zero-filled summary row for a sightline that produced no candidates.

    Use when the upstream cone query returned zero rows, so that the headline
    denominator (number of expected sightlines) stays consistent in the summary
    CSV instead of silently shrinking. ``status`` describes why the row is
    empty (e.g., ``no_candidates_in_cone``).
    """
    return {
        "frb_name": frb_name,
        "n_candidates": 0,
        "n_foreground": 0,
        "n_intersects_nominal": 0,
        "n_intersects_loose": 0,
        "n_intersects_strict": 0,
        "min_b_over_rvir": float("nan"),
        "closest_name": None,
        "closest_id": None,
        "status": status,
    }
