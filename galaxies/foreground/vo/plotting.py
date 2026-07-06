"""Plotting helpers for foreground-candidate search outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetColumns:
    """Column names needed to join candidate rows to FRB target metadata."""

    name: str = "TNS"
    ra: str = "RA_deg"
    dec: str = "Dec_deg"
    redshift: str = "z_spectroscopic"


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def safe_slug(value: str) -> str:
    """Return a filesystem-safe slug for a target name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return slug.strip("_") or "target"


def load_targets(
    path: str | Path,
    *,
    columns: TargetColumns | None = None,
) -> pd.DataFrame:
    """Load target metadata from CSV and normalize join columns."""
    if columns is None:
        columns = TargetColumns()
    targets = pd.read_csv(path).copy()
    required = [columns.name, columns.ra, columns.dec]
    missing = [column for column in required if column not in targets.columns]
    if missing:
        raise ValueError(f"target table missing required columns: {missing}")
    out = targets.rename(
        columns={
            columns.name: "frb_name",
            columns.ra: "frb_ra",
            columns.dec: "frb_dec",
            columns.redshift: "frb_z_from_targets",
        }
    )
    out["frb_name"] = out["frb_name"].astype(str).str.strip()
    return out


def add_offsets(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Join target coordinates and add tangent-plane offsets in arcminutes."""
    df = candidates.copy()
    df["frb_name"] = df["frb_name"].astype(str).str.strip()
    target_cols = [column for column in ["frb_name", "frb_ra", "frb_dec"] if column in targets]
    merged = df.merge(targets[target_cols], on="frb_name", how="left", validate="many_to_one")
    if merged["frb_ra"].isna().any() or merged["frb_dec"].isna().any():
        missing = sorted(merged.loc[merged["frb_ra"].isna(), "frb_name"].unique())
        raise ValueError(f"missing target coordinates for: {missing}")
    dec0_rad = np.deg2rad(merged["frb_dec"].astype(float))
    merged["dra_arcmin"] = (
        (merged["ra"].astype(float) - merged["frb_ra"].astype(float)) * np.cos(dec0_rad) * 60.0
    )
    merged["ddec_arcmin"] = (merged["dec"].astype(float) - merged["frb_dec"].astype(float)) * 60.0
    return merged


def plot_candidate_sky_map(
    rows: pd.DataFrame,
    *,
    frb_name: str,
    output: str | Path,
    search_radius_arcmin: float | None = None,
    label_top_n: int = 5,
) -> Path:
    """Plot candidate sky offsets around one FRB."""
    plt = _import_matplotlib()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    ax.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax.axvline(0, color="0.75", lw=0.8, zorder=0)

    if search_radius_arcmin is None:
        search_radius_arcmin = float(np.nanmax(rows["theta_arcmin"])) if len(rows) else 1.0
    circle = plt.Circle(
        (0, 0),
        search_radius_arcmin,
        fill=False,
        color="0.55",
        lw=1.0,
        ls="--",
        label=f"{search_radius_arcmin:g} arcmin search radius",
    )
    ax.add_patch(circle)

    if rows.empty:
        ax.scatter([0], [0], marker="+", s=160, color="black", label="FRB")
        ax.text(0.5, 0.5, "No foreground candidates", transform=ax.transAxes, ha="center")
    else:
        b = rows["b_kpc"].astype(float)
        marker_size = 60 + 200 * (1 - b.rank(pct=True, method="average"))
        scatter = ax.scatter(
            rows["dra_arcmin"],
            rows["ddec_arcmin"],
            c=rows["z"].astype(float),
            s=marker_size,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.88,
            label="Foreground candidate",
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Candidate redshift")

        label_rows = rows.sort_values("rank_key").head(label_top_n)
        for _, row in label_rows.iterrows():
            name = str(row.get("name") or row.get("id") or "").strip()
            if not name or name.lower() == "nan" or name == "---":
                name = "candidate"
            label = f"{name}\nz={row['z']:.3f}, b={row['b_kpc']:.0f} kpc"
            ax.annotate(
                label,
                (row["dra_arcmin"], row["ddec_arcmin"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.scatter([0], [0], marker="+", s=180, color="crimson", lw=2, label="FRB")
    limit = max(search_radius_arcmin * 1.08, 1.0)
    ax.set_xlim(limit, -limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta$RA cos(Dec) [arcmin]; east is left")
    ax.set_ylabel(r"$\Delta$Dec [arcmin]")
    ax.text(
        0.02,
        0.02,
        r"Labels show projected $b$ at candidate redshift",
        transform=ax.transAxes,
        fontsize=8,
        color="0.35",
        ha="left",
        va="bottom",
    )
    frb_z = (
        rows["frb_z"].dropna().iloc[0] if "frb_z" in rows and rows["frb_z"].notna().any() else None
    )
    z_text = f", z_FRB={frb_z:.4g}" if frb_z is not None else ""
    ax.set_title(f"{frb_name}: foreground candidates{z_text}")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_candidate_summary(
    rows: pd.DataFrame,
    *,
    output: str | Path,
) -> Path:
    """Plot a compact multi-FRB summary of foreground-candidate counts and distances."""
    plt = _import_matplotlib()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = rows.groupby("frb_name").size().sort_values(ascending=False)
    nearest = rows.groupby("frb_name")["b_kpc"].min().reindex(counts.index)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    axes[0].bar(range(len(counts)), counts.values, color="#4c78a8")
    axes[0].set_xticks(range(len(counts)), counts.index, rotation=60, ha="right")
    axes[0].set_ylabel("Foreground candidates")
    axes[0].set_title("Foreground count by FRB")

    axes[1].bar(range(len(nearest)), nearest.values, color="#f58518")
    axes[1].set_xticks(range(len(nearest)), nearest.index, rotation=60, ha="right")
    axes[1].set_ylabel("Nearest candidate impact parameter [kpc]")
    axes[1].set_title("Nearest foreground candidate")

    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_all_candidate_sky_maps(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    output_dir: str | Path,
    search_radius_arcmin: float | None = None,
    label_top_n: int = 5,
) -> list[Path]:
    """Generate one sky map per FRB plus an overview summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = add_offsets(candidates, targets)
    outputs: list[Path] = []
    for frb_name, group in rows.groupby("frb_name", sort=True):
        outputs.append(
            plot_candidate_sky_map(
                group.sort_values("rank_key"),
                frb_name=frb_name,
                output=output_dir / f"{safe_slug(frb_name)}_foreground_candidates.png",
                search_radius_arcmin=search_radius_arcmin,
                label_top_n=label_top_n,
            )
        )
    if not rows.empty:
        outputs.append(
            plot_candidate_summary(rows, output=output_dir / "foreground_candidate_summary.png")
        )
    return outputs
