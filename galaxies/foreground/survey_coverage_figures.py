#!/usr/bin/env python3
"""Coverage figures: matrix heatmap and per-survey exact MOC footprint maps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TARGETS
from .survey_footprint_mocs import (
    ALL_SKY_SURVEYS,
    DEFAULT_CACHE_DIR,
    load_survey_moc,
    moc_sky_area_deg2,
    rasterize_moc,
    survey_display_names,
)
from .utils import parse_coord

STATUS_ORDER = ["no_footprint", "footprint_empty", "catalog_hits", "foreground"]
STATUS_LABELS = {
    "no_footprint": "outside footprint",
    "footprint_empty": "in footprint, 0 cone hits",
    "catalog_hits": "catalog hits (no foreground pass)",
    "foreground": "foreground match",
}
STATUS_COLORS = {
    "no_footprint": "#bdbdbd",
    "footprint_empty": "#fff3e0",
    "catalog_hits": "#90caf9",
    "foreground": "#2e7d32",
}

SURVEY_PANEL_STYLE: dict[str, dict[str, str]] = {
    "NED": {"cmap": "Reds", "note": "NED TAP (all-sky)"},
    "GLADE+": {"cmap": "YlOrBr", "note": "Vizier VII/291/gladep MOC"},
    "DESI_DR8_NORTH": {"cmap": "Blues", "note": "Vizier VII/292/north MOC"},
    "SDSS_DR12": {"cmap": "Greens", "note": "Vizier V/147/sdss12 MOC"},
    "CLUSTERS": {"cmap": "Purples", "note": "PSZ2+MCXC+MCXC-II (all-sky)"},
}


def make_coverage_matrix(df: pd.DataFrame) -> plt.Figure:
    nicknames = sorted(df["nickname"].unique(), key=str.lower)
    surveys = list(df.groupby("survey").size().sort_values(ascending=False).index)
    z = np.full((len(nicknames), len(surveys)), -1.0)
    labels = [[""] * len(surveys) for _ in nicknames]

    for i, nick in enumerate(nicknames):
        sub = df[df["nickname"] == nick].set_index("survey")
        for j, survey in enumerate(surveys):
            if survey not in sub.index:
                continue
            row = sub.loc[survey]
            status = row["status"]
            z[i, j] = STATUS_ORDER.index(status)
            labels[i][j] = str(int(row.get("raw_count", 0)))

    fig, ax = plt.subplots(figsize=(max(8, len(surveys) * 1.1), max(5, len(nicknames) * 0.45)))
    cmap = matplotlib.colors.ListedColormap([STATUS_COLORS[s] for s in STATUS_ORDER])
    ax.imshow(z, aspect="auto", cmap=cmap, vmin=0, vmax=len(STATUS_ORDER) - 1)

    ax.set_xticks(range(len(surveys)))
    ax.set_xticklabels(surveys, rotation=45, ha="right")
    ax.set_yticks(range(len(nicknames)))
    ax.set_yticklabels(nicknames)
    ax.set_title("Foreground search: survey coverage per sightline\n(cell = raw cone hit count)")

    for i in range(len(nicknames)):
        for j in range(len(surveys)):
            if z[i, j] < 0:
                continue
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color="black")

    handles = [
        matplotlib.patches.Patch(color=STATUS_COLORS[s], label=STATUS_LABELS[s])
        for s in STATUS_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    return fig


def _sightline_marker_colors(df: pd.DataFrame) -> dict[str, str]:
    rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    colors: dict[str, str] = {}
    for nick, sub in df.groupby("nickname"):
        best = max(sub["status"], key=lambda s: rank.get(s, -1))
        colors[str(nick)] = STATUS_COLORS.get(best, "#333333")
    return colors


def _plot_survey_moc_panel(
    fig: plt.Figure,
    sub: tuple[int, int, int],
    survey: str,
    moc,
    hmap: np.ndarray,
    coords: list[tuple[str, object]],
    colors: dict[str, str],
) -> None:
    style = SURVEY_PANEL_STYLE[survey]
    hp.mollview(
        hmap,
        fig=fig,
        sub=sub,
        coord="C",
        cmap=style["cmap"],
        min=0,
        max=1,
        cbar=False,
        notext=True,
        title=survey.replace("_", " "),
    )
    ras = [c.ra.deg for _, c in coords]
    decs = [c.dec.deg for _, c in coords]
    hp.projscatter(
        ras,
        decs,
        lonlat=True,
        marker="o",
        s=28,
        c=[colors.get(n, "#d32f2f") for n, _ in coords],
        edgecolors="black",
        linewidths=0.4,
        zorder=10,
    )
    area = moc_sky_area_deg2(moc)
    subtitle = style["note"] if survey in ALL_SKY_SURVEYS else f"{style['note']}, {area:.0f} deg2"
    # healpy creates axes in sub-plot order; last axis matches this sub index.
    fig.axes[sub[2] - 1].set_xlabel(subtitle, fontsize=7)


def _draw_strip_inset(
    fig: plt.Figure,
    coords: list[tuple[str, object]],
    colors: dict[str, str],
) -> None:
    ax = fig.add_axes([0.08, 0.04, 0.84, 0.16])
    texts = []
    for name, coord in coords:
        ra, dec = coord.ra.deg, coord.dec.deg
        ax.scatter(
            ra,
            dec,
            s=80,
            c=colors.get(name, "#d32f2f"),
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        texts.append(ax.text(ra, dec, name, fontsize=8, fontweight="bold", zorder=6))
    ax.set_xlim(0, 360)
    ax.set_ylim(68.5, 75.5)
    ax.set_xticks(range(0, 361, 60))
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("All 12 co-detection sightlines (Dec 69-75 deg strip)", fontsize=10)
    ax.grid(True, alpha=0.2, lw=0.5)
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            ax=ax,
            expand=(1.25, 1.7),
            arrowprops=dict(arrowstyle="-", color="#616161", lw=0.5),
        )
    except ImportError:
        for t in texts:
            x, y = t.get_position()
            t.set_position((x + 3, y + 0.15))


def make_allsky_coverage_map(
    df: pd.DataFrame,
    targets: list[tuple[str, str, str, float]] = TARGETS,
    *,
    moc_cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> plt.Figure:
    """Five-panel exact Vizier MOC footprints + sightline strip inset."""
    colors = _sightline_marker_colors(df)
    coords = [(name, parse_coord(ra, dec)) for name, ra, dec, _ in targets]
    moc_cache_dir = Path(moc_cache_dir)

    fig = plt.figure(figsize=(16, 11))
    for idx, survey in enumerate(survey_display_names(), start=1):
        moc = load_survey_moc(survey, moc_cache_dir)
        hmap = rasterize_moc(moc)
        _plot_survey_moc_panel(fig, (2, 3, idx), survey, moc, hmap, coords, colors)

    _draw_strip_inset(fig, coords, colors)

    foot = [
        mpatches.Patch(color=STATUS_COLORS["foreground"], label="foreground match"),
        mpatches.Patch(color=STATUS_COLORS["catalog_hits"], label="catalog hits only"),
        mpatches.Patch(color=STATUS_COLORS["footprint_empty"], label="footprint, 0 hits"),
    ]
    fig.legend(handles=foot, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=9)
    fig.suptitle(
        "Exact survey footprints (CDS/Vizier MOC) and co-detection sightlines",
        fontsize=13,
        y=0.995,
    )
    return fig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--coverage-csv",
        default="scratch/repro-foreground-search-hpcc/survey_coverage.csv",
        help="survey_coverage.csv from run_search",
    )
    ap.add_argument("--out-dir", default="scratch/repro-foreground-coverage-figures")
    ap.add_argument(
        "--moc-cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="cache directory for Vizier MOC FITS files",
    )
    ap.add_argument(
        "--prefetch-mocs",
        action="store_true",
        help="fetch/cache all survey MOCs before plotting (needs network once)",
    )
    args = ap.parse_args(argv)

    if args.prefetch_mocs:
        from .survey_footprint_mocs import prefetch_all_survey_mocs

        prefetch_all_survey_mocs(args.moc_cache_dir)

    df = pd.read_csv(args.coverage_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    fig = make_coverage_matrix(df)
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(args.out_dir, f"survey_coverage_matrix.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    sky = make_allsky_coverage_map(df, moc_cache_dir=args.moc_cache_dir)
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(args.out_dir, f"survey_coverage_allsky.{ext}")
        sky.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {path}")
    plt.close(sky)

    manifest = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "figures": [
            {
                "file": "survey_coverage_matrix.png",
                "expectation": "Matrix: rows=sightlines, cols=surveys; color=footprint status; "
                "cell number=raw cone hit count.",
            },
            {
                "file": "survey_coverage_allsky.png",
                "expectation": "Five Mollweide panels (NED, GLADE+, DESI DR8 North, SDSS DR12, "
                "CLUSTERS) showing exact CDS/Vizier MOC footprints; sightlines overlaid; "
                "bottom strip zoom Dec 69-75 deg with labels.",
            },
        ],
    }
    man_path = os.path.join(args.out_dir, "figures.manifest.json")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {man_path}")


if __name__ == "__main__":
    main()
