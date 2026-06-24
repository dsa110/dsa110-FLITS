import os
import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from flits.plotting import use_flits_style

from .config import CLUSTER_M500_TO_M200, TARGETS
from .scattering_predict import r_delta_kpc
from .utils import get_angular_radius, parse_coord

# Mirror of search._CLUSTER_RE / CLASSIFICATION_COLUMNS, kept local so this plotting
# util need not import the astroquery-heavy search module. Keep in sync with search.py.
_CLUSTER_RE = re.compile(r"cluster|gclstr|clg|^cl\b", re.IGNORECASE)
_CLASSIFICATION_COLUMNS = ("classification", "Type", "class", "cl", "otype")


def _split_galaxies_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (galaxies, clusters) by object classification.

    Clusters are NED 'GClstr', SDSS/SIMBAD 'ClG', bare 'Cl', or 'cluster' in any
    classification column. Everything else (incl. blank/untyped) is a galaxy.
    """
    is_cluster = pd.Series(False, index=df.index)
    for col in _CLASSIFICATION_COLUMNS:
        if col in df.columns:
            is_cluster = is_cluster | df[col].astype(str).str.contains(_CLUSTER_RE)
    return df[~is_cluster].copy(), df[is_cluster].copy()


def plot_impact_vs_redshift(
    summary_df: pd.DataFrame, all_galaxies_df: pd.DataFrame, output_path: str | None = None
):
    """
    Plot impact parameter vs redshift for all identified foreground galaxies.
    """
    use_flits_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot galaxies
    # Use target_name for coloring if available, otherwise name (which might be the same) or target_id
    if "target_name" in all_galaxies_df.columns:
        color_col = "target_name"
    elif "name" in all_galaxies_df.columns:
        color_col = "name"
    else:
        color_col = "target_id"

    # Create a mapping for categorical colors if using names
    if color_col in ["target_name", "name"]:
        names = all_galaxies_df[color_col].unique()
        name_to_id = {name: i for i, name in enumerate(names)}
        colors = all_galaxies_df[color_col].map(name_to_id)
    else:
        colors = all_galaxies_df["target_id"]

    scatter = ax.scatter(
        all_galaxies_df["z"],
        all_galaxies_df["impact_kpc"],
        c=colors,
        cmap="tab20",
        s=100,
        edgecolor="k",
        alpha=0.8,
        label="Foreground Galaxies",
    )

    # Plot FRBs as vertical lines or markers at their redshifts
    for _, row in summary_df.iterrows():
        ax.axvline(row["z_frb"], color="gray", linestyle="--", alpha=0.3)
        # Label FRB at the top
        name = row.get("name", f"Target {row['target_id']}")
        ax.text(
            row["z_frb"],
            ax.get_ylim()[1],
            name,
            rotation=90,
            verticalalignment="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Redshift ($z$)")
    ax.set_ylabel("Impact Parameter ($b$ [kpc])")
    # ax.set_title('Foreground Galaxy Environment')

    # Add colorbar or legend
    if color_col == "name":
        # Legend is better for names
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=name,
                markerfacecolor=plt.cm.tab20(name_to_id[name] / 20),
                markersize=10,
            )
            for name in names
        ]
        ax.legend(
            handles=legend_elements, title="FRB Field", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
    else:
        cbar = plt.colorbar(scatter)
        cbar.set_label("Target ID")

    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_sightline(target_info: dict, galaxies_df: pd.DataFrame, output_path: str | None = None):
    """On-sky map of foreground galaxies AND galaxy clusters along one FRB sightline.

    FRB at the origin; galaxies coloured by redshift; clusters drawn as distinct
    crimson diamonds with a dashed R200 circle when a catalog M500 is available
    (most NED 'GClstr' rows carry no mass, so they appear as unsized markers).
    Concentric rings mark physical impact parameters at the mean foreground z; the
    frame auto-scales to enclose the outermost object so far clusters stay visible.
    """
    use_flits_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    target_coord = parse_coord(target_info["ra"], target_info["dec"])
    ra0, dec0 = target_coord.ra.deg, target_coord.dec.deg
    target_name = target_info.get("name", f"Target {target_info.get('target_id', 'Unknown')}")
    cos_dec = np.cos(np.radians(dec0))

    if galaxies_df.empty:
        gals, clusters = galaxies_df, galaxies_df
    else:
        gals, clusters = _split_galaxies_clusters(galaxies_df)

    # FRB at center
    ax.scatter(
        0, 0, marker="*", s=400, color="red", edgecolor="k", label=f"FRB {target_name}", zorder=10
    )

    offsets = []  # (dra, ddec) arcmin of every plotted object, for autoscaling
    texts = []  # label Text objects, position-deconflicted by adjustText at the end

    if not gals.empty:
        dra = (gals["ra"] - ra0) * 60.0 * cos_dec
        ddec = (gals["dec"] - dec0) * 60.0
        offsets += list(zip(dra, ddec))
        scatter = ax.scatter(
            dra,
            ddec,
            c=gals["z"],
            s=150,
            cmap="viridis",
            edgecolor="k",
            alpha=0.85,
            label="Foreground Galaxies",
            zorder=6,
        )
        for (_, row), x, y in zip(gals.iterrows(), dra, ddec):
            label = (
                row["name"]
                if pd.notna(row.get("name")) and row.get("name") != ""
                else f"z={row['z']:.3f}"
            )
            texts.append(ax.text(x, y, label, fontsize=8, fontweight="bold", ha="center", va="top"))
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Redshift ($z$)")

    if not clusters.empty:
        cdra = (clusters["ra"] - ra0) * 60.0 * cos_dec
        cddec = (clusters["dec"] - dec0) * 60.0
        offsets += list(zip(cdra, cddec))
        m500 = (
            pd.to_numeric(clusters["m500_msun"], errors="coerce")
            if "m500_msun" in clusters.columns
            else pd.Series(np.nan, index=clusters.index)
        )
        ax.scatter(
            cdra,
            cddec,
            marker="D",
            s=180,
            facecolor="none",
            edgecolor="crimson",
            linewidths=2.0,
            label="Foreground Clusters",
            zorder=7,
        )
        for (idx, row), x, y in zip(clusters.iterrows(), cdra, cddec):
            mass = m500.loc[idx]
            # Dashed R200 ring only when a catalog mass exists (R200 from M200 =
            # CLUSTER_M500_TO_M200 * M500); unmassed NED clusters stay unsized markers.
            if pd.notna(mass) and mass > 0:
                r200 = r_delta_kpc(CLUSTER_M500_TO_M200 * float(mass), float(row["z"]), 200.0)
                theta = get_angular_radius(float(row["z"]), r200).to(u.arcmin).value
                ax.add_artist(
                    plt.Circle(
                        (x, y),
                        theta,
                        color="crimson",
                        fill=False,
                        linestyle="--",
                        alpha=0.6,
                        zorder=4,
                    )
                )
            label = (
                row["name"]
                if pd.notna(row.get("name")) and row.get("name") != ""
                else f"cluster z={row['z']:.3f}"
            )
            texts.append(
                ax.text(
                    x,
                    y,
                    label,
                    fontsize=8,
                    color="crimson",
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )
            )

    # Representative redshift for the physical-impact rings.
    if not galaxies_df.empty:
        avg_z = float(pd.to_numeric(galaxies_df["z"], errors="coerce").mean())
    else:
        avg_z = target_info["z_frb"] / 2.0

    # Auto-scale to enclose every object (+margin), but never tighter than ~550 kpc.
    base_limit = get_angular_radius(avg_z, 550).to(u.arcmin).value
    max_off = max((float(np.hypot(a, b)) for a, b in offsets), default=0.0)
    limit = max(base_limit, max_off * 1.18, 1.0)

    if galaxies_df.empty:
        ax.text(
            0,
            0.85 * limit,
            "no catalogued foreground systems",
            color="gray",
            fontsize=10,
            ha="center",
            va="center",
            style="italic",
        )

    # Impact rings: 100 kpc .. 5 Mpc, drawn only where they fit the frame.
    for b_kpc in [100, 250, 500, 1000, 2000, 5000]:
        theta = get_angular_radius(avg_z, b_kpc).to(u.arcmin).value
        if theta > limit:
            continue
        ax.add_artist(plt.Circle((0, 0), theta, color="gray", fill=False, linestyle=":", alpha=0.5))
        label = f"{b_kpc} kpc" if b_kpc < 1000 else f"{b_kpc // 1000} Mpc"
        ax.text(0, theta, label, color="gray", fontsize=8, ha="center", va="bottom")

    ax.set_xlabel(r"$\Delta$ RA [arcmin]")
    ax.set_ylabel(r"$\Delta$ Dec [arcmin]")
    ax.set_title(f"Sightline Environment: {target_name} ($z={target_info['z_frb']}$)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlim(limit, -limit)  # RA increases to the left
    ax.set_ylim(-limit, limit)

    # Deconflict crowded labels (e.g. phineas' 9 clusters) after limits are fixed.
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    return fig, ax


def main(argv: list | None = None) -> int:
    """Write a per-sightline on-sky galaxy+cluster map for every target.

    Reads ``{results-dir}/{name}_galaxies.csv`` (empty/absent → a clean-sightline
    map) and writes ``{output-dir}/{name}_sky.png`` for all config.TARGETS.
    """
    import argparse

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Per-sightline on-sky galaxy+cluster maps.")
    parser.add_argument("--results-dir", default=os.path.join(base, "results"))
    parser.add_argument("--output-dir", default=os.path.join(base, "results"))
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    written = 0
    for name, ra_str, dec_str, z_frb in TARGETS:
        csv_path = os.path.join(args.results_dir, f"{name.lower()}_galaxies.csv")
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        info = {"name": name, "ra": ra_str, "dec": dec_str, "z_frb": z_frb}
        out_path = os.path.join(args.output_dir, f"{name.lower()}_sky.png")
        fig, _ = plot_sightline(info, df, output_path=out_path)
        plt.close(fig)
        g, c = _split_galaxies_clusters(df) if not df.empty else (df, df)
        print(f"  {name}: {len(g)} galaxies, {len(c)} clusters -> {out_path}")
        written += 1
    print(f"Wrote {written} sky maps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
