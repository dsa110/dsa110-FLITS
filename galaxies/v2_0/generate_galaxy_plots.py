#!/usr/bin/env python3
"""Generate per-target sightline and NFW mass-profile figures.

Outputs:
  results/{name}_sightline.png
  results/{name}_mass_profile.png
  results/galaxy_sightlines_report.html  (self-contained, base64-embedded)
  docs/index.html                        (copy for GitHub Pages)
"""

import base64
import io
import os
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.optimize import brentq

# --------------- styling ---------------
DARK_BLUE = "#1B365D"
LIGHT_BLUE = "#4A90E2"
ACCENT_ORANGE = "#F5A623"
ACCENT_RED = "#D0021B"
TEXT_DARK = "#333333"
GRID_COLOR = "#E5E5E5"
BG_LIGHT = "#FAFBFC"

# --------------- physics helpers ---------------

# Moster+2013 (MNRAS 428, 3121), Eq. 2 + Table 1, z=0 free parameters.
# M*/M_h = 2 N [ (M_h/M1)^-beta + (M_h/M1)^gamma ]^-1
_MOSTER_LOG_M1 = 11.590
_MOSTER_N = 0.0351
_MOSTER_BETA = 1.376
_MOSTER_GAMMA = 0.608


def _moster_log_mstar(log_mh):
    """log10 M* predicted by Moster+2013 Eq. 2 for a given log10 M_halo (both Msun)."""
    x = 10 ** (log_mh - _MOSTER_LOG_M1)  # M_h / M1
    ratio = 2.0 * _MOSTER_N / (x ** (-_MOSTER_BETA) + x**_MOSTER_GAMMA)
    return log_mh + np.log10(ratio)


def estimate_halo_mass(log_mstar):
    """Halo mass from stellar mass via the Moster+2013 (Eq. 2) SMHM relation.

    M*(M_h) is monotonic in M_h over the plotted range, so we invert Eq. 2
    numerically with brentq rather than using a sinh fitting form. Bracket
    spans dwarf to cluster halos (log M_h in [9.5, 15.5]).
    """
    # Clamp into the SMHM-invertible window so a degenerate estimate (e.g. a z~0.001
    # contaminant giving log M*~5.5) pins at the bracket edge instead of crashing
    # brentq with a same-sign bracket; the 1e-6 inset keeps a strict sign change.
    lo, hi = _moster_log_mstar(9.5), _moster_log_mstar(15.5)
    log_mstar = min(max(log_mstar, lo + 1e-6), hi - 1e-6)
    log_mh = brentq(lambda lmh: _moster_log_mstar(lmh) - log_mstar, 9.5, 15.5)
    return 10**log_mh


def get_rvir_and_rs(m_halo_msun, z):
    """R_200 and NFW scale radius r_s in kpc."""
    M_h = m_halo_msun * u.Msun
    H_z = cosmo.H(z)
    G = const.G
    r_vir = ((G * M_h / (100 * H_z**2)) ** (1 / 3)).to(u.kpc).value
    # Dutton & Maccio 2014 (MNRAS 441, 3359) c-M relation at z=0.
    log_c = 0.905 - 0.101 * np.log10(m_halo_msun / 1e12)
    # First-order redshift evolution (concentration declines with z); D&M14
    # give d(log c)/dz ~ -0.08 over the low-z regime probed here.
    log_c *= 1.0 - 0.08 * z
    c = 10**log_c
    r_s = r_vir / c
    return r_vir, r_s, c


def estimate_logmstar_from_photometry(row, z_gal):
    """Taylor+2011 (MNRAS 418, 1587, Eq. 8) color-mass estimate.

        log(M*/Msun) = 1.15 + 0.70 (g - i) - 0.4 M_i

    DESI Legacy imaging provides g, r, z; the Taylor calibration is in g-i and
    M_i, so this only fires when an i-band magnitude is also present. Returns
    None when the required bands are missing (caller falls back to 'assumed').
    """
    g = _get_mag(row, "g")
    i = _get_mag(row, "i")
    if g is None or i is None:
        return None
    dist_mod = cosmo.distmod(z_gal).value
    M_i = i - dist_mod  # absolute i-band (rest-frame k-correction neglected)
    return 1.15 + 0.70 * (g - i) - 0.4 * M_i


def _get_mag(row, band):
    """Return an apparent magnitude for ``band`` from common column spellings, or None."""
    for col in (band, f"{band}mag", f"mag_{band}", f"{band}_mag", f"{band}MAG"):
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return None


def nfw_enclosed_mass(r, m_halo, r_vir, r_s, c):
    x = r / r_s
    f_x = np.log(1 + x) - x / (1 + x)
    f_c = np.log(1 + c) - c / (1 + c)
    return m_halo * f_x / f_c


# --------------- per-target figure generators ---------------


def make_sightline_fig(target_name, z_frb, gal_rows):
    """Return (fig, b64_png) for the sightline spatial intersection plot."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150, facecolor=BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)

    # FRB sightline
    ax.axhline(0, color=DARK_BLUE, linewidth=2.5, zorder=1)

    # Comoving distance to the FRB itself
    d_frb = cosmo.comoving_distance(z_frb).to(u.Mpc).value

    # Data-bracketed x-limits: zoom onto where the galaxies actually sit,
    # not the full 0->d_frb path (most foreground galaxies cluster well short
    # of the FRB, so the old 0..1.1*d_frb window left them crowded at the edge).
    d_coms = [g["d_com"] for g in gal_rows]
    d_min = min(d_coms)
    x_range = max(d_frb - d_min, 1.0)  # guard against a single galaxy at ~d_frb
    x_min = d_min - 0.15 * x_range
    x_max = d_frb + 0.05 * x_range
    x_span = x_max - x_min
    halo_half_w = 0.015 * x_span  # halo shading width scales with the view, not a fixed ±6 Mpc

    # z-ordered continuous colormap: galaxy colour encodes redshift.
    z_vals = [g["z_gal"] for g in gal_rows]
    z_lo, z_hi = min(z_vals), max(z_vals)
    if z_hi <= z_lo:
        z_lo, z_hi = z_lo - 0.01, z_hi + 0.01
    norm = Normalize(vmin=z_lo, vmax=z_hi)
    cmap = plt.cm.plasma

    texts = []
    for idx, g in enumerate(gal_rows):
        c = cmap(norm(g["z_gal"]))
        # Virial halo shading (width proportional to the data x-range)
        ax.fill_between(
            [g["d_com"] - halo_half_w, g["d_com"] + halo_half_w],
            [g["impact"] - g["r_vir"]] * 2,
            [g["impact"] + g["r_vir"]] * 2,
            color=c,
            alpha=0.18,
            zorder=2,
        )
        # Galaxy centre
        ax.scatter(
            g["d_com"], g["impact"], color=c, edgecolors="white", s=100, zorder=5, linewidths=1.2
        )
        # Impact parameter connector
        ax.plot(
            [g["d_com"], g["d_com"]],
            [0, g["impact"]],
            color=ACCENT_ORANGE,
            linestyle="--",
            linewidth=1.5,
            zorder=3,
        )
        # Annotate the R_200 extent next to the first shaded halo
        if idx == 0:
            ax.annotate(
                r"$R_{200}$",
                xy=(g["d_com"] + halo_half_w, g["impact"] + g["r_vir"]),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=8,
                color=DARK_BLUE,
                fontweight="bold",
                zorder=6,
            )
        # Label (positions deconflicted below via adjustText)
        label = f"z={g['z_gal']:.4f}  b={g['impact']:.0f} kpc"
        texts.append(
            ax.text(
                g["d_com"],
                g["impact"],
                label,
                fontsize=8.5,
                color=TEXT_DARK,
                fontweight="bold",
                zorder=6,
            )
        )

    # Mark FRB endpoint
    ax.axvline(d_frb, color=ACCENT_RED, linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        d_frb,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 50 else 50,
        f"FRB z={z_frb:.3f}",
        fontsize=8,
        color=ACCENT_RED,
        ha="right",
    )

    ax.set_xlabel("Comoving Distance (Mpc)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel("Impact Parameter  b  (kpc)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_title(
        f"{target_name} — Intervening Galaxies & $R_{{200}}$ Halos",
        fontsize=13,
        fontweight="bold",
        color=DARK_BLUE,
        pad=12,
    )
    ax.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.7)
    ax.set_xlim(x_min, x_max)
    y_max = max((g["impact"] + g["r_vir"] for g in gal_rows), default=100) * 1.3
    ax.set_ylim(-30, max(y_max, 100))

    # Resolve label collisions
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

    # Redshift colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Redshift  z", fontsize=8, color=TEXT_DARK)
    cbar.ax.tick_params(labelsize=7)

    legend_elements = [
        plt.Line2D([0], [0], color=DARK_BLUE, linewidth=2, label="FRB Sightline"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DARK_BLUE,
            markeredgecolor="white",
            markersize=10,
            label="Galaxy Centre",
        ),
        plt.Line2D(
            [0],
            [0],
            color=ACCENT_ORANGE,
            linestyle="--",
            linewidth=1.5,
            label="Impact Parameter (b)",
        ),
        Patch(facecolor=LIGHT_BLUE, edgecolor="none", alpha=0.3, label="Virial Radius ($R_{200}$)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor=GRID_COLOR,
        fontsize=8,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return fig, b64


def make_mass_profile_fig(target_name, z_frb, gal_rows):
    """Return (fig, b64_png) for the NFW enclosed-mass profile."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150, facecolor=BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)

    # Extend the radial grid (and x-limit) to enclose the largest halo so big
    # halos are not clipped at the old fixed 350 kpc edge.
    max_rvir = max((g["r_vir"] for g in gal_rows), default=350)
    x_upper = max(max_rvir * 1.3, 350)
    r_arr = np.linspace(0.1, x_upper, 500)

    # z-ordered continuous colormap matching the sightline figure.
    z_vals = [g["z_gal"] for g in gal_rows]
    z_lo, z_hi = min(z_vals), max(z_vals)
    if z_hi <= z_lo:
        z_lo, z_hi = z_lo - 0.01, z_hi + 0.01
    norm = Normalize(vmin=z_lo, vmax=z_hi)
    cmap = plt.cm.plasma

    for idx, g in enumerate(gal_rows):
        c = cmap(norm(g["z_gal"]))
        m_enc = nfw_enclosed_mass(r_arr, g["m_halo"], g["r_vir"], g["r_s"], g["c"])
        label = f"z={g['z_gal']:.4f}  log M★={g['log_mstar']:.1f}"
        ax.plot(r_arr, m_enc / 1e11, color=c, linewidth=2.2, label=label)
        # Impact parameter marker
        m_at_b = nfw_enclosed_mass(g["impact"], g["m_halo"], g["r_vir"], g["r_s"], g["c"])
        ax.axvline(g["impact"], color=c, linestyle=":", alpha=0.5)
        ax.scatter(
            g["impact"], m_at_b / 1e11, color=c, edgecolor="white", s=60, zorder=5, linewidths=1.2
        )
        # R_vir marker
        ax.axvline(g["r_vir"], color=c, linestyle="--", alpha=0.3)

    ax.set_xlabel("Physical Radius  r  (kpc)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel(
        r"Enclosed NFW Mass  $M(<r)$  ($10^{11}\,M_{\odot}$)",
        fontsize=11,
        fontweight="bold",
        color=TEXT_DARK,
    )
    ax.set_title(
        f"{target_name} — Enclosed DM Halo Mass Profiles",
        fontsize=13,
        fontweight="bold",
        color=DARK_BLUE,
        pad=12,
    )
    ax.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.7)
    ax.set_xlim(0, x_upper)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor=GRID_COLOR,
        fontsize=8.5,
        ncol=1,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return fig, b64


# --------------- HTML builder ---------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FRB Sightline Galaxy Intersection Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg-dark: #0f172a; --card-bg: #1e293b; --text-light: #f8fafc;
  --text-muted: #94a3b8; --accent-blue: #3b82f6; --accent-orange: #f5a623;
  --accent-red: #ef4444; --accent-green: #10b981;
  --border-color: rgba(255,255,255,0.06);
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:var(--bg-dark);color:var(--text-light);padding:2rem;line-height:1.6}
.container{max-width:1400px;margin:0 auto}
header{margin-bottom:2rem;border-bottom:1px solid var(--border-color);padding-bottom:1.2rem}
h1{font-size:2.2rem;font-weight:700;letter-spacing:-0.04em;
   background:linear-gradient(135deg,#3b82f6,#60a5fa);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.4rem}
.subtitle{color:var(--text-muted);font-size:1.05rem}

/* tabs */
.tabs{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:1.5rem}
.tab-btn{padding:.5rem 1.1rem;border:1px solid var(--border-color);border-radius:8px;
  background:transparent;color:var(--text-muted);cursor:pointer;font-family:'Inter';
  font-size:.88rem;font-weight:600;transition:all .2s}
.tab-btn:hover,.tab-btn.active{background:var(--accent-blue);color:#fff;border-color:var(--accent-blue)}
.tab-panel{display:none}
.tab-panel.active{display:block}

.card{background:var(--card-bg);border-radius:12px;border:1px solid var(--border-color);
  padding:1.5rem;box-shadow:0 4px 20px rgba(0,0,0,.15);margin-bottom:1.5rem}
.card-title{font-size:1.2rem;font-weight:600;margin-bottom:1rem;color:var(--text-light);
  border-left:4px solid var(--accent-blue);padding-left:.75rem}
.chart-container{width:100%;border-radius:8px;overflow:hidden;background:#fff;padding:.5rem}
.chart-container img{width:100%;height:auto;display:block}
.info-panel{margin-top:1rem;background:rgba(59,130,246,.05);border-radius:8px;
  border:1px solid rgba(59,130,246,.15);padding:.8rem;font-size:.88rem;color:#93c5fd}

table{width:100%;border-collapse:collapse;text-align:left;margin-top:.8rem;font-size:.92rem}
th,td{padding:.6rem .9rem;border-bottom:1px solid var(--border-color)}
th{color:var(--text-muted);font-weight:600;text-transform:uppercase;font-size:.78rem;letter-spacing:.04em}
tr:hover{background:rgba(255,255,255,.02)}
.badge{display:inline-block;padding:.2rem .55rem;border-radius:50px;font-size:.72rem;
  font-weight:600;text-transform:uppercase;letter-spacing:.02em}
.badge-yes{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3)}
.badge-no{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3)}
.badge-catalog{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3)}
.badge-photometric{background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid rgba(59,130,246,.3)}
.badge-assumed{background:rgba(245,166,35,.15);color:#fbbf24;border:1px solid rgba(245,166,35,.3)}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
@media(max-width:1024px){.grid-2{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<header>
  <h1>FRB Sightline Galaxy Intersection Dashboard</h1>
  <p class="subtitle">Per-target intervening galaxies &amp; virial halo calculations &mdash; __N_SIGHTLINES__ northern-cap sightlines</p>
</header>
<div class="tabs" id="tabs"></div>
<div id="panels"></div>
</div>
<script>
document.addEventListener('DOMContentLoaded',()=>{
  const btns=document.querySelectorAll('.tab-btn');
  const panels=document.querySelectorAll('.tab-panel');
  btns.forEach(b=>b.addEventListener('click',()=>{
    btns.forEach(x=>x.classList.remove('active'));
    panels.forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById(b.dataset.target).classList.add('active');
  }));
  if(btns.length) btns[0].click();
});
</script>
</body></html>
"""


def build_html(target_sections):
    """Return complete HTML string with embedded per-target tabs."""
    tabs_html = ""
    panels_html = ""

    for i, sec in enumerate(target_sections):
        tid = f"panel-{sec['name'].lower()}"
        active = " active" if i == 0 else ""
        tabs_html += f'<button class="tab-btn{active}" data-target="{tid}">{sec["name"]}</button>\n'

        # table rows
        trows = ""
        for g in sec["gal_rows"]:
            intersects = g["impact"] <= g["r_vir"]
            intersect = "Yes" if intersects else "No"
            badge = "badge-yes" if intersects else "badge-no"
            x_val = g["impact"] / g["r_s"]
            f_x = np.log(1 + x_val) - x_val / (1 + x_val)
            f_c = np.log(1 + g["c"]) - g["c"] / (1 + g["c"])
            mass_frac = f_x / f_c
            src = g.get("mass_source", "assumed")
            src_badge = (
                f"badge-{src}" if src in ("catalog", "photometric", "assumed") else "badge-assumed"
            )
            trows += f"""<tr>
              <td style="font-weight:bold;color:#f1f5f9">z={g["z_gal"]:.4f}</td>
              <td>{sec["z_frb"]:.3f}</td>
              <td>{g["d_com"]:.1f} Mpc</td>
              <td style="font-weight:bold;color:#f5a623">{g["impact"]:.1f} kpc</td>
              <td>{g["r_vir"]:.1f} kpc</td>
              <td>log M★={g["log_mstar"]:.2f} <span class="badge {src_badge}">{src}</span></td>
              <td>10<sup>{np.log10(g["m_halo"]):.2f}</sup> M<sub>☉</sub></td>
              <td>{mass_frac * 100:.1f}%</td>
              <td><span class="badge {badge}">{intersect}</span></td>
            </tr>"""

        panels_html += f"""
<div class="tab-panel{active}" id="{tid}">
  <div class="grid-2">
    <div class="card">
      <h2 class="card-title">{sec["name"]} — Sightline Spatial Distribution</h2>
      <div class="chart-container"><img src="data:image/png;base64,{sec["b64_sightline"]}" alt="sightline"></div>
      <div class="info-panel"><strong>Note:</strong> Shaded regions = virial radii (R₂₀₀). Horizontal line = FRB sightline (b=0). Overlap ⇒ sightline intersects the halo.</div>
    </div>
    <div class="card">
      <h2 class="card-title">{sec["name"]} — NFW Enclosed Mass Profile</h2>
      <div class="chart-container"><img src="data:image/png;base64,{sec["b64_mass"]}" alt="mass profile"></div>
      <div class="info-panel"><strong>Note:</strong> Dotted vertical lines = impact parameter; dashed = R₂₀₀. Scatter = enclosed mass at b.</div>
    </div>
  </div>
  <div class="card">
    <h2 class="card-title">{sec["name"]} — Galaxy & Halo Properties</h2>
    <table>
      <thead><tr>
        <th>Galaxy</th><th>z (FRB)</th><th>Comoving Dist</th><th>Impact b</th>
        <th>R_vir</th><th>Stellar Mass (source)</th><th>Halo Mass</th><th>Encl. Mass %</th><th>Intersects R₂₀₀</th>
      </tr></thead>
      <tbody>{trows}</tbody>
    </table>
  </div>
</div>"""

    html = (
        _HTML_HEAD.replace("__N_SIGHTLINES__", str(len(target_sections)))
        .replace(
            '<div class="tabs" id="tabs"></div>', f'<div class="tabs" id="tabs">{tabs_html}</div>'
        )
        .replace('<div id="panels"></div>', f'<div id="panels">{panels_html}</div>')
    )
    return html


# --------------- main ---------------


def main():
    # Resolve paths relative to the repo root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results")
    docs_dir = os.path.join(base_dir, "docs")

    # Import targets from config
    sys.path.insert(0, base_dir)
    from galaxies.v2_0.config import TARGETS
    from galaxies.v2_0.plotting import _split_galaxies_clusters

    target_sections = []

    for name, ra_str, dec_str, z_frb in TARGETS:
        csv_path = os.path.join(results_dir, f"{name.lower()}_galaxies.csv")
        if not os.path.exists(csv_path):
            print(f"  {name}: no CSV found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        # Drop cluster rows: this NFW/L*-halo dashboard models galaxies only; clusters
        # would mis-render as L* halos. Clusters now live in the on-sky maps (plotting.py).
        df, _clusters = _split_galaxies_clusters(df)
        if df.empty:
            print(f"  {name}: no foreground galaxies (clusters shown in sky maps), skipping.")
            continue

        # Build per-galaxy data
        gal_rows = []
        for _, row in df.iterrows():
            z_gal = row["z"]
            impact = row["impact_kpc"]

            if "M_star" in row and not np.isnan(row["M_star"]) and row["M_star"] > 0:
                log_mstar = row["M_star"]
                mass_source = "catalog"
            else:
                phot = estimate_logmstar_from_photometry(row, z_gal)
                if phot is not None:
                    log_mstar, mass_source = phot, "photometric"
                else:
                    # No catalog M* and no usable photometry (e.g. DESI grz
                    # without i-band): assume a typical L* galaxy.
                    log_mstar, mass_source = 10.0, "assumed"

            m_halo = estimate_halo_mass(log_mstar)
            r_vir, r_s, c = get_rvir_and_rs(m_halo, z_gal)
            d_com = cosmo.comoving_distance(z_gal).to(u.Mpc).value

            gal_rows.append(
                {
                    "z_gal": z_gal,
                    "d_com": d_com,
                    "impact": impact,
                    "log_mstar": log_mstar,
                    "mass_source": mass_source,
                    "m_halo": m_halo,
                    "r_vir": r_vir,
                    "r_s": r_s,
                    "c": c,
                }
            )

        if not gal_rows:
            continue

        print(f"  {name}: {len(gal_rows)} galaxies")

        # Generate per-target figures
        fig1, b64_sl = make_sightline_fig(name, z_frb, gal_rows)
        sl_path = os.path.join(results_dir, f"{name.lower()}_sightline.png")
        fig1.savefig(sl_path, dpi=300)
        plt.close(fig1)
        print(f"    → {sl_path}")

        fig2, b64_mp = make_mass_profile_fig(name, z_frb, gal_rows)
        mp_path = os.path.join(results_dir, f"{name.lower()}_mass_profile.png")
        fig2.savefig(mp_path, dpi=300)
        plt.close(fig2)
        print(f"    → {mp_path}")

        target_sections.append(
            {
                "name": name,
                "z_frb": z_frb,
                "gal_rows": gal_rows,
                "b64_sightline": b64_sl,
                "b64_mass": b64_mp,
            }
        )

    if not target_sections:
        print("No galaxy data found. Exiting.")
        return

    # Build HTML
    html = build_html(target_sections)

    html_path = os.path.join(results_dir, "galaxy_sightlines_report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nSaved HTML report → {html_path}")

    # Copy to docs/ for GitHub Pages
    os.makedirs(docs_dir, exist_ok=True)
    docs_path = os.path.join(docs_dir, "index.html")
    shutil.copy2(html_path, docs_path)
    print(f"Copied to → {docs_path}")


if __name__ == "__main__":
    main()
