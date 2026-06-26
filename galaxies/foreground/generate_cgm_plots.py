#!/usr/bin/env python3
"""Generate CGM and scattering-prior dashboard figures from unified CSVs.

Outputs, when ``main()`` is run:
  results/{name}_cgm.png
  results/galaxy_cgm_report.html
  docs/cgm.html
"""
from __future__ import annotations

import ast
import base64
import io
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .scattering_predict import predict_mgii_wr
except ImportError:  # pragma: no cover - supports direct script execution.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from galaxies.foreground.scattering_predict import predict_mgii_wr

# --------------- styling ---------------
DARK_BLUE = "#1B365D"
LIGHT_BLUE = "#4A90E2"
ACCENT_ORANGE = "#F5A623"
ACCENT_RED = "#D0021B"
TEXT_DARK = "#333333"
GRID_COLOR = "#E5E5E5"
BG_LIGHT = "#FAFBFC"

_PREDICTION_NOTE = "Predicted from scaling-relation priors - not direct measurements"
_TAU_DETECTABILITY_MS = 1.0e-3


def _coerce_flags(value):
    """Return a dict from an in-memory flags dict or CSV round-tripped repr."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _finite(x):
    """Return True for finite, non-sentinel scalar values."""
    if x is None:
        return False
    try:
        value = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > -9990.0


def _safe_float(x):
    return float(x) if _finite(x) else np.nan


def _clean_unified_df(df):
    if df is None:
        return pd.DataFrame()
    cleaned = df.copy()
    for col in cleaned.columns:
        if col == "cgm_extractable_flags":
            continue
        if pd.api.types.is_bool_dtype(cleaned[col]):
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned.loc[cleaned[col] <= -9990, col] = np.nan
    return cleaned


def _truthy(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64


def _placeholder_fig(title, text):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150, facecolor=BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color=TEXT_DARK, wrap=True)
    ax.set_title(title, fontsize=13, fontweight="bold", color=DARK_BLUE, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig, _fig_to_b64(fig)


def make_tau_rank_fig(name, z_frb, unified_df):
    """Return (fig, b64_png) for ranked predicted scattering time at 1 GHz."""
    df = _clean_unified_df(unified_df)
    if df.empty or "intersects_rvir" not in df:
        return _placeholder_fig(
            f"{name} - Predicted Scattering Rank",
            f"No galaxies intersect R_vir for {name}",
        )

    rows = df[df["intersects_rvir"].map(_truthy)].copy()
    if rows.empty:
        return _placeholder_fig(
            f"{name} - Predicted Scattering Rank",
            f"No galaxies intersect R_vir for {name}",
        )

    rows["pred_tau_scat_ms_1GHz"] = rows["pred_tau_scat_ms_1GHz"].map(_safe_float)
    rows = rows.sort_values("pred_tau_scat_ms_1GHz", ascending=False, na_position="last")
    values = rows["pred_tau_scat_ms_1GHz"].fillna(0.0).to_numpy(dtype=float)
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150, facecolor=BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)

    lo = rows.get("pred_tau_scat_ms_1GHz_lo", pd.Series(np.nan, index=rows.index)).map(_safe_float)
    hi = rows.get("pred_tau_scat_ms_1GHz_hi", pd.Series(np.nan, index=rows.index)).map(_safe_float)
    lo_values = lo.to_numpy(dtype=float)
    hi_values = hi.to_numpy(dtype=float)
    lo_values = np.where(np.isfinite(lo_values), lo_values, values)
    hi_values = np.where(np.isfinite(hi_values), hi_values, values)
    lower = np.maximum(values - lo_values, 0.0)
    upper = np.maximum(hi_values - values, 0.0)

    colors = [LIGHT_BLUE if _truthy(v) else ACCENT_RED for v in rows.get("is_star_forming", [])]
    if len(colors) != len(rows):
        colors = [LIGHT_BLUE] * len(rows)
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    ax.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="none", ecolor=TEXT_DARK, capsize=4, lw=1, zorder=4)

    for i, (_, row) in enumerate(rows.iterrows()):
        z_label = row.get("z", np.nan)
        br = row.get("b_over_rvir", np.nan)
        label = f"z={z_label:.3f}\nb/Rvir={br:.2f}" if _finite(z_label) and _finite(br) else "prior"
        y_text = values[i] if values[i] > 0 else max(np.nanmax(values), 1.0e-3) * 0.05
        ax.annotate(label, xy=(i, y_text), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=TEXT_DARK)

    # A 1 microsecond guide is an FRB single-pulse temporal-resolution scale,
    # so it is shown as an order-of-magnitude detectability reference, not a
    # calibrated detection threshold.
    ax.axhline(_TAU_DETECTABILITY_MS, color=ACCENT_ORANGE, linestyle="--", linewidth=1.6, zorder=2)
    ax.text(
        len(rows) - 0.5,
        _TAU_DETECTABILITY_MS,
        "~1 us detectability floor",
        ha="right",
        va="bottom",
        fontsize=8,
        color=ACCENT_ORANGE,
    )

    positive = values[np.isfinite(values) & (values > 0.0)]
    if len(positive) and not np.allclose(positive, 0.0):
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.88, "No positive finite tau priors; using linear scale", transform=ax.transAxes, ha="center", color=ACCENT_RED, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) if _finite(r) else str(i + 1) for i, r in enumerate(rows.get("scattering_rank", x + 1))])
    ax.set_xlabel("Scattering rank (1 = strongest)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel(r"Predicted $\tau_{\rm scat}$ at 1 GHz (ms)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_title(f"{name} - Predicted CGM Scattering Priors", fontsize=13, fontweight="bold", color=DARK_BLUE, pad=12)
    ax.text(0.01, 0.98, _PREDICTION_NOTE, transform=ax.transAxes, va="top", fontsize=8.5, color=TEXT_DARK)
    ax.grid(True, axis="y", linestyle=":", color=GRID_COLOR, alpha=0.8, zorder=0)
    fig.tight_layout()
    return fig, _fig_to_b64(fig)


def make_covering_fraction_fig(name, unified_df):
    """Return (fig, b64_png) for cool-CGM covering-fraction priors."""
    df = _clean_unified_df(unified_df)
    needed = {"b_over_rvir", "cool_fc"}
    if df.empty or not needed.issubset(df.columns):
        return _placeholder_fig(f"{name} - Cool CGM Covering Fraction", f"No covering-fraction priors available for {name}")

    rows = df.copy()
    rows["b_over_rvir"] = rows["b_over_rvir"].map(_safe_float)
    rows["cool_fc"] = rows["cool_fc"].map(_safe_float)
    rows["cool_fc_lo"] = rows.get("cool_fc_lo", pd.Series(np.nan, index=rows.index)).map(_safe_float)
    rows["cool_fc_hi"] = rows.get("cool_fc_hi", pd.Series(np.nan, index=rows.index)).map(_safe_float)
    rows = rows[np.isfinite(rows["b_over_rvir"]) & np.isfinite(rows["cool_fc"])]
    if rows.empty:
        return _placeholder_fig(f"{name} - Cool CGM Covering Fraction", f"No covering-fraction priors available for {name}")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150, facecolor=BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)
    for is_sf, color, label in ((True, LIGHT_BLUE, "Star forming"), (False, ACCENT_RED, "Passive")):
        subset = rows[rows.get("is_star_forming", False).map(_truthy) == is_sf]
        if subset.empty:
            continue
        y = subset["cool_fc"].to_numpy(dtype=float)
        lo_values = subset["cool_fc_lo"].to_numpy(dtype=float)
        hi_values = subset["cool_fc_hi"].to_numpy(dtype=float)
        lo_values = np.where(np.isfinite(lo_values), lo_values, y)
        hi_values = np.where(np.isfinite(hi_values), hi_values, y)
        ylo = np.maximum(y - lo_values, 0.0)
        yhi = np.maximum(hi_values - y, 0.0)
        ax.errorbar(
            subset["b_over_rvir"],
            y,
            yerr=np.vstack([ylo, yhi]),
            fmt="o",
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markersize=8,
            capsize=3,
            alpha=0.9,
            label=label,
            zorder=3,
        )

    ax.set_xlabel(r"Impact parameter / $R_{\rm vir}$", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel("Cool CGM covering-fraction prior", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{name} - Cool CGM Covering Fraction Priors", fontsize=13, fontweight="bold", color=DARK_BLUE, pad=12)
    ax.text(0.01, 0.98, _PREDICTION_NOTE, transform=ax.transAxes, va="top", fontsize=8.5, color=TEXT_DARK)
    ax.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.8)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor=GRID_COLOR, fontsize=8.5)
    fig.tight_layout()
    return fig, _fig_to_b64(fig)


def make_mgii_fig(name, unified_df):
    """Return (fig, b64_png) for predicted MgII equivalent-width priors."""
    df = _clean_unified_df(unified_df)
    if df.empty or not {"impact_kpc", "pred_mgii_wr"}.issubset(df.columns):
        return _placeholder_fig(f"{name} - Predicted MgII", f"No MgII priors available for {name}")

    rows = df.copy()
    rows["impact_kpc"] = rows["impact_kpc"].map(_safe_float)
    rows["pred_mgii_wr"] = rows["pred_mgii_wr"].map(_safe_float)
    rows = rows[np.isfinite(rows["impact_kpc"]) & np.isfinite(rows["pred_mgii_wr"]) & (rows["impact_kpc"] > 0) & (rows["pred_mgii_wr"] > 0)]
    if rows.empty:
        return _placeholder_fig(f"{name} - Predicted MgII", f"No MgII priors available for {name}")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150, facecolor=BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)
    colors = [LIGHT_BLUE if _truthy(v) else ACCENT_RED for v in rows.get("is_star_forming", [])]
    if len(colors) != len(rows):
        colors = [LIGHT_BLUE] * len(rows)
    ax.scatter(rows["impact_kpc"], rows["pred_mgii_wr"], s=75, c=colors, edgecolor="white", linewidth=1.1, zorder=4, label="Unified-row prior")

    x_max = max(300.0, float(rows["impact_kpc"].max()) * 1.15)
    impact_grid = np.linspace(5.0, x_max, 300)
    # Nielsen+2013 MAGIICAT and Anand+2024 motivate the MgII broken-power-law
    # underlay; calling the canonical predictor keeps the reference curve
    # consistent with the per-row prior generation.
    wr_grid = np.array([predict_mgii_wr(b, logmstar=10.5) for b in impact_grid], dtype=float)
    valid = np.isfinite(wr_grid) & (wr_grid > 0.0)
    ax.plot(impact_grid[valid], wr_grid[valid], color=DARK_BLUE, linestyle="--", linewidth=2, label="Nielsen/Anand reference")

    ax.set_yscale("log")
    ax.set_xlabel("Impact parameter (kpc)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel(r"Predicted MgII 2796 $W_r$ (Angstrom)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_title(f"{name} - Predicted MgII Absorption Priors", fontsize=13, fontweight="bold", color=DARK_BLUE, pad=12)
    ax.text(0.01, 0.98, "predicted (prior) - not a direct MgII measurement", transform=ax.transAxes, va="top", fontsize=8.5, color=TEXT_DARK)
    ax.grid(True, which="both", linestyle=":", color=GRID_COLOR, alpha=0.8)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor=GRID_COLOR, fontsize=8.5)
    fig.tight_layout()
    return fig, _fig_to_b64(fig)


def make_wise_diagnostic_fig(name, unified_df):
    """Return (fig, b64_png) for the WISE color-color AGN diagnostic."""
    df = _clean_unified_df(unified_df)
    required = {"W1mag", "W2mag", "W3mag"}
    if df.empty or not required.issubset(df.columns):
        return _placeholder_fig(f"{name} - WISE Diagnostic", "WISE not available at this sightline")

    rows = df.copy()
    for col in required:
        rows[col] = rows[col].map(_safe_float)
    rows = rows[np.isfinite(rows["W1mag"]) & np.isfinite(rows["W2mag"]) & np.isfinite(rows["W3mag"])]
    if rows.empty:
        return _placeholder_fig(f"{name} - WISE Diagnostic", "WISE not available at this sightline")

    x = rows["W2mag"] - rows["W3mag"]
    y = rows["W1mag"] - rows["W2mag"]
    agn = rows.get("wise_agn_flag", pd.Series(False, index=rows.index)).map(_truthy)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150, facecolor=BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)
    ax.scatter(x[~agn], y[~agn], s=80, color=LIGHT_BLUE, edgecolor="white", linewidth=1.1, label="WISE measured", zorder=4)
    if agn.any():
        ax.scatter(x[agn], y[agn], s=90, color=ACCENT_ORANGE, edgecolor="white", linewidth=1.1, label="WISE AGN flag", zorder=5)

    # Stern+2012 ApJ 753,30 selects WISE AGN with W1-W2 >= 0.8 mag (Vega);
    # this horizontal line is the minimum robust diagnostic when W4 is absent.
    ax.axhline(0.8, color=ACCENT_RED, linestyle="--", linewidth=1.8, label="Stern+2012 AGN threshold")
    ax.fill_between([max(0.0, x.min() - 0.3), x.max() + 0.3], 0.8, max(1.6, y.max() + 0.2), color=ACCENT_RED, alpha=0.08)
    ax.set_xlabel("W2 - W3 (Vega mag)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel("W1 - W2 (Vega mag)", fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax.set_title(f"{name} - WISE Color Diagnostic", fontsize=13, fontweight="bold", color=DARK_BLUE, pad=12)
    ax.text(0.01, 0.98, "WISE colors shown only where catalog magnitudes are measured", transform=ax.transAxes, va="top", fontsize=8.5, color=TEXT_DARK)
    ax.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.8)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor=GRID_COLOR, fontsize=8.5)
    fig.tight_layout()
    return fig, _fig_to_b64(fig)


_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FRB CGM & Scattering Dashboard</title>
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
  <h1>FRB CGM &amp; Scattering Dashboard</h1>
  <p class="subtitle">Scaling-relation priors for cool CGM, MgII absorption, WISE colors, and FRB scattering &mdash; __N_SIGHTLINES__ sightlines</p>
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


def _target_summary_table(sec):
    df = _clean_unified_df(sec["unified_df"])
    if df.empty:
        return "<tr><td colspan=\"4\">No rows available</td></tr>"
    n_intersect = int(df.get("intersects_rvir", pd.Series(False, index=df.index)).map(_truthy).sum())
    wise_cols = {"W1mag", "W2mag", "W3mag"}
    if wise_cols.issubset(df.columns):
        n_wise = int(df[list(wise_cols)].apply(lambda col: col.map(_finite)).all(axis=1).sum())
    else:
        n_wise = 0

    rank_rows = df.copy()
    if "scattering_rank" in rank_rows:
        rank_rows["scattering_rank"] = rank_rows["scattering_rank"].map(_safe_float)
        rank_rows = rank_rows.sort_values("scattering_rank", na_position="last")
    top = rank_rows.iloc[0]
    top_z = top.get("z", np.nan)
    top_tau = top.get("pred_tau_scat_ms_1GHz", np.nan)
    top_text = "not available"
    if _finite(top_z) and _finite(top_tau):
        top_text = f"z={float(top_z):.4f}, tau={float(top_tau):.3g} ms"

    return f"""<tr>
      <td>{len(df)}</td>
      <td><span class="badge {'badge-yes' if n_intersect else 'badge-no'}">{n_intersect}</span></td>
      <td>{top_text}</td>
      <td>{n_wise}</td>
    </tr>"""


def build_cgm_html(target_sections):
    """Return complete self-contained HTML with embedded per-target tabs."""
    tabs_html = ""
    panels_html = ""

    for i, sec in enumerate(target_sections):
        tid = f"panel-{sec['name'].lower()}"
        active = " active" if i == 0 else ""
        tabs_html += f'<button class="tab-btn{active}" data-target="{tid}">{sec["name"]}</button>\n'
        summary_row = _target_summary_table(sec)
        panels_html += f"""
<div class="tab-panel{active}" id="{tid}">
  <div class="info-panel"><strong>Honesty note:</strong> Tau, MgII equivalent width, cool covering fraction, and related CGM values are predicted from scaling-relation priors - not direct measurements. WISE diagnostics are shown only when WISE magnitudes are measured; unavailable WISE data are not converted to zeros.</div>
  <div class="card">
    <h2 class="card-title">{sec['name']} - CGM Summary</h2>
    <table>
      <thead><tr><th>Total rows</th><th>Intersects Rvir</th><th>Top scattering row</th><th>Rows with WISE</th></tr></thead>
      <tbody>{summary_row}</tbody>
    </table>
  </div>
  <div class="grid-2">
    <div class="card">
      <h2 class="card-title">{sec['name']} - Predicted Scattering Rank</h2>
      <div class="chart-container"><img src="data:image/png;base64,{sec['b64_tau']}" alt="predicted scattering rank"></div>
    </div>
    <div class="card">
      <h2 class="card-title">{sec['name']} - Cool CGM Covering Fraction</h2>
      <div class="chart-container"><img src="data:image/png;base64,{sec['b64_fc']}" alt="cool CGM covering fraction"></div>
    </div>
    <div class="card">
      <h2 class="card-title">{sec['name']} - MgII Prior</h2>
      <div class="chart-container"><img src="data:image/png;base64,{sec['b64_mgii']}" alt="predicted MgII equivalent width"></div>
    </div>
    <div class="card">
      <h2 class="card-title">{sec['name']} - WISE Diagnostic</h2>
      <div class="chart-container"><img src="data:image/png;base64,{sec['b64_wise']}" alt="WISE diagnostic"></div>
    </div>
  </div>
</div>"""

    html = _HTML_HEAD.replace("__N_SIGHTLINES__", str(len(target_sections))).replace(
        '<div class="tabs" id="tabs"></div>',
        f'<div class="tabs" id="tabs">{tabs_html}</div>',
    ).replace(
        '<div id="panels"></div>',
        f'<div id="panels">{panels_html}</div>',
    )
    return html


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results")
    docs_dir = os.path.join(base_dir, "docs")

    sys.path.insert(0, base_dir)
    from galaxies.foreground.config import TARGETS

    target_sections = []
    for name, ra_str, dec_str, z_frb in TARGETS:
        csv_path = os.path.join(results_dir, f"{name.lower()}_unified.csv")
        if not os.path.exists(csv_path):
            print(f"  {name}: no unified CSV found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  {name}: unified CSV empty, skipping.")
            continue
        df = _clean_unified_df(df)

        fig_tau, b64_tau = make_tau_rank_fig(name, z_frb, df)
        fig_fc, b64_fc = make_covering_fraction_fig(name, df)
        fig_mgii, b64_mgii = make_mgii_fig(name, df)
        fig_wise, b64_wise = make_wise_diagnostic_fig(name, df)

        cgm_path = os.path.join(results_dir, f"{name.lower()}_cgm.png")
        fig_tau.savefig(cgm_path, dpi=300)
        print(f"  {name}: wrote {cgm_path}")

        for fig in (fig_tau, fig_fc, fig_mgii, fig_wise):
            plt.close(fig)

        target_sections.append(
            {
                "name": name,
                "z_frb": z_frb,
                "b64_tau": b64_tau,
                "b64_fc": b64_fc,
                "b64_mgii": b64_mgii,
                "b64_wise": b64_wise,
                "unified_df": df,
            }
        )

    if not target_sections:
        print("No unified CGM data found.")
        return

    html = build_cgm_html(target_sections)
    os.makedirs(results_dir, exist_ok=True)
    html_path = os.path.join(results_dir, "galaxy_cgm_report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved HTML report -> {html_path}")

    os.makedirs(docs_dir, exist_ok=True)
    docs_path = os.path.join(docs_dir, "cgm.html")
    shutil.copy2(html_path, docs_path)
    print(f"Copied to -> {docs_path}")


if __name__ == "__main__":
    main()
