# rp_analysis_notebook.py – analysis & visualisation notebook (script style)
"""High‑level analysis script that reproduces Figures 7 & 8 of Pradeep et al. (2025) from
Monte‑Carlo CSVs produced by batch_rp_sweep.py.  Supports multi‑frequency sweeps
and optional host‑screen diagnostics.

Usage:
    python rp_analysis_notebook.py                 # default λ=0.21 m, 20 MHz fs
    python rp_analysis_notebook.py --lam 0.10      # analyse 3 GHz sweep
    python rp_analysis_notebook.py --mode pow      # power‑multiplication bursts

This script is intentionally linear (one cell per section) so it can be
executed as a stand‑alone .py **or** imported into a Jupyter notebook and
run cell‑by‑cell via %run.
"""

from __future__ import annotations
import argparse, pathlib, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from frb_scintillator import Screen, Scintillator


CSV_DIR = pathlib.Path("csv")
FIG_DIR = pathlib.Path("figs"); FIG_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------
# 1.  Command‑line interface
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Reproduce RP figures from CSVs")
parser.add_argument("--lam", type=float, default=0.21, help="wavelength [m]")
parser.add_argument("--mode", choices=["coh", "pow"], default="coh",
                    help="propagation mode (coh/pow, must match CSV names)")
parser.add_argument("--fs",  type=int, default=20e6, help="sample rate [Hz]")
parser.add_argument("--nchan", type=int, default=512,
                   help="number of spectral channels (default 512)")
parser.add_argument("--save", action="store_true", help="save figs to PNG")
args = parser.parse_args()

# ------------------------------------------------------------------------
# 2.  Load CSVs
# ------------------------------------------------------------------------
pattern = f"summary_RP=*_{args.mode}_lam={args.lam:.3f}_fs={args.fs}_nchan={args.nchan}_*.csv"
files = sorted(CSV_DIR.glob(pattern))
if not files:
    print("No CSV files found for pattern:", pattern)
    sys.exit(1)

df = pd.concat(pd.read_csv(f) for f in files)
print(f"Loaded {len(df)} bursts from {len(files)} CSV files")

# ------------------------------------------------------------------------
# 3.  Figure 7 – m² vs RP  (all bursts)
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6,4))
for rp in sorted(df["RP_nominal"].unique()):
    sub = df[df["RP_nominal"] == rp]
    ax.scatter(np.full(len(sub), rp), sub["m2"], s=8, alpha=0.4,
               label=f"RP={rp:.2f}")
ax.set_xlabel("Resolution power RP")
ax.set_ylabel("Modulation‑index $m^2$")
ax.set_title("Figure 7: Burst‑level $m^2$ vs RP")
ax.set_xlim(0, 10)
ax.legend(title="Nominal RP", loc="upper right")
if args.save:
    fig.savefig(FIG_DIR / f"fig7_m2_vs_rp_mode_{args.mode}.png", dpi=150)
plt.show()

# ------------------------------------------------------------------------
# 4.  Figure 8 – bandwidths (MW & host) vs RP
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6,4))
for rp in sorted(df["RP_nominal"].unique()):
    sub = df[df["RP_nominal"] == rp]
    ax.errorbar(np.full(len(sub), rp), sub["nu_s_mw"], fmt="o", ms=4,
                alpha=0.4, label=None, color="tab:blue")
    ax.errorbar(np.full(len(sub), rp), sub["nu_s_host"], fmt="x", ms=4,
                alpha=0.4, color="tab:orange")
ax.set_xlabel("Resolution power RP")
ax.set_ylabel("Scintillation bandwidth HWHM [Hz]")
ax.set_title("Figure 8: MW (blue) & host (orange) bandwidths vs RP")
ax.set_xlim(0, 10)
ax.set_yscale("log")
if args.save:
    fig.savefig(FIG_DIR / f"fig8_bandwidth_vs_rp_mode_{args.mode}.png", dpi=150)
plt.show()

# ------------------------------------------------------------------------
# 5.  Additional physics checks (example)
# ------------------------------------------------------------------------
print("\nQuick diagnostics:")
for rp in sorted(df["RP_nominal"].unique()):
    sub = df[df["RP_nominal"] == rp]
    print(f"RP {rp:.2f}: m² mean={sub['m2'].mean():.2f}, MW ν_s median={sub['nu_s_mw'].median():.1f} Hz")
