#!/usr/bin/env python
"""Per-burst CHIME-DSA DM agreement figure (V6 / Phase 6 P6.2).

Reads ``crossmatching/dm_provenance.csv`` and plots, per burst, the DM residual
 delta_dm = DM_chime - DM_dsa with its combined error bar, a zero line, and the
+-1 pc/cm^3 "agreement floor" band that dm_status cites. A twin axis reports the
agreement in sigma.

The visual makes the P6.2 finding legible: every constrained burst sits inside
the ~1 pc/cm^3 floor (they agree), yet the formal sigma is large because it is
divided by the DSA-side 0.1 pc/cm^3 *placeholder* uncertainty -- so a large
sigma here is an artifact of the placeholder floor, not a real tension. CHIME
also reads systematically below DSA (7 of 8 residuals negative).

    conda run -n flits python scripts/plot_dm_agreement.py
"""
from __future__ import annotations

import csv
import pathlib
from math import hypot

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).parents[1]
CSV = ROOT / "crossmatching/dm_provenance.csv"
OUT = ROOT / "crossmatching/dm_agreement.png"

AGREEMENT_FLOOR = 1.0  # pc/cm^3, cited in chime_side_inputs dm_status


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))

    names = [r["nickname"] for r in rows]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    axr = ax.twinx()

    # +-1 pc/cm^3 agreement floor band and zero line.
    ax.axhspan(-AGREEMENT_FLOOR, AGREEMENT_FLOOR, color="0.85", zorder=0,
               label=f"$\\pm{AGREEMENT_FLOOR:.0f}$ pc cm$^{{-3}}$ agreement floor")
    ax.axhline(0.0, color="0.4", lw=1.0, zorder=1)

    con_x, con_y, con_yerr, con_sig = [], [], [], []
    unc_x = []
    for i, r in enumerate(rows):
        if r["delta_dm"]:
            delta = float(r["delta_dm"])
            # combined error = quadrature of CHIME stat err and DSA placeholder.
            comb = hypot(float(r["dm_chime_err"]), float(r["dm_dsa_err"]))
            con_x.append(i)
            con_y.append(delta)
            con_yerr.append(comb)
            con_sig.append(float(r["delta_dm_sigma"]))
        else:
            unc_x.append(i)

    ax.errorbar(con_x, con_y, yerr=con_yerr, fmt="o", color="C0",
                capsize=3, lw=1.2, ms=6, zorder=3, label="constrained (CHIME DM)")
    # sigma annotations above each constrained point.
    for xi, yi, si in zip(con_x, con_y, con_sig, strict=True):
        ax.annotate(f"{si:+.1f}$\\sigma$", (xi, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color="C0")

    for xi in unc_x:
        ax.scatter([xi], [0.0], marker="x", s=55, color="0.5", zorder=3)
    if unc_x:
        ax.scatter([], [], marker="x", s=55, color="0.5",
                   label="CHIME-unconstrained (no DM)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(r"$\Delta\mathrm{DM} = \mathrm{DM}_{\rm CHIME}"
                  r" - \mathrm{DM}_{\rm DSA}$  (pc cm$^{-3}$)")
    ax.set_xlabel("burst (chronological)")

    # Twin sigma axis using the 1 pc/cm^3 physical floor (association.py
    # sigma_eff convention): sigma = delta / max(quadrature, 1.0). Since the
    # combined error sits below 1, the floor governs and sigma ~= delta here.
    dm_floor = 1.0
    lo, hi = ax.get_ylim()
    axr.set_ylim(lo / dm_floor, hi / dm_floor)
    axr.set_ylabel(r"agreement ($\sigma_{\rm eff}$, floor $=1$ pc cm$^{-3}$)")

    ax.set_title("CHIME$-$DSA DM agreement per co-detected burst (V6 / P6.2)\n"
                 "$\\sigma_{\\rm eff}=\\max(\\mathrm{quadrature},\\,1$ pc cm"
                 "$^{-3})$; all constrained bursts agree within $\\pm1$ "
                 "(CHIME reads systematically low)",
                 fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
