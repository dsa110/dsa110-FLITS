#!/usr/bin/env python
"""Figures for the A1 trigger-calibration report (Phase 4/6 deliverable).

Reads reports/a1_trigger_calibration.json + its .cells/ checkpoints and
writes three figures next to the report under figures/:
  - a1_null_dlnz_contactsheet.png : per-cell null dlnZ histograms with the
    0.5/1/5% envelope thresholds (visual-vetting rule: tails must be
    inspected, not just quantiled);
  - a1_threshold_vs_rate.png      : envelope threshold vs false-escalation
    rate;
  - a1_power_curves.png           : escalation probability at the 1%
    operating point vs width ratio f, per m2 ratio.

Usage: python simulation/scripts/plot_a1_trigger_calibration.py \
           [--report reports/a1_trigger_calibration.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_cells(cells_dir):
    cells = {}
    for f in sorted(Path(cells_dir).glob("*.json")):
        d = json.loads(f.read_text())
        cells[f.stem] = np.asarray(d["sample"], dtype=float)
    return cells


def contact_sheet(null_cells, thresholds, out_png):
    keys = sorted(null_cells)
    n = len(keys)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    for ax, k in zip(axes.ravel(), keys):
        v = null_cells[k]
        finite = v[np.isfinite(v)]
        ax.hist(finite, bins=30, color="#4878a8", alpha=0.85)
        for rate, ls in ((0.01, "-"), (0.05, "--")):
            ax.axvline(thresholds[str(rate)], color="#b0413e", ls=ls, lw=1)
        n_fail = int(np.sum(~np.isfinite(v)))
        ax.set_title(k.replace("null__", ""), fontsize=6.5)
        ax.text(0.97, 0.92, f"fail {n_fail}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6,
                color="#b0413e" if n_fail else "#666666")
        ax.tick_params(labelsize=6)
    fig.suptitle("A1 null $\\Delta\\ln Z$ per cell (red: 1% solid / 5% dashed envelope)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def threshold_curve(null_cells, out_png, rates=None):
    rates = rates if rates is not None else np.geomspace(0.002, 0.2, 25)
    env = []
    for r in rates:
        per_cell = [np.quantile(v[np.isfinite(v)], 1 - r)
                    for v in null_cells.values() if np.isfinite(v).any()]
        env.append(max(per_cell))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.semilogx(rates, env, color="#4878a8")
    for r, m in ((0.005, "o"), (0.01, "s"), (0.05, "^")):
        per_cell = [np.quantile(v[np.isfinite(v)], 1 - r)
                    for v in null_cells.values() if np.isfinite(v).any()]
        ax.plot(r, max(per_cell), m, color="#b0413e")
    ax.set_xlabel("false-escalation rate")
    ax.set_ylabel(r"$\Delta\ln Z$ threshold (envelope)")
    ax.set_title("A1 operating curve")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def power_curves(power_cells, op_dlnz, out_png):
    # keys: power__f-<f>_m2_ratio-<m>
    rows = {}
    for k, v in power_cells.items():
        parts = dict(p.split("-", 1) for p in
                     k.replace("power__", "").split("_") if "-" in p)
        f = float(parts.get("f", "nan"))
        m = float(parts.get("ratio", parts.get("m2_ratio", "nan")))
        finite = v[np.isfinite(v)]
        rows.setdefault(m, []).append((f, float(np.mean(finite >= op_dlnz))))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m, pts in sorted(rows.items()):
        pts = sorted(pts)
        ax.semilogx([p[0] for p in pts], [p[1] for p in pts], "o-",
                    label=rf"$m_2^2/m_1^2$ = {m:g}")
    ax.axhline(1.0, color="#999999", lw=0.5)
    ax.set_xlabel(r"width ratio $f = \gamma_2/\gamma_1$")
    ax.set_ylabel(rf"P(escalate) at $\Delta\ln Z \geq$ {op_dlnz:.1f}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("A1 detection power at the 1% operating point")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/a1_trigger_calibration.json")
    args = ap.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text())
    cells = _load_cells(report_path.parent /
                        (report_path.stem + ".cells"))
    null_cells = {k: v for k, v in cells.items() if k.startswith("null__")}
    power = {k: v for k, v in cells.items() if k.startswith("power__")}

    fig_dir = report_path.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    contact_sheet(null_cells, report["thresholds"],
                  fig_dir / "a1_null_dlnz_contactsheet.png")
    threshold_curve(null_cells, fig_dir / "a1_threshold_vs_rate.png")
    power_curves(power, report["recommended_operating_point"]["dlnz"],
                 fig_dir / "a1_power_curves.png")
    print(f"wrote 3 figures to {fig_dir}")


if __name__ == "__main__":
    main()
