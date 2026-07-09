#!/usr/bin/env python
"""PBF systematic figure: mixed (per-band) vs all-exponential α per co-detection.

Exports vector SVG (+ PDF/PNG) to Faber2026 figures/ for \\includegraphics{alpha_pbf_systematic}.
Labels from burst_metadata TNS map (ADR-0002). zach recolored grey (ADR-0003).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scattering.scat_analysis.burst_metadata import load_tns_name  # noqa: E402

REFIT = Path(__file__).resolve().parent
ALLEXP_DIR = REFIT / "_a1_fits"
OUT_DIR = Path(__file__).resolve().parents[3] / "figures"  # Faber2026/figures

RAIL_EDGE = 0.1
KOLM = 4.4

# Canonical all-exp model tag (grade_allexp.py)
CANON = {
    "casey": "sharedzeta",
    "chromatica": "sharedzeta",
    "freya": "sharedzeta",
    "wilhelm": "sharedzeta",
    "mahi": "C1D1",
    "phineas": "C3D3",
    "oran": "C2D1",
    "isha": "C2D1",
    "johndoeII": "C2D1",
    "zach": "",
    "hamilton": "sharedzeta",
}

# Bursts without _a1_fits pull (local/HPC-only): (alpha_exp, err_plus, err_minus)
EXP_FALLBACK = {
    "whitney": (5.12, 0.09, 0.09),
    "hamilton": (1.504, 0.05, 0.05),
}

# Profile-bias demonstrator withheld — never green (ADR-0003)
FORCE_GREY = {"zach"}

# Original mixed-PBF ladder α (HPCC campaign, ALLEXP_PBF_RUN.md table)
MIXED_LADDER = {
    "hamilton": 1.50,
    "johndoeII": 1.58,
    "casey": 2.40,
    "wilhelm": 2.56,
    "oran": 2.69,
    "mahi": 3.80,
    "chromatica": 3.28,
    "zach": 2.41,
    "phineas": 3.33,
    "freya": 4.36,
    "whitney": 5.21,
    "isha": 5.39,
}

# Plot order (x-axis): ascending mixed α for readability
BURSTS = [
    "hamilton",
    "johndoeII",
    "casey",
    "wilhelm",
    "oran",
    "mahi",
    "chromatica",
    "zach",
    "phineas",
    "freya",
    "whitney",
    "isha",
]


def _alpha(fit: dict) -> tuple[float, float, float]:
    a = fit["alpha"]
    return a["median"], a.get("err_plus", 0.0), a.get("err_minus", 0.0)


def _railed(fit: dict) -> bool:
    med, _, _ = _alpha(fit)
    lo, hi = fit["alpha_bounds"]
    return min(med - lo, hi - med) < RAIL_EDGE


def _load_mixed(nick: str) -> dict | None:
    if nick not in MIXED_LADDER:
        return None
    med = MIXED_LADDER[nick]
    return {"alpha": {"median": med, "err_plus": 0.0, "err_minus": 0.0}, "alpha_bounds": [1.0, 6.0]}


def _load_exp(nick: str) -> dict | None:
    tag = CANON.get(nick, "")
    suffix = f"_{tag}" if tag else ""
    fp = ALLEXP_DIR / f"{nick}_joint_fit{suffix}_pbf-exp-exp.json"
    if fp.exists():
        return json.loads(fp.read_text())
    if nick in EXP_FALLBACK:
        med, ep, em = EXP_FALLBACK[nick]
        return {"alpha": {"median": med, "err_plus": ep, "err_minus": em}, "alpha_bounds": [1.0, 6.0]}
    return None


def _classify(nick: str, d_alpha: float, rail_exp: bool) -> str:
    if nick in FORCE_GREY or rail_exp:
        return "grey"
    if abs(d_alpha) <= 0.1:
        return "green"
    return "orange"


def main():
    rows = []
    for nick in BURSTS:
        mixed = _load_mixed(nick)
        exp = _load_exp(nick)
        if mixed is None or exp is None:
            print(f"skip {nick}: mixed={mixed is not None} exp={exp is not None}", file=sys.stderr)
            continue
        am, _, _ = _alpha(mixed)
        ae, ep, em = _alpha(exp)
        rows.append(
            {
                "nick": nick,
                "label": load_tns_name(nick),
                "alpha_mixed": am,
                "alpha_exp": ae,
                "err_p": ep,
                "err_m": em,
                "d_alpha": ae - am,
                "rail_exp": _railed(exp),
                "color": _classify(nick, ae - am, _railed(exp)),
            }
        )

    rows.sort(key=lambda r: r["alpha_mixed"])
    x = np.arange(len(rows))
    colors = {"green": "#1b7837", "orange": "#d95f02", "grey": "#888888"}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, r in enumerate(rows):
        c = colors[r["color"]]
        ax.plot(i, r["alpha_mixed"], marker="o", mfc="white", mec=c, ms=7, mew=1.4, zorder=3)
        ax.errorbar(
            i,
            r["alpha_exp"],
            yerr=[[r["err_m"]], [r["err_p"]]],
            fmt="o",
            color=c,
            ms=7,
            capsize=2.5,
            zorder=4,
        )

    ax.axhline(KOLM, color="#737373", ls="--", lw=1.2, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"Scattering index $\alpha$")
    ax.set_xlabel("Co-detected FRB")
    ax.set_ylim(0.8, 6.5)
    ax.grid(axis="y", alpha=0.25)

    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w", markeredgecolor=colors["green"], label=r"$|\Delta\alpha|\leq0.1$ (PBF-insensitive)"),
        Line2D([0], [0], marker="o", color=colors["orange"], linestyle="None", label=r"$|\Delta\alpha|>0.1$ (PBF-sensitive)"),
        Line2D([0], [0], marker="o", color=colors["grey"], linestyle="None", label="railed / withheld"),
        Line2D([0], [0], color="#737373", ls="--", label=rf"Kolmogorov $\alpha={KOLM}$"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="upper left")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "alpha_pbf_systematic"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"))
    print(f"wrote {stem}.{{svg,pdf,png}}  ({len(rows)} bursts)")
    for r in rows:
        print(f"  {r['label']:16s} mixed={r['alpha_mixed']:.3f} exp={r['alpha_exp']:.3f} d={r['d_alpha']:+.3f} {r['color']}")


if __name__ == "__main__":
    main()
