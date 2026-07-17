#!/usr/bin/env python
"""Regenerate jointmodel_pair data/model/residual triptychs from the NEW-prep NPZ
dumps (S/N-driven resolution + robust common window). Mirrors the manuscript
plot_jointmodel_pair layout but resolves each burst's dump by its actual tag
(the beta-campaign suffix naming and the run_joint_fit tag naming differ), so it
plots exactly what the campaign produced.

  FLITS_RUNS=~/Developer/scratch/flits-local-runs conda run -n flits \
    python plot_triptychs_new.py --burst whitney_fine:_C2D2 --burst freya:''
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scattering"))
sys.path.insert(0, str(HERE))

from plot_jointmodel_pair import _aligned_bands  # noqa: E402
from flits.batch.codetection_plots import plot_codetection  # noqa: E402

RUNS = Path(os.environ.get("FLITS_RUNS", os.path.expanduser("~/Developer/scratch/flits-local-runs")))

# Evidence-selected component counts (joint_ladder/_figs.py chosen map; whitney base->C2D2,
# beta-native re-fit). tag matches run_joint_fit's output suffix.
CHOSEN = {
    "freya": "", "casey": "", "chromatica": "", "wilhelm": "",
    "hamilton": "_C4D1", "mahi": "_C1D1", "oran": "_C2D1", "isha": "_C2D1",
    "whitney_fine": "_C2D2", "johndoeII": "_C2D2", "phineas": "_C3D3", "zach": "_C2D3",
}


def resid_verdict(burst, tag):
    """One-line residual accept-gate verdict for the triptych title."""
    import json
    fp = RUNS / "data/joint" / f"{burst}_jointmodel{tag}_resid.json"
    if not fp.exists():
        return ""
    r = json.load(open(fp))
    c, d = r.get("C", {}), r.get("D", {})
    tags = []
    for band, x in (("C", c), ("D", d)):
        if x.get("escalate"):
            tags.append(f"{band}:ESCALATE {x.get('resid_prof_max', 0):+.0f}s")
        elif x.get("shape_mismatch"):
            tags.append(f"{band}:shape {x.get('resid_prof_max_abs', 0):.0f}s")
    return ("  resid " + " ".join(tags)) if tags else "  resid clean"


def find_npz(burst: str, tag: str | None) -> Path:
    d = RUNS / "data/joint"
    if tag is not None:
        p = d / f"{burst}_jointmodel{tag}.npz"
        if p.exists():
            return p
    # fall back to any jointmodel dump for this burst (newest)
    cands = sorted(d.glob(f"{burst}_jointmodel*.npz"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no jointmodel NPZ for {burst} under {d}")
    return cands[-1]


def make(burst: str, tag: str | None, out_dir: Path) -> Path:
    fp = find_npz(burst, tag)
    z = np.load(fp, allow_pickle=True)
    bands = _aligned_bands(z, burst)
    chi = ""
    if "chi2C" in z:
        chi = f"  chi2 C={float(z['chi2C']):.2f} D={float(z['chi2D']):.2f}"
    alpha = float(z["alpha"]) if "alpha" in z else float("nan")
    tau = float(z["tau_1ghz"]) if "tau_1ghz" in z else float("nan")
    nC = int(z["nC"]) if "nC" in z else 1
    nD = int(z["nD"]) if "nD" in z else 1
    verdict = resid_verdict(burst, tag if tag is not None else "")
    fig = plot_codetection(
        bands,
        columns=("data", "model", "resid"),
        show_model_on_data=False,
        per_band_scale=True,
        gap_label=False,
        figsize=(12.4, 4.9),
        band_labels=False,
        show_column_titles=True,
        per_band_marginals=True,
        title=f"{burst} NEW (C{nC}D{nD})  alpha={alpha:.2f} tau={tau:.3f} ms{chi}{verdict}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{burst.removesuffix('_fine')}_jointmodel_pair_NEW.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--burst", action="append", default=None,
                    help="burst[:tag], e.g. whitney_fine:_C2D2 or freya: (empty tag). "
                    "Omit to render all 12 at their chosen counts (CHOSEN map).")
    ap.add_argument("--out-dir", type=Path, default=HERE / "joint_tf_figs")
    a = ap.parse_args()
    specs = a.burst if a.burst else [f"{b}:{t}" for b, t in CHOSEN.items()]
    for spec in specs:
        burst, sep, tag = spec.partition(":")
        tag = tag if sep else None
        try:
            out = make(burst, tag, a.out_dir)
            print(f"{burst}: {out}")
        except Exception as e:  # keep going; report the failure
            print(f"{burst}: FAILED {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
