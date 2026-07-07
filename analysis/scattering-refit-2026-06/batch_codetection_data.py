#!/usr/bin/env python
"""Render connected CHIME+gap+DSA observation figures for all co-detections.

Uses raw .npy cubes (not joint-fit models). TOA alignment from
crossmatching/toa_crossmatch_results.json.

  FLITS_DATA=~/Data/Faber2026/dsa110 python batch_codetection_data.py [out-dir]

Default out-dir: ../figures/prototypes/codetection (Faber2026, gitignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
MS_ROOT = REPO.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scattering"))

from flits.batch.codetection_data import BURST_ORDER, load_codetection_bands  # noqa: E402
from flits.batch.codetection_plots import plot_codetection_observations  # noqa: E402


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (
        ("png", dict(dpi=200)),
        ("pdf", dict(bbox_inches="tight")),
        ("svg", dict(bbox_inches="tight")),
    ):
        fp = out_dir / f"{stem}.{ext}"
        fig.savefig(fp, **kw)
        print(f"wrote {fp}")


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else MS_ROOT / "figures/prototypes/codetection"
    ok, fail = [], []
    for burst in BURST_ORDER:
        try:
            bands = load_codetection_bands(burst)
            fig = plot_codetection_observations(
                bands,
                per_band_scale=True,
                gap_label=False,
                figsize=(4.8, 4.9),
            )
            _save(fig, out_dir, f"{burst}_codetection")
            plt.close(fig)
            ok.append(burst)
        except Exception as exc:
            print(f"FAIL {burst}: {exc}", file=sys.stderr)
            fail.append(burst)
    print(f"done: {len(ok)}/{len(BURST_ORDER)} bursts → {out_dir}")
    if fail:
        raise SystemExit(f"failed: {', '.join(fail)}")


if __name__ == "__main__":
    main()
