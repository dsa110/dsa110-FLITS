#!/usr/bin/env python
"""Assemble the CHIME sub-band profile comparison manuscript figure.

Lays two bursts side by side across ``nsub`` CHIME sub-bands. The per-band
profiling (baseline, on-pulse, w_rms / w_tail) is reused verbatim from
within_chime_test.py so this figure and that diagnostic cannot drift. Vector
output (PDF for LaTeX + SVG, plus a PNG for the gallery/review gate) via _figsave,
written into the ``chime_subband/`` manuscript section (build_manuscript.py picks
it up). Input configs/data resolve under ``$FLITS_RUNS``; override the figure
output dir with ``$FLITS_FIGOUT``.

  python analysis/scattering-refit-2026-06/chime_subband_compare.py [burst1 burst2 ...] [--nsub N]
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figsave import save_fig
from within_chime_test import RUNS, onpulse_widths, prepare

BURSTS = ["johndoeII", "wilhelm"]


def main():
    ap = argparse.ArgumentParser(description="CHIME sub-band profile comparison figure")
    ap.add_argument("bursts", nargs="*", help="burst nicknames (default: johndoeII wilhelm)")
    ap.add_argument("--nsub", type=int, default=4, help="number of CHIME sub-bands")
    a = ap.parse_args()
    bursts = a.bursts or BURSTS
    nsub = a.nsub
    datadir = f"{RUNS}/data/joint"  # BurstDataset scratch (diagnostics)
    out = os.environ.get("FLITS_FIGOUT") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "chime_subband"
    )
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(
        nsub, len(bursts), figsize=(4.2 * len(bursts), 2.0 * nsub), sharex=True, squeeze=False
    )
    for j, b in enumerate(bursts):
        m = prepare(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime", datadir)
        freq, t, data = m.freq, m.time, m.data  # (nch, ntime), GHz ascending
        edges = np.linspace(0, freq.size, nsub + 1).astype(int)
        for i in range(nsub):
            sl = slice(edges[i], edges[i + 1])
            w_rms, w_tail, p = onpulse_widths(np.nansum(data[sl], axis=0), t)
            ax = axes[i][j]
            ax.plot(t, p, "k", lw=0.8)
            ax.set_title(
                f"{b} {freq[sl][0]:.3f}-{freq[sl][-1]:.3f} GHz  "
                f"w_rms={w_rms:.3f} w_tail={w_tail:.3f} ms",
                fontsize=8,
            )
        axes[-1][j].set_xlabel("time (ms)")
    fig.tight_layout()
    print(f"wrote {save_fig(fig, f'{out}/chime_subband_compare', dpi=110)}")


if __name__ == "__main__":
    main()
