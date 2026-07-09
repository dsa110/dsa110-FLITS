#!/usr/bin/env python
"""Render codetection-style data/model/residual figures from jointmodel NPZ dumps.

The input NPZ files are produced by ``dump_jointmodel.py`` and contain the
per-band data, recovered best-fit model, axes, noise, and validity masks. This
script makes one manuscript-style prototype per citable beta-campaign row:
observed data, recovered 2-D model, and whitened residual panels.
Outputs default to the manuscript repo under
``figures/prototypes/jointmodel_pair/``. That directory is intentionally
gitignored as a local prototype area; promote selected finals into ``figures/``
when they should be tracked.

  FLITS_RUNS=~/Developer/scratch/flits-local-runs \\
    conda run -n flits python plot_jointmodel_pair.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MS_ROOT = REPO.parent
DEFAULT_RUNS = Path(os.environ.get("FLITS_RUNS", "/Users/jakobfaber/Developer/scratch/flits-local-runs"))

KNOWN_MULTIPLICITY_FLAGS = {
    "hamilton": "CHIME data show two components; current sharedzeta fit is C1D1 and misses the leading CHIME component.",
    "whitney_fine": "DSA data show two narrow components; current C2D2 fit is degenerate and misses the second visible DSA component.",
}

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scattering"))

from flits.batch.codetection_data import (  # noqa: E402
    chime_toa_shift_ms,
    crop_bands_to_subburst_window,
    toa_offset_ms,
)
from flits.batch.codetection_plots import BandSpectrum, plot_codetection  # noqa: E402


def citable_rows() -> list[dict]:
    verdicts = json.loads((REPO / "analysis/beta_campaign/beta_campaign_verdicts.json").read_text())
    return [row for row in verdicts["rows"] if row["final"] != "FAIL"]


def jointmodel_npz(runs: Path, burst: str, suffix: str) -> Path:
    return runs / "data/joint" / f"{burst}_jointmodel{suffix}.npz"


def _band(z, band: str) -> BandSpectrum:
    label = "CHIME/FRB" if band == "C" else "DSA-110"
    return BandSpectrum(
        freq_mhz=np.asarray(z[f"freq{band}"], float) * 1e3,
        time_ms=np.asarray(z[f"time{band}"], float),
        data=np.asarray(z[f"data{band}"], float),
        model=np.asarray(z[f"model{band}"], float),
        sigma=np.asarray(z[f"noise{band}"], float),
        label=label,
        channel_valid=np.asarray(z[f"valid{band}"], bool),
    )


def _aligned_bands(z, burst: str) -> list[BandSpectrum]:
    chime = _band(z, "C")
    dsa = _band(z, "D")
    lookup = burst.removesuffix("_fine")
    offset = toa_offset_ms(lookup)
    if offset is not None:
        shift_c = chime_toa_shift_ms(dsa, chime, offset)
        chime = BandSpectrum(
            freq_mhz=chime.freq_mhz,
            time_ms=chime.time_ms + shift_c,
            data=chime.data,
            model=chime.model,
            sigma=chime.sigma,
            label=chime.label,
            channel_valid=chime.channel_valid,
        )
    return crop_bands_to_subburst_window([chime, dsa], center=True)


def plot_pair(row: dict, runs: Path, out_dir: Path, *, dpi: int = 200) -> Path:
    burst = row["burst"]
    suffix = row["suffix"]
    fp = jointmodel_npz(runs, burst, suffix)
    if not fp.exists():
        raise FileNotFoundError(
            f"missing {fp}; run: FLITS_REPO={REPO} FLITS_RUNS={runs} "
            f"python {HERE / 'dump_jointmodel.py'} {burst} {suffix}"
        )

    z = np.load(fp, allow_pickle=True)
    bands = _aligned_bands(z, burst)
    fig = plot_codetection(
        bands,
        columns=("data", "model", "resid"),
        show_model_on_data=False,
        per_band_scale=True,
        gap_label=False,
        figsize=(12.4, 4.9),
        band_labels=False,
        show_column_titles=False,
        per_band_marginals=True,
        title=None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{burst.removesuffix('_fine')}_jointmodel_pair"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_RUNS,
        help="FLITS runs root containing data/joint jointmodel NPZ files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MS_ROOT / "figures/prototypes/jointmodel_pair",
        help="output directory for PNG/PDF/SVG figures",
    )
    parser.add_argument("--burst", action="append", help="optional citable burst nickname filter")
    args = parser.parse_args()

    want = set(args.burst or [])
    rows = [row for row in citable_rows() if not want or row["burst"] in want or row["burst"].removesuffix("_fine") in want]
    if not rows:
        raise SystemExit("no citable rows matched")

    written = []
    for row in rows:
        if row["burst"] in KNOWN_MULTIPLICITY_FLAGS:
            print(f"WARNING {row['burst']}: {KNOWN_MULTIPLICITY_FLAGS[row['burst']]}", file=sys.stderr)
        written.append(plot_pair(row, args.runs, args.out_dir))
    print(f"rendered {len(written)} jointmodel pair figure(s) to {args.out_dir}")
    for fp in written:
        print(fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
