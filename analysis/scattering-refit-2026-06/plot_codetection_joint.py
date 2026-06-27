#!/usr/bin/env python
"""Joint npz → ``plot_codetection`` (native freq, TOA-aligned).

  python plot_codetection_joint.py <npz> <out-dir> [tag]

Writes {tag}_codetection.{pdf,svg,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flits.batch.codetection_joint import spectra_from_joint_bands
from flits.batch.codetection_plots import plot_codetection
from plot_jointmodel_prototypes import _bands, _load, _suptitle


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (
        ("png", dict(dpi=150)),
        ("pdf", dict(bbox_inches="tight")),
        ("svg", dict(bbox_inches="tight")),
    ):
        fp = out_dir / f"{stem}.{ext}"
        fig.savefig(fp, **kw)
        print(f"wrote {fp}")


def main():
    npz_fp = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else npz_fp.name.split("_jointmodel")[0]
    z, meta = _load(npz_fp)
    bands, xlim, offset_ms, align_note = _bands(z, npz_fp, meta["burst"])
    title = _suptitle(meta, offset_ms=offset_ms, align_note=align_note)
    spectra = spectra_from_joint_bands(bands, xlim=xlim)
    fig = plot_codetection(
        spectra,
        title=title,
        per_band_scale=True,
        figsize=(11.0, 5.0),
    )
    _save(fig, out_dir, f"{tag}_codetection")


if __name__ == "__main__":
    main()
