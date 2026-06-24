#!/usr/bin/env python3
"""Convert a recovered CHIME upchannelized Stokes-I .npy into the .npz the FLITS scint pipeline reads.

The pipeline's DynamicSpectrum.from_numpy_file expects keys power_2d / frequencies_mhz / times_s, and
it does `np.flip(power_2d, axis=0)` on load then re-flips to ascending iff frequencies are descending.
Net: storing power_2d freq-DESCENDING + frequencies_mhz ASCENDING lands aligned-ascending. The worker
already wrote <name>_chime_upchan.npy ascending, so we flip the rows back to descending here.

  python npy_to_npz.py casey --U 16    # -> scintillation/data/casey_chime.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHIME_NATIVE_DT_S = 2.56e-6  # time bin after upchannelizing = fftsize * native = 2*U * 2.56us
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("name")
    p.add_argument(
        "--U", type=int, required=True, help="upchannelization factor used (dt = 2*U*2.56us)"
    )
    p.add_argument("--products", default=str(HERE / "products"))
    p.add_argument("--out", default=None)
    p.add_argument(
        "--fmin", type=float, default=None, help="keep only freq >= fmin MHz (high-band focus)"
    )
    p.add_argument("--fmax", type=float, default=None, help="keep only freq <= fmax MHz")
    args = p.parse_args(argv)

    prod = Path(args.products)
    stokes = np.load(prod / f"{args.name}_chime_upchan.npy")  # (n_freq, n_time), freq ascending
    freq = np.load(prod / f"{args.name}_chime_freq.npy")  # MHz, ascending
    assert stokes.shape[0] == freq.size, f"freq/spec mismatch {stokes.shape} vs {freq.shape}"

    sliced = args.fmin is not None or args.fmax is not None
    if sliced:
        keep = np.ones(freq.size, bool)
        if args.fmin is not None:
            keep &= freq >= args.fmin
        if args.fmax is not None:
            keep &= freq <= args.fmax
        assert keep.sum() >= 256, f"slice keeps only {keep.sum()} fine channels"
        stokes, freq = stokes[keep], freq[keep]

    dt = 2 * args.U * CHIME_NATIVE_DT_S
    times_s = np.arange(stokes.shape[1]) * dt

    suffix = "_chime_hi" if sliced else "_chime"
    out = (
        Path(args.out) if args.out else REPO / "scintillation" / "data" / f"{args.name}{suffix}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # power_2d freq-descending + frequencies_mhz ascending -> loader aligns to ascending.
    np.savez(out, power_2d=stokes[::-1, :], frequencies_mhz=freq, times_s=times_s)

    # ponytail self-check: reload through the pipeline's own loader and confirm alignment + shape.
    import sys

    sys.path.insert(0, str(REPO / "scintillation" / "scint_analysis"))
    from core import DynamicSpectrum  # noqa: PLC0415

    ds = DynamicSpectrum.from_numpy_file(str(out))
    assert ds.power.shape == stokes.shape, f"roundtrip shape {ds.power.shape} != {stokes.shape}"
    assert ds.frequencies[0] < ds.frequencies[-1], "frequencies not ascending after load"
    # the loaded power row 0 (lowest freq) must equal our ascending stokes row 0
    lo = ds.power[0].filled(np.nan) if np.ma.isMaskedArray(ds.power) else ds.power[0]
    assert np.allclose(np.nan_to_num(lo), np.nan_to_num(stokes[0]), atol=0), "freq/power misaligned"
    print(
        f"[{args.name}] {out.name}: power {ds.power.shape}, "
        f"{ds.frequencies[0]:.1f}-{ds.frequencies[-1]:.1f} MHz, dt={dt * 1e3:.4f} ms, n_time={times_s.size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
