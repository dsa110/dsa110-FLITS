#!/usr/bin/env python3
"""Build paired corrected/uncorrected CHIME scintillation NPZ products."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scintillation.scint_analysis.chime_product import (  # noqa: E402
    ChimeProductConfig,
    build_chime_products,
    burst_track_mask,
    coarse_alignment_offsets,
    load_chime_target,
    write_chime_products,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--frequencies", type=Path, required=True)
    parser.add_argument("--time0-metadata", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--aligned-burst-bin", type=int, required=True)
    parser.add_argument("--burst-half-width", type=int, required=True)
    parser.add_argument("--off-pulse", type=int, nargs=2, required=True, metavar=("START", "STOP"))
    parser.add_argument("--rfi-mask", type=Path)
    parser.add_argument("--rank", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    target = load_chime_target(args.target)
    power = np.load(args.power)
    frequencies = np.load(args.frequencies)
    metadata = json.loads(args.time0_metadata.read_text())
    coarse = np.asarray(metadata["freq_mhz"], dtype=float)
    dt_s = 2.56e-6 * 2 * int(target["upchannel_factor"])
    offsets = coarse_alignment_offsets(
        coarse,
        np.asarray(metadata["fpga_count"]),
        delta_time_s=float(metadata["delta_time"]),
        dm=float(target["dm"]),
        dt_s=dt_s,
    )
    parent = np.argmin(np.abs(frequencies[:, None] - coarse[None, :]), axis=1)
    mask = burst_track_mask(
        n_channels=power.shape[0],
        n_times=power.shape[1],
        channel_offsets=offsets[parent],
        aligned_center_bin=args.aligned_burst_bin,
        half_width_bins=args.burst_half_width,
    )
    result = build_chime_products(
        power,
        frequencies,
        coarse,
        coarse_offsets=offsets,
        burst_mask=mask,
        rfi_mask=np.load(args.rfi_mask) if args.rfi_mask else None,
        config=ChimeProductConfig(
            target=args.target,
            dm=float(target["dm"]),
            upchannel_factor=int(target["upchannel_factor"]),
            dt_s=dt_s,
            off_pulse=tuple(args.off_pulse),
            guard_bins=1,
            correction_rank=args.rank,
            aligned_burst_bin=args.aligned_burst_bin,
            burst_half_width_bins=args.burst_half_width,
        ),
    )
    paths = write_chime_products(
        result,
        args.output_prefix,
        input_paths=[args.power, args.frequencies, args.time0_metadata]
        + ([args.rfi_mask] if args.rfi_mask else []),
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
