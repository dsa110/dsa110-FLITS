#!/usr/bin/env python3
"""Quick C1 all-pairs cross-ACF diagnostic for Freya CHIME.

Uses the same retained CHIME product as the B4 four-stream validation, but
preserves individual on-pulse time samples and forms every distinct-time
polarization cross-product instead of collapsing to four spectra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
B4_ROOT = ROOT / "analysis" / "chime-recovery-2026-07-12"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(B4_ROOT))

from scintillation.scint_analysis.cross_acf import (  # noqa: E402
    all_pairs_cross_acf,
    fit_cross_lorentzian,
)
import validate_freya_highband_crossacf as b4  # noqa: E402

DATA = Path.home() / "Data/Faber2026/dsa110/upchan_codetections/crossacf-2026-07-14"
DEFAULT_OUTPUT = Path(__file__).parent / "diagnostic"


def _acf_summary(cross, channel_width_mhz, label):
    fit = fit_cross_lorentzian(
        cross,
        channel_width_mhz=channel_width_mhz,
        first_lag_bin=b4.FIRST_LAG_BIN,
        fit_max_mhz=b4.FIT_MAXIMA_MHZ[-1],
        block_length=b4.CHANNELS_PER_COARSE,
    )
    return {
        "label": label,
        "fit": fit,
        "max_abs_z": float(np.nanmax(np.abs(cross.acf / cross.error)))
        if np.isfinite(cross.error).any()
        else None,
        "acf_mean": float(np.nanmean(cross.acf)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pol0", type=Path, default=b4.DEFAULT_POL0)
    parser.add_argument("--pol1", type=Path, default=b4.DEFAULT_POL1)
    parser.add_argument("--stokes", type=Path, default=b4.DEFAULT_STOKES)
    parser.add_argument("--frequencies", type=Path, default=b4.DEFAULT_FREQUENCIES)
    parser.add_argument("--time0-metadata", type=Path, default=b4.DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frequencies_full = np.load(args.frequencies)
    select = (frequencies_full >= b4.BAND_MHZ[0]) & (frequencies_full <= b4.BAND_MHZ[1])
    frequencies = np.asarray(frequencies_full[select], dtype=float)
    power = [
        np.asarray(np.load(path, mmap_mode="r")[select], dtype=float)
        for path in (args.pol0, args.pol1)
    ]

    metadata = json.loads(args.time0_metadata.read_text())
    coarse = np.asarray(metadata["freq_mhz"], dtype=float)
    target = b4.load_chime_target("freya")
    dt_s = 2.56e-6 * 2 * int(target["upchannel_factor"])
    offsets = b4.coarse_alignment_offsets(
        coarse,
        np.asarray(metadata["fpga_count"]),
        delta_time_s=float(metadata["delta_time"]),
        dm=float(target["dm"]),
        dt_s=dt_s,
    )
    good_channels = b4._channel_mask(power, frequencies)
    dynamic = [
        b4._build_polarization_product(item, frequencies, coarse, offsets, good_channels)[0]
        for item in power
    ]
    baselines = [b4._row_nanmean(item[:, b4.OFF_PULSE[0] : b4.OFF_PULSE[1]]) for item in dynamic]
    # residual = baseline-subtracted intensity, whose per-sample mean is the
    # burst amplitude for on-pulse samples.
    residuals = [item - baselines[p][:, None] for p, item in enumerate(dynamic)]

    parent = np.argmin(np.abs(frequencies[:, None] - coarse[None, :]), axis=1)
    channel_width = float(np.nanmedian(np.diff(frequencies)))

    on_pulse = [item[:, b4.BURST_WINDOW[0] : b4.BURST_WINDOW[1]] for item in residuals]
    off_start = b4.OFF_PULSE[0]
    off_width = b4.BURST_WINDOW[1] - b4.BURST_WINDOW[0]
    off_pulse = [item[:, off_start : off_start + off_width] for item in residuals]

    # Use the on-pulse envelope per fold as the normalization so the same
    # amplitude scale applies to off-pulse nulls.
    on_norms = [np.nanmean(item, axis=0) for item in on_pulse]

    on_cross = all_pairs_cross_acf(
        on_pulse,
        parent,
        max_lag_bins=b4.MAX_LAG_BINS,
        exclude_same_time=True,
        normalizations=on_norms,
    )
    off_cross = all_pairs_cross_acf(
        off_pulse,
        parent,
        max_lag_bins=b4.MAX_LAG_BINS,
        exclude_same_time=True,
        normalizations=on_norms,
    )
    on_corrected = b4._remove_instrument_template(on_cross, [off_cross])

    on_summary = _acf_summary(on_corrected, channel_width, "on-pulse")
    off_summary = _acf_summary(off_cross, channel_width, "off-pulse")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, cross, summary in [
        ("onpulse", on_cross, on_summary),
        ("offpulse", off_cross, off_summary),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        lags = cross.lag_bins * channel_width * 1e3
        ax.errorbar(lags, cross.acf, yerr=cross.error, fmt=".", ms=4, alpha=0.8)
        if summary["fit"] is not None:
            ax.plot(
                np.asarray(summary["fit"]["fit_lags_mhz"]) * 1e3,
                summary["fit"]["model_acf"],
                lw=2,
                label="Lorentzian fit",
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set(
            xlabel="Frequency lag (kHz)",
            ylabel="Cross covariance",
            title=f"C1 all-pairs {label} ACF",
        )
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.savefig(args.output_dir / f"freya_c1_{label}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    result = {
        "experiment": "c1-allpairs-crossgp diagnostic",
        "band_mhz": list(b4.BAND_MHZ),
        "channel_width_mhz": channel_width,
        "n_selected_channels": int(select.sum()),
        "n_good_channels": int(good_channels.sum()),
        "on_pulse": on_summary,
        "off_pulse": off_summary,
    }
    (args.output_dir / "diagnostic.json").write_text(
        json.dumps(b4._jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(b4._jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
