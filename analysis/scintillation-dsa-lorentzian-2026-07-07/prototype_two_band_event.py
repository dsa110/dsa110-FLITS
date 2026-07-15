#!/usr/bin/env python3
"""PROTOTYPE — one-event CHIME+DSA gamma/ACF publication tile.

Question: can one compact event tile show the two-band gamma measurements and
representative ACF fits clearly enough to scale to a 6x2 all-event figure?

This is a diagnostic layout experiment, not a qualified bandwidth measurement
or a manuscript figure producer. In particular, a CHIME ``diagnostic_only``
result remains diagnostic in the plot and in the returned metadata.

Run from the FLITS root:
  NUMBA_DISABLE_JIT=1 python \
    analysis/scintillation-dsa-lorentzian-2026-07-07/prototype_two_band_event.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_OUTPUT_DIR = Path("/tmp/two-band-event-prototype")
FIGURE_OUTPUT = Path("/tmp/freya-two-band-scintillation-prototype.png")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("FLITS_ROOT", str(Path.home() / "Data/Faber2026/dsa110"))

spec = importlib.util.spec_from_file_location("scint_driver", HERE / "run_dsa_lorentzian_fits.py")
drv = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(drv)


def run_band(burst: str, band: str, *, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path("scintillation/configs/bursts") / f"{burst}_{band}.yaml"
    loaded = drv.config_mod.load_config(config_path)
    base = drv._config_for_fresh_acf(loaded, output_dir=output_dir)
    candidates = []
    payloads = {}
    for count in drv.SUBBAND_CANDIDATES:
        result, plots = drv._fit_prepared_config(
            drv._config_with_subband_count(base, count),
            config_path,
            output_dir=output_dir,
            max_components=3,
        )
        result["requested_num_subbands"] = count
        candidates.append(result)
        payloads[count] = plots
    result, selection = drv._select_subband_candidate(candidates)
    result["subband_selection"] = selection
    return result, payloads[int(result["requested_num_subbands"])]


def component_rows(result, band):
    rows = []
    for subband in result["subbands"]:
        for component in subband["selected_components"]:
            width = float(component.get("dnu_mhz", np.nan))
            if np.isfinite(width) and width > 0:
                rows.append(
                    {
                        "subband": int(subband["index"]),
                        "center_freq_mhz": float(subband["center_freq_mhz"]),
                        "dnu_mhz": width,
                        "dnu_err_mhz": float(component.get("dnu_err", np.nan)),
                        "usable": not component.get("quality_flags"),
                        "band": band,
                    }
                )
    return drv._assign_gamma_tracks(rows)


def representative(payloads, *, count: int = 2):
    if len(payloads) < count:
        raise ValueError(f"need at least {count} ACF payloads, got {len(payloads)}")
    centers = np.asarray([p["summary"]["center_freq_mhz"] for p in payloads], dtype=float)
    median = float(np.median(centers))
    indices = sorted(
        range(len(payloads)), key=lambda index: (abs(centers[index] - median), centers[index])
    )
    return [payloads[index] for index in indices[:count]]


def plot_acf(ax, payload, *, color, band):
    lags = np.asarray(payload["lags"], dtype=float)
    acf = np.asarray(payload["acf"], dtype=float)
    err = np.asarray(payload["err"], dtype=float)
    fit_range = min(float(payload["summary"]["fit_range_mhz"]), 12.0)
    keep = np.isfinite(lags) & np.isfinite(acf) & (np.abs(lags) <= fit_range) & (lags != 0)
    idx = drv._decimated_indices(keep, max_points=350)
    order = idx[np.argsort(lags[idx])]
    err_keep = keep & np.isfinite(err) & (err > 0)
    err_idx = drv._decimated_indices(err_keep, max_points=350)
    ax.errorbar(lags[err_idx], acf[err_idx], yerr=err[err_idx], fmt="none", ecolor="0.85", lw=0.45)
    ax.plot(lags[order], acf[order], color=color, lw=0.9)
    xfit = np.linspace(-fit_range, fit_range, 700)
    ax.plot(xfit, drv._model_curve(xfit, payload["fit"]), color="black", lw=1.4)
    center = float(payload["summary"]["center_freq_mhz"])
    ax.set_title(rf"{band}, $\nu={center:.0f}$ MHz", loc="left", fontsize=7.5)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlim(-fit_range, fit_range)
    ax.set_ylabel(r"ACF ($m^2$)")
    ax.tick_params(direction="in", top=True, right=True)


def build_figure(*, dsa, dsa_payloads, chime, chime_payloads):
    """Build the Freya diagnostic tile from prepared two-band fit payloads."""
    chime_diagnostic = chime.get("measurement_status") != "measurement"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
        }
    )
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.05, 1.35], hspace=0.08, wspace=0.15)
    ax_gamma = fig.add_subplot(gs[:, 0])
    acf_axes = [fig.add_subplot(gs[index, 1]) for index in range(4)]

    styles = {
        "chime": {"color": "#D55E00", "label": "CHIME", "offset": -5.0},
        "dsa": {"color": "#0072B2", "label": "DSA-110", "offset": 5.0},
    }
    all_rows = component_rows(chime, "chime") + component_rows(dsa, "dsa")
    for band in ("chime", "dsa"):
        for row in [item for item in all_rows if item["band"] == band]:
            style = styles[band]
            marker = "o" if row["gamma_track"] == 1 else "s"
            filled = row["usable"] and not (band == "chime" and chime_diagnostic)
            ax_gamma.errorbar(
                row["center_freq_mhz"] + style["offset"],
                row["dnu_mhz"],
                yerr=row["dnu_err_mhz"] if np.isfinite(row["dnu_err_mhz"]) else None,
                fmt=marker,
                ms=5,
                mfc=style["color"] if filled else "none",
                mec=style["color"],
                ecolor=style["color"],
                alpha=0.95 if row["usable"] else 0.4,
                capsize=2,
            )

    diagnostic_fit = drv._fit_gamma_power_law(all_rows)
    if diagnostic_fit is not None:
        frequency = np.linspace(420.0, 1500.0, 500)
        log_frequency = np.log(frequency / diagnostic_fit["nu_ref_mhz"])
        design = np.column_stack((np.ones_like(log_frequency), log_frequency))
        log_curve = (
            np.log(diagnostic_fit["gamma_ref_mhz"]) + diagnostic_fit["alpha"] * log_frequency
        )
        log_sigma = np.sqrt(np.einsum("ij,jk,ik->i", design, diagnostic_fit["covariance"], design))
        curve = np.exp(log_curve)
        lower = np.exp(log_curve - log_sigma)
        upper = np.exp(log_curve + log_sigma)
        ax_gamma.fill_between(
            frequency,
            lower,
            upper,
            color="0.25",
            alpha=0.16,
            linewidth=0,
        )
        ax_gamma.plot(
            frequency,
            curve,
            color="0.25",
            ls="--" if chime_diagnostic else "-",
            lw=1.5,
        )
        label_frequency = 900.0
        label_gamma = (
            diagnostic_fit["gamma_ref_mhz"]
            * (label_frequency / diagnostic_fit["nu_ref_mhz"]) ** diagnostic_fit["alpha"]
        )
        ax_gamma.text(
            label_frequency,
            label_gamma * 1.45,
            rf"diagnostic $\alpha={diagnostic_fit['alpha']:.2f}$"
            if chime_diagnostic
            else rf"$\alpha={diagnostic_fit['alpha']:.2f}$",
            color="0.25",
            fontsize=8.5,
        )

    ax_gamma.set_yscale("log")
    ax_gamma.set_xlim(390, 1560)
    ax_gamma.set_xlabel(r"Observing frequency, $\nu$ (MHz)")
    ax_gamma.set_ylabel(r"$\Delta\nu_d$ (MHz)")
    ax_gamma.tick_params(direction="in", top=True, right=True)
    ax_gamma.set_title("FRB 20230325A", loc="left", fontsize=9)
    if chime_diagnostic:
        ax_gamma.text(
            0.03,
            0.02,
            "Diagnostic layout only\nnot a qualified measurement",
            transform=ax_gamma.transAxes,
            color="#A33A2A",
            fontsize=7.5,
            va="bottom",
        )

    all_acfs = [
        *(("CHIME", "#D55E00", payload) for payload in representative(chime_payloads)),
        *(("DSA-110", "#0072B2", payload) for payload in representative(dsa_payloads)),
    ]
    for index, (band, color, payload) in enumerate(all_acfs):
        plot_acf(acf_axes[index], payload, color=color, band=band)
        if index < len(all_acfs) - 1:
            acf_axes[index].tick_params(labelbottom=False)
    acf_axes[-1].set_xlabel("Frequency lag (MHz)", fontsize=8.0)

    metadata = {
        "status": "diagnostic_only" if chime_diagnostic else "measurement",
        "chime_measurement_status": chime.get("measurement_status", "missing"),
        "included_diagnostic_chime": chime_diagnostic,
        "acf_panel_count": len(all_acfs),
        "diagnostic_fit": diagnostic_fit,
    }
    return fig, metadata


def main():
    burst = "freya"
    dsa, dsa_payloads = run_band(burst, "dsa", output_dir=RUN_OUTPUT_DIR)
    chime, chime_payloads = run_band(burst, "chime", output_dir=RUN_OUTPUT_DIR)
    fig, metadata = build_figure(
        dsa=dsa,
        dsa_payloads=dsa_payloads,
        chime=chime,
        chime_payloads=chime_payloads,
    )

    out = FIGURE_OUTPUT
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(out)
    print(
        "CHIME status:",
        chime.get("measurement_status"),
        chime.get("artifact_control", {}).get("failed_checks"),
    )
    diagnostic_fit = metadata["diagnostic_fit"]
    print("Figure status:", metadata["status"])
    print("Diagnostic gamma_1 scaling index:", diagnostic_fit["alpha"] if diagnostic_fit else None)


if __name__ == "__main__":
    main()
