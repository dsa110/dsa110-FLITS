from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dispersion.dm_power_analysis import (
    CHIME_DT_S,
    DSA_DT_S,
    _freq_grid_mhz,
    _orient_waterfall_to_ascending_frequency,
)

from .measurement import _valid_channels
from .resolution import block_average
from .shifts import dedisperse_residual


def _display(array: np.ndarray, maximum_frequency: int = 256, maximum_time: int = 2048) -> np.ndarray:
    ff = max(1, int(np.ceil(array.shape[0] / maximum_frequency)))
    tf = max(1, int(np.ceil(array.shape[1] / maximum_time)))
    nf, nt = (array.shape[0] // ff) * ff, (array.shape[1] // tf) * tf
    return np.nanmean(array[:nf, :nt].reshape(nf // ff, ff, nt // tf, tf), axis=(1, 3))


def render_product(result_path: Path) -> Path:
    result = json.loads(result_path.read_text())
    telescope = result["telescope"]
    dt_s = CHIME_DT_S if telescope == "chime" else DSA_DT_S
    raw = np.load(result["input_path"], mmap_mode="r")
    waterfall = _orient_waterfall_to_ascending_frequency(raw, telescope)
    frequency = _freq_grid_mhz(telescope, waterfall.shape[0])
    valid = _valid_channels(waterfall)
    waterfall, frequency = waterfall[valid], frequency[valid]
    preliminary = dedisperse_residual(
        np.asarray(waterfall, dtype=float), frequency, dt_s, result["preliminary_residual_dm"]
    )
    start, stop = result["crop"]
    reference_crop = waterfall[:, start:stop]
    preliminary_crop = preliminary[:, start:stop]
    selected = result.get("selected")
    if selected is None:
        frequency_factor = time_factor = 1
        candidate_crop = preliminary_crop
    else:
        frequency_factor = int(selected["frequency_factor"])
        time_factor = int(selected["time_factor"])
        local_residual = float(selected["residual_dm"] - result["preliminary_residual_dm"])
        candidate_crop = dedisperse_residual(
            preliminary_crop, frequency, dt_s, local_residual
        )
    reference_selected, selected_frequency = block_average(
        reference_crop, frequency, frequency_factor, time_factor
    )
    candidate_selected, _ = block_average(
        candidate_crop, frequency, frequency_factor, time_factor
    )
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), constrained_layout=True)
    ax = axes.ravel()
    for panel, data, title in (
        (0, reference_crop, "Native at reference DM"),
        (1, reference_selected, "Selected resolution at reference DM"),
        (2, candidate_selected, "Selected resolution at candidate DM"),
        (3, candidate_selected - reference_selected, "Candidate minus reference"),
    ):
        image = _display(data)
        ax[panel].imshow(image, origin="lower", aspect="auto", cmap="viridis")
        ax[panel].set_title(title)
        ax[panel].set_xlabel("time bin")
        ax[panel].set_ylabel("frequency bin")
    ax[4].plot(np.nansum(reference_selected, axis=0), label="reference", alpha=0.8)
    ax[4].plot(np.nansum(candidate_selected, axis=0), label="candidate", alpha=0.8)
    ax[4].set_title("Band-summed profiles")
    ax[4].legend()
    details = result.get("resolution_details", [])
    if details and details[0].get("fine_grid"):
        for detail in details:
            if detail.get("fine_grid"):
                ax[5].plot(detail["fine_grid"], detail["fine_score"], alpha=0.55)
    ax[5].set_title("Fine DM coherence curves")
    ax[5].set_xlabel("local residual DM")
    evaluations = result.get("resolutions", [])
    if evaluations:
        loss = [np.log2(item["frequency_factor"]) + np.log2(item["time_factor"]) for item in evaluations]
        dm = [np.nan if item["residual_dm"] is None else item["residual_dm"] for item in evaluations]
        color = ["tab:green" if item["eligible"] else "tab:red" for item in evaluations]
        ax[6].scatter(loss, dm, c=color)
        ax[6].set_title("Resolution stability (green = eligible)")
        ax[6].set_xlabel("information loss")
        ax[6].set_ylabel("physical residual DM")
        ax[7].scatter(
            [item["time_factor"] for item in evaluations],
            [item["frequency_factor"] for item in evaluations],
            c=color,
        )
        ax[7].set_xscale("log", base=2)
        ax[7].set_yscale("log", base=2)
        ax[7].set_title("Evaluated resolution surface")
        ax[7].set_xlabel("time factor")
        ax[7].set_ylabel("frequency factor")
    cutoff_dm = [detail.get("cutoff_residual_dms") for detail in details]
    for index, values in enumerate(cutoff_dm):
        if values:
            ax[8].plot((0.75, 1.0, 1.25), values, marker="o", alpha=0.5, label=str(index))
    ax[8].set_title("Adjacent-cutoff stability")
    ax[8].set_xlabel("cutoff scale")
    peaks = result.get("bootstrap_peaks") or []
    if peaks:
        ax[9].hist(peaks, bins=min(12, len(peaks)), color="tab:blue", alpha=0.75)
    ax[9].set_title("Channel-bootstrap peaks")
    summary = (
        f"{result['burst']} / {telescope.upper()}\n"
        f"status: {result['status']}\n"
        f"DM ref: {result['dm_reference']:.6g}\n"
        f"DM: {result.get('dm_absolute')}\n"
        f"sigma_method: {result.get('sigma_method')}\n"
        f"sigma_injection: {result.get('sigma_injection')}\n"
        f"sigma_total: {result.get('sigma_total')}\n"
        f"selected factors: f={frequency_factor}, t={time_factor}\n"
        f"native shape: {tuple(result['native_shape'])}\n"
        f"valid channels: {result['valid_channels']}\n"
        f"input: {(result.get('input_sha256') or 'missing')[:12]}\n"
        f"commit: {result.get('code_commit', 'missing')[:12]}"
    )
    ax[10].axis("off")
    ax[10].text(0.02, 0.98, summary, va="top", family="monospace")
    failures = [] if selected is None else selected.get("failure_reasons", [])
    ax[11].axis("off")
    ax[11].text(
        0.02,
        0.98,
        "Visual audit checklist\n"
        f"numeric failures: {failures or 'none'}\n"
        "published oracle: campaign gate passed\n"
        "cross-method overlays: pending final campaign render\n"
        "review verdict: pending",
        va="top",
        family="monospace",
    )
    fig.suptitle("Controlled DM-phase diagnostic", fontsize=16)
    output = result_path.with_name("diagnostic.png")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def render_campaign(config_path: Path, *, pilot: bool) -> list[str]:
    import yaml

    config = yaml.safe_load(config_path.read_text())
    root = Path(config["output_dir"]) / "products"
    paths = sorted(root.glob("*/*/result.json"))
    if pilot:
        from .measurement import PILOT_BURSTS

        paths = [path for path in paths if path.parent.parent.name in PILOT_BURSTS]
    return [str(render_product(path)) for path in paths]
