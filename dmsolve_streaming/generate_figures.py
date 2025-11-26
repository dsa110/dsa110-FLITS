#!/usr/bin/env python3
"""
Generate all figures for the StreamingDMEstimator documentation.
Uses consistent LaTeX-style fonts (STIX) throughout.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import from local module
from streaming_dm_estimator import (
    StreamingDMEstimator,
    generate_test_spectrum,
    K_DM,
    iterative_dm_estimate,
    channel_variance_clip,
)

# =============================================================================
# GLOBAL STYLE SETTINGS
# =============================================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIXGeneral", "Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12

# Common sizes
TITLE_SIZE = 14
LABEL_SIZE = 14
TICK_SIZE = 13
LEGEND_SIZE = 12


def setup_axes(ax, xlabel=None, ylabel=None, title=None):
    """Apply consistent styling to axes."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


# =============================================================================
# FIGURE 1: S/N vs Bias
# =============================================================================
def generate_fig1():
    """S/N vs bias for different weight powers."""
    print("Generating fig1_snr_vs_bias.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    snr_values = [3, 5, 7, 10, 15, 20, 30, 50]
    powers = [1, 2, 3, 4]
    n_trials = 100  # Match caption claim

    results = {p: {"bias_mean": [], "bias_std": []} for p in powers}

    for snr in snr_values:
        for p in powers:
            biases = []
            for _ in range(n_trials):
                ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)
                est = StreamingDMEstimator(freqs, times, weight_power=p)
                est.process_spectrum(ds)
                result = est.get_estimate(apply_correction=False)
                if result:
                    biases.append((result.dm - dm_true) / dm_true * 100)
            if biases:
                results[p]["bias_mean"].append(np.mean(biases))
                results[p]["bias_std"].append(np.std(biases))
            else:
                results[p]["bias_mean"].append(np.nan)
                results[p]["bias_std"].append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    markers = ["o", "s", "^", "D"]

    for i, p in enumerate(powers):
        means = np.array(results[p]["bias_mean"])
        stds = np.array(results[p]["bias_std"])
        ax.errorbar(
            snr_values,
            means,
            yerr=stds,
            fmt=markers[i] + "-",
            color=colors[i],
            label=f"p = {p}",
            linewidth=2,
            markersize=8,
            capsize=4,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvspan(0, 5, alpha=0.15, color="red", label="Unreliable (S/N < 5)")

    setup_axes(
        ax,
        xlabel="Signal-to-noise ratio (S/N)",
        ylabel="Bias (%)",
        title="DM estimation bias vs S/N",
    )
    ax.set_xlim(0, 55)
    ax.legend(fontsize=LEGEND_SIZE, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig1_snr_vs_bias.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 2: Batch vs Streaming
# =============================================================================
def generate_fig2():
    """Batch vs streaming mode comparison."""
    print("Generating fig2_batch_vs_streaming.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0
    n_trials = 100

    batch_results = []
    stream_results = []

    for _ in range(n_trials):
        ds = generate_test_spectrum(dm_true, 0.1, 0.005, 15, freqs, times)

        # Batch mode
        est_batch = StreamingDMEstimator(freqs, times, weight_power=3)
        est_batch.process_spectrum(ds)
        r_batch = est_batch.get_estimate()

        # Streaming mode (channel by channel)
        est_stream = StreamingDMEstimator(freqs, times, weight_power=3)
        for i, freq in enumerate(freqs):
            est_stream.process_channel(freq, times, ds[i, :])
        r_stream = est_stream.get_estimate()

        if r_batch and r_stream:
            batch_results.append(r_batch.dm)
            stream_results.append(r_stream.dm)

    batch_results = np.array(batch_results)
    stream_results = np.array(stream_results)
    diff = batch_results - stream_results

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(
        batch_results, stream_results, alpha=0.6, s=50, edgecolor="black", linewidth=0.5
    )
    lims = [
        min(batch_results.min(), stream_results.min()) - 5,
        max(batch_results.max(), stream_results.max()) + 5,
    ]
    ax1.plot(lims, lims, "r--", linewidth=2, label="Perfect agreement")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    setup_axes(
        ax1,
        xlabel="Batch mode DM (pc/cm³)",
        ylabel="Streaming mode DM (pc/cm³)",
        title="Batch vs streaming mode",
    )
    ax1.legend(fontsize=LEGEND_SIZE)

    # Histogram
    ax2 = axes[1]
    ax2.hist(diff, bins=30, edgecolor="black", alpha=0.7, color="#377eb8")
    ax2.axvline(0, color="red", linestyle="--", linewidth=2)
    ax2.axvline(
        np.mean(diff),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean = {np.mean(diff):.4f}",
    )
    setup_axes(
        ax2,
        xlabel="Difference (pc/cm³)",
        ylabel="Count",
        title=f"Difference distribution (std = {np.std(diff):.4f})",
    )
    ax2.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig("figures/fig2_batch_vs_streaming.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 3: Channel Ordering
# =============================================================================
def generate_fig3():
    """Convergence with different channel orderings."""
    print("Generating fig3_channel_ordering.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    ds = generate_test_spectrum(dm_true, 0.1, 0.005, 15, freqs, times)

    # Sequential order
    seq_estimates = []
    est_seq = StreamingDMEstimator(freqs, times, weight_power=3)
    for i in range(len(freqs)):
        est_seq.process_channel(freqs[i], times, ds[i, :])
        r = est_seq.get_estimate()
        seq_estimates.append(r.dm if r else np.nan)

    # Interleaved order (band edges first)
    n = len(freqs)
    interleaved_idx = []
    for i in range((n + 1) // 2):
        interleaved_idx.append(i)
        if n - 1 - i != i:
            interleaved_idx.append(n - 1 - i)

    int_estimates = []
    est_int = StreamingDMEstimator(freqs, times, weight_power=3)
    for i in interleaved_idx:
        est_int.process_channel(freqs[i], times, ds[i, :])
        r = est_int.get_estimate()
        int_estimates.append(r.dm if r else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    seq_err = np.abs(np.array(seq_estimates) - dm_true) / dm_true * 100
    int_err = np.abs(np.array(int_estimates) - dm_true) / dm_true * 100

    ax.plot(
        range(1, len(seq_err) + 1),
        seq_err,
        "b-",
        linewidth=2,
        label="Sequential order",
        marker="o",
        markersize=4,
    )
    ax.plot(
        range(1, len(int_err) + 1),
        int_err,
        "g-",
        linewidth=2,
        label="Interleaved order",
        marker="s",
        markersize=4,
    )

    ax.axhline(
        5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="5% threshold"
    )

    setup_axes(
        ax,
        xlabel="Number of channels processed",
        ylabel="|Error| (%)",
        title="Convergence speed: sequential vs interleaved",
    )
    ax.set_yscale("log")
    ax.set_ylim(0.01, 200)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("figures/fig3_channel_ordering.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 4: RFI Robustness
# =============================================================================
def generate_fig4():
    """RFI robustness with and without clipping."""
    print("Generating fig4_rfi_robustness.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    n_bad_channels = [0, 1, 2, 3, 5, 10, 15, 20]
    n_trials = 30

    results_raw = {"mean": [], "std": []}
    results_clip = {"mean": [], "std": []}

    for n_bad in n_bad_channels:
        biases_raw = []
        biases_clip = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, 15, freqs, times)

            # Add RFI to random channels
            if n_bad > 0:
                bad_idx = np.random.choice(len(freqs), n_bad, replace=False)
                ds[bad_idx, :] += np.random.randn(n_bad, len(times)) * 50

            # Raw (no clipping)
            est_raw = StreamingDMEstimator(freqs, times, weight_power=3)
            est_raw.process_spectrum(ds)
            r_raw = est_raw.get_estimate()

            # With clipping - mask indicates BAD channels
            ds_clip, bad_mask = channel_variance_clip(ds, sigma_clip=3.0)
            good_mask = ~bad_mask
            if good_mask.sum() >= 4:  # Need at least 4 good channels
                est_clip = StreamingDMEstimator(freqs[good_mask], times, weight_power=3)
                est_clip.process_spectrum(ds[good_mask, :])
                r_clip = est_clip.get_estimate()
            else:
                r_clip = None

            if r_raw:
                biases_raw.append((r_raw.dm - dm_true) / dm_true * 100)
            if r_clip:
                biases_clip.append((r_clip.dm - dm_true) / dm_true * 100)

        results_raw["mean"].append(np.mean(biases_raw) if biases_raw else np.nan)
        results_raw["std"].append(np.std(biases_raw) if biases_raw else np.nan)
        results_clip["mean"].append(np.mean(biases_clip) if biases_clip else np.nan)
        results_clip["std"].append(np.std(biases_clip) if biases_clip else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        n_bad_channels,
        results_raw["mean"],
        yerr=results_raw["std"],
        fmt="o-",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Without RFI clipping",
    )
    ax.errorbar(
        n_bad_channels,
        results_clip["mean"],
        yerr=results_clip["std"],
        fmt="s-",
        color="#4daf4a",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="With variance clipping",
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    setup_axes(
        ax,
        xlabel="Number of RFI-contaminated channels",
        ylabel="Bias (%)",
        title="RFI robustness",
    )
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig4_rfi_robustness.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 5: Sparse Channels
# =============================================================================
def generate_fig5():
    """Performance with sparse frequency channels."""
    print("Generating fig5_sparse_channels.png...")

    np.random.seed(42)
    dm_true = 300.0
    times = np.linspace(0, 0.5, 256)

    n_channels_list = [4, 8, 16, 32, 64, 128]
    n_trials = 50

    results = {"mean": [], "std": []}

    for n_ch in n_channels_list:
        freqs = np.linspace(1500, 1100, n_ch)
        biases = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, 15, freqs, times)
            est = StreamingDMEstimator(freqs, times, weight_power=3)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r:
                biases.append((r.dm - dm_true) / dm_true * 100)

        results["mean"].append(np.mean(biases) if biases else np.nan)
        results["std"].append(np.std(biases) if biases else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        n_channels_list,
        results["mean"],
        yerr=results["std"],
        fmt="o-",
        color="#377eb8",
        linewidth=2,
        markersize=10,
        capsize=4,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(
        2, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="2% threshold"
    )
    ax.axhline(-2, color="red", linestyle=":", linewidth=1.5, alpha=0.7)

    setup_axes(
        ax,
        xlabel="Number of frequency channels",
        ylabel="Bias (%)",
        title="Performance vs channel count",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_channels_list)
    ax.set_xticklabels([str(n) for n in n_channels_list])
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig5_sparse_channels.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 6: Threshold Sensitivity
# =============================================================================
def generate_fig6():
    """Sensitivity to detection threshold."""
    print("Generating fig6_threshold_sensitivity.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    n_trials = 50

    results = {"bias_mean": [], "bias_std": [], "n_pixels": []}

    for thresh in thresholds:
        biases = []
        n_pix = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, 10, freqs, times)
            est = StreamingDMEstimator(
                freqs, times, sigma_threshold=thresh, weight_power=3
            )
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r:
                biases.append((r.dm - dm_true) / dm_true * 100)
                n_pix.append(r.n_pixels)

        results["bias_mean"].append(np.mean(biases) if biases else np.nan)
        results["bias_std"].append(np.std(biases) if biases else np.nan)
        results["n_pixels"].append(np.mean(n_pix) if n_pix else 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bias vs threshold
    ax1 = axes[0]
    ax1.errorbar(
        thresholds,
        results["bias_mean"],
        yerr=results["bias_std"],
        fmt="o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        capsize=4,
    )
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.axvspan(3, 4, alpha=0.2, color="green", label="Optimal range")
    setup_axes(
        ax1,
        xlabel=r"Threshold ($\sigma$)",
        ylabel="Bias (%)",
        title="Bias vs threshold",
    )
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, alpha=0.3)

    # Pixels vs threshold
    ax2 = axes[1]
    ax2.plot(
        thresholds,
        results["n_pixels"],
        "o-",
        color="#984ea3",
        linewidth=2,
        markersize=8,
    )
    ax2.axvspan(3, 4, alpha=0.2, color="green", label="Optimal range")
    setup_axes(
        ax2,
        xlabel=r"Threshold ($\sigma$)",
        ylabel="Pixels selected",
        title="Signal pixels vs threshold",
    )
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig6_threshold_sensitivity.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 7: DM Range
# =============================================================================
def generate_fig7():
    """DM range and edge effects."""
    print("Generating fig7_dm_range.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    obs_window = times[-1] - times[0]

    dm_values = [50, 100, 150, 200, 250, 300, 400, 500, 600]
    n_trials = 30

    results = {"bias_mean": [], "bias_std": [], "sweep_time": []}

    for dm in dm_values:
        # Calculate sweep time
        sweep = K_DM * dm * (freqs[-1] ** -2 - freqs[0] ** -2)
        results["sweep_time"].append(sweep)

        biases = []
        for _ in range(n_trials):
            ds = generate_test_spectrum(dm, 0.05, 0.003, 15, freqs, times)
            est = StreamingDMEstimator(freqs, times, weight_power=3)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r:
                biases.append((r.dm - dm) / dm * 100)

        results["bias_mean"].append(np.mean(biases) if biases else np.nan)
        results["bias_std"].append(np.std(biases) if biases else np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Bias vs DM
    ax1 = axes[0]
    ax1.errorbar(
        dm_values,
        results["bias_mean"],
        yerr=results["bias_std"],
        fmt="o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        capsize=4,
    )
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    setup_axes(ax1, ylabel="Bias (%)", title="DM estimation accuracy vs true DM")
    ax1.grid(True, alpha=0.3)

    # Sweep time vs DM
    ax2 = axes[1]
    ax2.plot(
        dm_values,
        results["sweep_time"],
        "o-",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
    )
    ax2.axhline(
        obs_window,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Observation window ({obs_window:.2f} s)",
    )
    setup_axes(
        ax2,
        xlabel="True DM (pc/cm³)",
        ylabel="Sweep time (s)",
        title="Dispersion sweep time",
    )
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig7_dm_range.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 8: Multiple Pulses
# =============================================================================
def generate_fig8():
    """Effect of multiple pulses."""
    print("Generating fig8_multiple_pulses.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 1.0, 512)
    dm_true = 200.0

    n_pulses_list = [1, 2, 3, 4, 5]
    n_trials = 30

    results = {"bias_mean": [], "bias_std": []}

    for n_pulses in n_pulses_list:
        biases = []

        for _ in range(n_trials):
            # Generate multiple pulses
            ds = np.random.randn(len(freqs), len(times))
            pulse_times = np.linspace(0.1, 0.8, n_pulses)

            for t0 in pulse_times:
                ds_pulse = generate_test_spectrum(
                    dm_true, t0, 0.005, 15 / np.sqrt(n_pulses), freqs, times
                )
                ds += ds_pulse - np.random.randn(
                    len(freqs), len(times)
                )  # Remove double noise

            est = StreamingDMEstimator(freqs, times, weight_power=3)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r:
                biases.append((r.dm - dm_true) / dm_true * 100)

        results["bias_mean"].append(np.mean(biases) if biases else np.nan)
        results["bias_std"].append(np.std(biases) if biases else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        n_pulses_list,
        results["bias_mean"],
        yerr=results["bias_std"],
        fmt="o-",
        color="#377eb8",
        linewidth=2,
        markersize=10,
        capsize=4,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    setup_axes(
        ax,
        xlabel="Number of pulses",
        ylabel="Bias (%)",
        title="Effect of multiple pulses on DM estimation",
    )
    ax.set_xticks(n_pulses_list)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig8_multiple_pulses.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 9: Low S/N Failure
# =============================================================================
def generate_fig9():
    """Low S/N failure mode."""
    print("Generating fig9_low_snr_failure.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    snr_values = [2, 3, 4, 5, 6, 7, 8, 10]
    n_trials = 50

    results = {"bias_mean": [], "bias_std": [], "success_rate": []}

    for snr in snr_values:
        biases = []
        n_valid = 0

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)
            est = StreamingDMEstimator(freqs, times, weight_power=4)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r and not np.isnan(r.dm) and abs(r.dm) < 1000:
                biases.append((r.dm - dm_true) / dm_true * 100)
                n_valid += 1

        results["bias_mean"].append(np.mean(biases) if biases else np.nan)
        results["bias_std"].append(np.std(biases) if biases else np.nan)
        results["success_rate"].append(n_valid / n_trials * 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bias at low S/N
    ax1 = axes[0]
    ax1.errorbar(
        snr_values,
        results["bias_mean"],
        yerr=results["bias_std"],
        fmt="o-",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        capsize=4,
    )
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.axhline(
        -15, color="orange", linestyle=":", linewidth=1.5, label="-15% threshold"
    )
    ax1.axvspan(0, 5, alpha=0.15, color="red", label="Unreliable")
    setup_axes(ax1, xlabel="S/N", ylabel="Bias (%)", title="Bias at low S/N")
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, alpha=0.3)

    # Success rate
    ax2 = axes[1]
    ax2.plot(
        snr_values,
        results["success_rate"],
        "o-",
        color="#4daf4a",
        linewidth=2,
        markersize=8,
    )
    ax2.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axvspan(0, 5, alpha=0.15, color="red", label="Unreliable")
    setup_axes(
        ax2, xlabel="S/N", ylabel="Success rate (%)", title="Valid estimate rate"
    )
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig9_low_snr_failure.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 10: Scattering Effects
# =============================================================================
def generate_fig10():
    """Effect of scattering on DM estimation."""
    print("Generating fig10_scattering.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    # Scattering timescales (relative to pulse width)
    scatter_factors = [0, 0.5, 1.0, 2.0, 3.0, 5.0]
    base_width = 0.005  # 5 ms base pulse width
    n_trials = 50

    results = {"bias_mean": [], "bias_std": []}

    for scatter in scatter_factors:
        biases = []

        for _ in range(n_trials):
            # Generate pulse with scattering (exponential tail)
            ds = np.random.randn(len(freqs), len(times))
            freq_ref = freqs.max()

            for i, freq in enumerate(freqs):
                delay = K_DM * dm_true * (freq**-2 - freq_ref**-2)
                t_arrival = 0.1 + delay

                # Gaussian pulse convolved with exponential (approximation)
                pulse_width = base_width
                scatter_time = scatter * base_width

                for j, t in enumerate(times):
                    dt = t - t_arrival
                    if scatter_time > 0 and dt > -3 * pulse_width:
                        # Asymmetric pulse shape (Gaussian + exponential tail)
                        if dt < 0:
                            ds[i, j] += 15 * np.exp(-0.5 * (dt / pulse_width) ** 2)
                        else:
                            ds[i, j] += (
                                15
                                * np.exp(-dt / scatter_time)
                                * np.exp(
                                    -0.5
                                    * (dt / pulse_width) ** 2
                                    / (1 + dt / scatter_time)
                                )
                            )
                    elif scatter_time == 0:
                        ds[i, j] += 15 * np.exp(
                            -0.5 * ((t - t_arrival) / pulse_width) ** 2
                        )

            est = StreamingDMEstimator(freqs, times, weight_power=3)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r:
                biases.append((r.dm - dm_true) / dm_true * 100)

        results["bias_mean"].append(np.mean(biases) if biases else np.nan)
        results["bias_std"].append(np.std(biases) if biases else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        scatter_factors,
        results["bias_mean"],
        yerr=results["bias_std"],
        fmt="o-",
        color="#984ea3",
        linewidth=2,
        markersize=10,
        capsize=4,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    setup_axes(
        ax,
        xlabel="Scattering time / pulse width",
        ylabel="Bias (%)",
        title="Effect of scattering on DM estimation",
    )
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(
        "Scattering causes\npositive bias",
        xy=(3, results["bias_mean"][4]),
        xytext=(4, results["bias_mean"][4] + 5),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="#984ea3"),
    )

    plt.tight_layout()
    plt.savefig("figures/fig10_scattering.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 11: Pulse Width Effects
# =============================================================================
def generate_fig11():
    """Effect of pulse width on estimation accuracy."""
    print("Generating fig11_pulse_width.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    dm_true = 300.0

    # Pulse widths in seconds
    pulse_widths = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    n_trials = 50

    results = {"bias_mean": [], "bias_std": [], "std_dm": []}

    for width in pulse_widths:
        # Adjust time array to accommodate pulse
        times = np.linspace(0, 0.5, 256)
        dt = times[1] - times[0]

        biases = []
        dm_stds = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, width, 15, freqs, times)
            est = StreamingDMEstimator(freqs, times, weight_power=3)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r:
                biases.append((r.dm - dm_true) / dm_true * 100)

        results["bias_mean"].append(np.mean(biases) if biases else np.nan)
        results["bias_std"].append(np.std(biases) if biases else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        [w * 1000 for w in pulse_widths],
        results["bias_mean"],
        yerr=results["bias_std"],
        fmt="o-",
        color="#ff7f00",
        linewidth=2,
        markersize=10,
        capsize=4,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    setup_axes(
        ax,
        xlabel="Pulse width (ms)",
        ylabel="Bias (%)",
        title="Effect of pulse width on DM estimation",
    )
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig11_pulse_width.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 12: Iterative vs Single-Pass
# =============================================================================
def generate_fig12():
    """Comparison of iterative refinement vs single-pass."""
    print("Generating fig12_iterative.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    snr_values = [5, 7, 10, 15, 20]
    n_trials = 30

    results_single = {"mean": [], "std": []}
    results_iter = {"mean": [], "std": []}

    for snr in snr_values:
        biases_single = []
        biases_iter = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)

            # Single-pass with p=4
            est_single = StreamingDMEstimator(freqs, times, weight_power=4)
            est_single.process_spectrum(ds)
            r_single = est_single.get_estimate()

            # Iterative refinement
            r_iter = iterative_dm_estimate(ds, freqs, times, n_iterations=3)

            if r_single:
                biases_single.append((r_single.dm - dm_true) / dm_true * 100)
            if r_iter:
                biases_iter.append((r_iter.dm - dm_true) / dm_true * 100)

        results_single["mean"].append(
            np.mean(biases_single) if biases_single else np.nan
        )
        results_single["std"].append(np.std(biases_single) if biases_single else np.nan)
        results_iter["mean"].append(np.mean(biases_iter) if biases_iter else np.nan)
        results_iter["std"].append(np.std(biases_iter) if biases_iter else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        snr_values,
        results_single["mean"],
        yerr=results_single["std"],
        fmt="o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Single-pass (p=4)",
    )
    ax.errorbar(
        snr_values,
        results_iter["mean"],
        yerr=results_iter["std"],
        fmt="s-",
        color="#4daf4a",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Iterative (3 passes)",
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvspan(0, 5, alpha=0.1, color="red")

    setup_axes(
        ax, xlabel="S/N", ylabel="Bias (%)", title="Single-pass vs iterative refinement"
    )
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig12_iterative.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 13: Computational Timing Benchmarks
# =============================================================================
def generate_fig13():
    """Computational timing benchmarks."""
    print("Generating fig13_timing.png...")
    import time

    np.random.seed(42)

    # Different problem sizes
    n_channels_list = [16, 32, 64, 128, 256]
    n_times = 512
    dm_true = 300.0
    n_dm_trials = 100  # For trial dedispersion

    times_streaming = []
    times_iterative = []
    times_trial_dd = []

    for n_ch in n_channels_list:
        freqs = np.linspace(1500, 1100, n_ch)
        time_arr = np.linspace(0, 0.5, n_times)
        ds = generate_test_spectrum(dm_true, 0.1, 0.005, 15, freqs, time_arr)

        # Streaming estimator timing
        t0 = time.perf_counter()
        for _ in range(10):
            est = StreamingDMEstimator(freqs, time_arr, weight_power=4)
            est.process_spectrum(ds)
            _ = est.get_estimate()
        t_streaming = (time.perf_counter() - t0) / 10 * 1000  # ms
        times_streaming.append(t_streaming)

        # Iterative estimator timing
        t0 = time.perf_counter()
        for _ in range(10):
            _ = iterative_dm_estimate(ds, freqs, time_arr, n_iterations=3)
        t_iterative = (time.perf_counter() - t0) / 10 * 1000  # ms
        times_iterative.append(t_iterative)

        # Trial dedispersion timing (simplified)
        t0 = time.perf_counter()
        dm_trials = np.linspace(100, 500, n_dm_trials)
        for _ in range(3):  # Only 3 reps due to slowness
            snr_max = 0
            best_dm = 0
            for dm_trial in dm_trials:
                # Dedisperse
                dedispersed = np.zeros(n_times)
                for i, freq in enumerate(freqs):
                    shift = int(
                        K_DM
                        * dm_trial
                        * (freq**-2 - freqs[0] ** -2)
                        / (time_arr[1] - time_arr[0])
                    )
                    if 0 <= shift < n_times:
                        dedispersed += np.roll(ds[i, :], -shift)
                snr = dedispersed.max() / dedispersed.std()
                if snr > snr_max:
                    snr_max = snr
                    best_dm = dm_trial
        t_trial = (time.perf_counter() - t0) / 3 * 1000  # ms
        times_trial_dd.append(t_trial)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute timing
    ax1 = axes[0]
    ax1.plot(
        n_channels_list,
        times_streaming,
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        label="Streaming (single-pass)",
    )
    ax1.plot(
        n_channels_list,
        times_iterative,
        "s-",
        color="#4daf4a",
        linewidth=2,
        markersize=8,
        label="Iterative (3 passes)",
    )
    ax1.plot(
        n_channels_list,
        times_trial_dd,
        "^-",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        label=f"Trial DD ({n_dm_trials} trials)",
    )

    setup_axes(
        ax1,
        xlabel="Number of channels",
        ylabel="Time (ms)",
        title="Computation time vs problem size",
    )
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, alpha=0.3, which="both")

    # Speedup
    ax2 = axes[1]
    speedup_streaming = [t / s for t, s in zip(times_trial_dd, times_streaming)]
    speedup_iterative = [t / i for t, i in zip(times_trial_dd, times_iterative)]

    ax2.plot(
        n_channels_list,
        speedup_streaming,
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        label="Streaming vs Trial DD",
    )
    ax2.plot(
        n_channels_list,
        speedup_iterative,
        "s-",
        color="#4daf4a",
        linewidth=2,
        markersize=8,
        label="Iterative vs Trial DD",
    )
    ax2.axhline(1, color="black", linestyle="--", alpha=0.5)

    setup_axes(
        ax2,
        xlabel="Number of channels",
        ylabel="Speedup factor",
        title="Speedup over trial dedispersion",
    )
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig13_timing.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 14: Cramér-Rao Bound Comparison
# =============================================================================
def generate_fig14():
    """Compare estimator variance to Cramér-Rao lower bound."""
    print("Generating fig14_cramer_rao.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0
    freq_ref = freqs.max()

    # Compute dispersion coordinates
    x = freqs**-2 - freq_ref**-2

    snr_values = [5, 7, 10, 15, 20, 30, 50]
    n_trials = 100

    # For each S/N, compute:
    # 1. Empirical variance of DM estimates
    # 2. Cramér-Rao lower bound

    results = {"empirical_std": [], "crb_std": [], "efficiency": []}

    for snr in snr_values:
        dm_estimates = []

        for trial in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)
            est = StreamingDMEstimator(freqs, times, weight_power=4)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r and not np.isnan(r.dm):
                dm_estimates.append(r.dm)

        if len(dm_estimates) > 10:
            empirical_std = np.std(dm_estimates)
        else:
            empirical_std = np.nan
        results["empirical_std"].append(empirical_std)

        # Cramér-Rao bound calculation
        # For weighted least squares: Var(β) = σ² / Σ w(x - x̄)²
        # The minimum variance is achieved when w ∝ 1/σ² (constant noise)
        # For DM: Var(DM) = Var(β) / K_DM²

        # Generate one spectrum to estimate signal properties
        ds_ref = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)
        noise_sigma = 1.0  # By construction

        # For matched filter, the Fisher information is:
        # I(DM) = (1/σ²) * Σᵢ (∂t/∂DM)² * signal_power_i
        # where ∂t/∂DM = K_DM * xᵢ

        # Approximate the signal power at each frequency
        pulse_amplitude = snr * noise_sigma

        # Fisher information for DM (theoretical)
        # I(DM) ≈ (SNR²/N_eff) * K_DM² * Σ x²
        # where N_eff is effective noise per frequency channel

        # Simplified CRB for centroid estimation:
        # σ_DM ≈ σ_t / (K_DM * Δx)
        # where σ_t is timing precision and Δx is frequency leverage

        # For Gaussian pulse with width w and peak S/N:
        # σ_t ≈ w / SNR
        pulse_width = 0.005  # seconds
        sigma_t = pulse_width / snr  # timing precision

        # Frequency leverage
        delta_x = x.max() - x.min()

        # CRB for DM
        crb_dm = sigma_t / (K_DM * delta_x)
        results["crb_std"].append(crb_dm)

        # Efficiency
        if not np.isnan(empirical_std) and crb_dm > 0:
            efficiency = (crb_dm / empirical_std) ** 2 * 100  # percentage
        else:
            efficiency = np.nan
        results["efficiency"].append(efficiency)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Standard deviation comparison
    ax1 = axes[0]
    ax1.plot(
        snr_values,
        results["empirical_std"],
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        label="Streaming estimator",
    )
    ax1.plot(
        snr_values,
        results["crb_std"],
        "s--",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        label="Cramér-Rao bound",
    )

    setup_axes(
        ax1,
        xlabel="S/N",
        ylabel="Std dev of DM (pc/cm³)",
        title="Estimator precision vs theoretical limit",
    )
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, alpha=0.3, which="both")

    # Efficiency
    ax2 = axes[1]
    ax2.plot(
        snr_values,
        results["efficiency"],
        "o-",
        color="#984ea3",
        linewidth=2,
        markersize=10,
    )
    ax2.axhline(100, color="green", linestyle="--", linewidth=2, label="100% = optimal")
    ax2.axhspan(80, 120, alpha=0.1, color="green")

    setup_axes(
        ax2,
        xlabel="S/N",
        ylabel="Efficiency (%)",
        title="Estimator efficiency (CRB² / Var)",
    )
    ax2.set_ylim(
        0, max(150, max([e for e in results["efficiency"] if not np.isnan(e)]) + 20)
    )
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig14_cramer_rao.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 15: Why Efficiency Drops at Low S/N
# =============================================================================
def generate_fig15():
    """Analyze why efficiency drops at lower S/N."""
    print("Generating fig15_efficiency_analysis.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    snr_values = [3, 5, 7, 10, 15, 20, 30, 50]
    n_trials = 80

    # Track multiple metrics
    results = {
        "bias": [],
        "std": [],
        "signal_fraction": [],
        "noise_contamination": [],
        "effective_snr": [],
    }

    for snr in snr_values:
        biases = []
        stds = []
        sig_fracs = []
        noise_contam = []
        eff_snrs = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)

            est = StreamingDMEstimator(
                freqs, times, weight_power=4, sigma_threshold=3.0
            )
            est.process_spectrum(ds)
            r = est.get_estimate()

            if r:
                biases.append((r.dm - dm_true) / dm_true * 100)
                sig_fracs.append(r.signal_fraction * 100)

                # Estimate noise contamination: fraction of "signal" pixels that are actually noise
                # At 3σ threshold, ~0.27% of pure noise pixels exceed threshold
                expected_noise_above_thresh = 0.0027 * len(freqs) * len(times)
                noise_contam.append(
                    expected_noise_above_thresh / max(r.n_pixels, 1) * 100
                )

                # Effective S/N of selected pixels
                # Higher power weighting reduces effective S/N contribution from noise
                eff_snrs.append(snr * (r.signal_fraction**0.5))

        results["bias"].append(np.mean(np.abs(biases)) if biases else np.nan)
        results["std"].append(np.std(biases) if biases else np.nan)
        results["signal_fraction"].append(np.mean(sig_fracs) if sig_fracs else np.nan)
        results["noise_contamination"].append(
            np.mean(noise_contam) if noise_contam else np.nan
        )
        results["effective_snr"].append(np.mean(eff_snrs) if eff_snrs else np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Bias vs S/N (absolute)
    ax1 = axes[0, 0]
    ax1.plot(
        snr_values, results["bias"], "o-", color="#e41a1c", linewidth=2, markersize=8
    )
    ax1.axhline(1, color="green", linestyle="--", alpha=0.7, label="1% target")
    setup_axes(ax1, xlabel="S/N", ylabel="|Bias| (%)", title="A. Bias magnitude vs S/N")
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B: Signal fraction selected
    ax2 = axes[0, 1]
    ax2.plot(
        snr_values,
        results["signal_fraction"],
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
    )
    setup_axes(
        ax2,
        xlabel="S/N",
        ylabel="Signal fraction (%)",
        title="B. Fraction of pixels above threshold",
    )
    ax2.grid(True, alpha=0.3)

    # Panel C: Noise contamination
    ax3 = axes[1, 0]
    ax3.plot(
        snr_values,
        results["noise_contamination"],
        "o-",
        color="#ff7f00",
        linewidth=2,
        markersize=8,
    )
    ax3.axhline(50, color="red", linestyle="--", alpha=0.7, label="50% contamination")
    setup_axes(
        ax3,
        xlabel="S/N",
        ylabel="Noise contamination (%)",
        title="C. Estimated noise pixels in selection",
    )
    ax3.legend(fontsize=LEGEND_SIZE)
    ax3.grid(True, alpha=0.3)

    # Panel D: Standard deviation
    ax4 = axes[1, 1]
    ax4.plot(
        snr_values, results["std"], "o-", color="#984ea3", linewidth=2, markersize=8
    )
    setup_axes(ax4, xlabel="S/N", ylabel="Std dev (%)", title="D. Estimate variance")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Why Efficiency Drops at Low S/N", fontsize=16, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig("figures/fig15_efficiency_analysis.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 16: Adaptive Weighting Strategies
# =============================================================================
def generate_fig16():
    """Investigate adaptive weighting to improve efficiency."""
    print("Generating fig16_adaptive_weighting.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0
    freq_ref = freqs.max()
    x = freqs**-2 - freq_ref**-2

    snr_values = [5, 7, 10, 15, 20]
    n_trials = 60

    # Different weighting strategies
    strategies = {
        "p=1 (standard)": {"power": 1, "adaptive": False},
        "p=4 (high power)": {"power": 4, "adaptive": False},
        "Adaptive p": {"power": None, "adaptive": True},
        "Iterative (3x)": {"power": 3, "iterative": True},
    }

    results = {name: {"bias": [], "std": [], "efficiency": []} for name in strategies}

    # Compute CRB for reference
    pulse_width = 0.005
    delta_x = x.max() - x.min()

    for snr in snr_values:
        crb_dm = (pulse_width / snr) / (K_DM * delta_x)

        for name, config in strategies.items():
            biases = []

            for _ in range(n_trials):
                ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)

                if config.get("iterative"):
                    r = iterative_dm_estimate(ds, freqs, times, n_iterations=3)
                elif config.get("adaptive"):
                    # Adaptive power selection based on estimated S/N
                    # First pass: estimate S/N
                    est_init = StreamingDMEstimator(freqs, times, weight_power=2)
                    est_init.process_spectrum(ds)
                    r_init = est_init.get_estimate()

                    if r_init:
                        # Estimate S/N from signal fraction
                        est_snr = max(3, min(30, r_init.signal_fraction * 500))

                        # Choose power based on estimated S/N
                        if est_snr < 7:
                            adaptive_p = 2
                        elif est_snr < 15:
                            adaptive_p = 3
                        else:
                            adaptive_p = 4

                        est = StreamingDMEstimator(
                            freqs, times, weight_power=adaptive_p
                        )
                        est.process_spectrum(ds)
                        r = est.get_estimate()
                    else:
                        r = None
                else:
                    est = StreamingDMEstimator(
                        freqs, times, weight_power=config["power"]
                    )
                    est.process_spectrum(ds)
                    r = est.get_estimate()

                if r and not np.isnan(r.dm):
                    biases.append((r.dm - dm_true) / dm_true * 100)

            if biases:
                results[name]["bias"].append(np.mean(np.abs(biases)))
                std_dm = np.std(biases) / 100 * dm_true  # Convert back to pc/cm³
                results[name]["std"].append(np.std(biases))
                efficiency = (crb_dm / std_dm) ** 2 * 100 if std_dm > 0 else np.nan
                results[name]["efficiency"].append(
                    min(efficiency, 150)
                )  # Cap for display
            else:
                results[name]["bias"].append(np.nan)
                results[name]["std"].append(np.nan)
                results[name]["efficiency"].append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    markers = ["o", "s", "^", "D"]

    # Panel A: Bias comparison
    ax1 = axes[0]
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(
            snr_values,
            data["bias"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    ax1.axhline(1, color="gray", linestyle="--", alpha=0.5)
    setup_axes(ax1, xlabel="S/N", ylabel="|Bias| (%)", title="A. Bias comparison")
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE - 1, loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B: Standard deviation
    ax2 = axes[1]
    for i, (name, data) in enumerate(results.items()):
        ax2.plot(
            snr_values,
            data["std"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    setup_axes(ax2, xlabel="S/N", ylabel="Std dev (%)", title="B. Variance comparison")
    ax2.set_yscale("log")
    ax2.legend(fontsize=LEGEND_SIZE - 1, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    # Panel C: Efficiency
    ax3 = axes[2]
    for i, (name, data) in enumerate(results.items()):
        ax3.plot(
            snr_values,
            data["efficiency"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    ax3.axhline(100, color="green", linestyle="--", linewidth=2, label="Optimal (100%)")
    ax3.axhspan(80, 120, alpha=0.1, color="green")
    setup_axes(
        ax3, xlabel="S/N", ylabel="Efficiency (%)", title="C. Cramér-Rao efficiency"
    )
    ax3.set_ylim(0, 130)
    ax3.legend(fontsize=LEGEND_SIZE - 1, loc="lower right")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "Adaptive Weighting Strategies", fontsize=16, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig("figures/fig16_adaptive_weighting.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 17: Natural Weighting from First Principles
# =============================================================================
def generate_fig17():
    """Derive and test the 'natural' weighting from first principles."""
    print("Generating fig17_natural_weighting.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    snr_values = [5, 7, 10, 15, 20, 30]
    n_trials = 60

    # The key insight: what IS the optimal weight for a pixel?
    #
    # For weighted least squares regression t = β₀ + β₁x:
    #   - The optimal weight for observation i is w_i = 1/Var(t_i)
    #   - But we don't observe t directly - we observe intensity I
    #
    # A pixel with intensity I above threshold τ contributes to our estimate.
    # The "evidence" that this pixel is signal (not noise) is (I - τ).
    #
    # The NATURAL weight is: w = (I - τ) / σ² = signal excess / noise variance
    # This is equivalent to p=1 with proper normalization.
    #
    # Higher powers (p>1) are a BIAS-VARIANCE TRADEOFF:
    #   - They reduce noise contamination (lower variance)
    #   - But they over-weight the peak and under-weight the wings (bias)

    # Test different weighting philosophies
    strategies = {
        "p=1 (natural)": 1.0,
        "p=1.5": 1.5,
        "p=2 (SNR²)": 2.0,
        "p=3": 3.0,
        "p=4": 4.0,
    }

    results = {name: {"bias": [], "std": [], "mse": []} for name in strategies}

    for snr in snr_values:
        for name, p in strategies.items():
            biases = []

            for _ in range(n_trials):
                ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)
                est = StreamingDMEstimator(
                    freqs, times, weight_power=p, sigma_threshold=3.0
                )
                est.process_spectrum(ds)
                r = est.get_estimate()
                if r and not np.isnan(r.dm):
                    biases.append((r.dm - dm_true) / dm_true * 100)

            if biases:
                mean_bias = np.mean(biases)
                std_bias = np.std(biases)
                mse = mean_bias**2 + std_bias**2  # MSE = bias² + variance
                results[name]["bias"].append(abs(mean_bias))
                results[name]["std"].append(std_bias)
                results[name]["mse"].append(np.sqrt(mse))
            else:
                results[name]["bias"].append(np.nan)
                results[name]["std"].append(np.nan)
                results[name]["mse"].append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    markers = ["o", "s", "^", "D", "v"]

    # Panel A: Bias
    ax1 = axes[0]
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(
            snr_values,
            data["bias"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    setup_axes(ax1, xlabel="S/N", ylabel="|Bias| (%)", title="A. Bias (accuracy)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE - 1)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B: Standard deviation
    ax2 = axes[1]
    for i, (name, data) in enumerate(results.items()):
        ax2.plot(
            snr_values,
            data["std"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    setup_axes(ax2, xlabel="S/N", ylabel="Std dev (%)", title="B. Variance (precision)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=LEGEND_SIZE - 1)
    ax2.grid(True, alpha=0.3, which="both")

    # Panel C: MSE (total error)
    ax3 = axes[2]
    for i, (name, data) in enumerate(results.items()):
        ax3.plot(
            snr_values,
            data["mse"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    setup_axes(
        ax3,
        xlabel="S/N",
        ylabel="RMSE (%)",
        title="C. Total error (bias² + variance)^½",
    )
    ax3.set_yscale("log")
    ax3.legend(fontsize=LEGEND_SIZE - 1)
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Natural Weighting: The Bias-Variance Tradeoff",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig("figures/fig17_natural_weighting.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 18: The Unifying Formula
# =============================================================================
def generate_fig18():
    """Test a single unifying formula: p = 1 + c/S/N."""
    print("Generating fig18_unifying_formula.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    snr_values = [5, 7, 10, 15, 20, 30, 50]
    n_trials = 60

    # The unifying formula: p = 1 + c / estimated_S/N
    # This interpolates between:
    #   - High S/N: p → 1 (natural weighting)
    #   - Low S/N: p increases (more aggressive filtering)

    strategies = {
        "Fixed p=1": {"type": "fixed", "p": 1},
        "Fixed p=3": {"type": "fixed", "p": 3},
        "Adaptive: p = 1 + 10/S/N": {"type": "adaptive", "c": 10},
        "Adaptive: p = 1 + 20/S/N": {"type": "adaptive", "c": 20},
    }

    results = {name: {"bias": [], "std": [], "p_used": []} for name in strategies}

    for snr in snr_values:
        for name, config in strategies.items():
            biases = []
            p_values = []

            for _ in range(n_trials):
                ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)

                if config["type"] == "fixed":
                    p = config["p"]
                else:
                    # First pass: estimate S/N
                    est_init = StreamingDMEstimator(freqs, times, weight_power=1.5)
                    est_init.process_spectrum(ds)
                    r_init = est_init.get_estimate()

                    if r_init:
                        # Estimate S/N from signal fraction (rough proxy)
                        est_snr = max(3, r_init.signal_fraction * 300)
                        p = 1 + config["c"] / est_snr
                        p = max(1, min(p, 5))  # Clamp to reasonable range
                    else:
                        p = 2  # Fallback

                p_values.append(p)

                est = StreamingDMEstimator(freqs, times, weight_power=p)
                est.process_spectrum(ds)
                r = est.get_estimate()

                if r and not np.isnan(r.dm):
                    biases.append((r.dm - dm_true) / dm_true * 100)

            if biases:
                results[name]["bias"].append(np.mean(np.abs(biases)))
                results[name]["std"].append(np.std(biases))
                results[name]["p_used"].append(np.mean(p_values))
            else:
                results[name]["bias"].append(np.nan)
                results[name]["std"].append(np.nan)
                results[name]["p_used"].append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    markers = ["o", "s", "^", "D"]

    # Panel A: p values used
    ax1 = axes[0]
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(
            snr_values,
            data["p_used"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    setup_axes(
        ax1,
        xlabel="True S/N",
        ylabel="Power p used",
        title="A. Adaptive power selection",
    )
    ax1.legend(fontsize=LEGEND_SIZE - 1)
    ax1.grid(True, alpha=0.3)

    # Panel B: Bias comparison
    ax2 = axes[1]
    for i, (name, data) in enumerate(results.items()):
        ax2.plot(
            snr_values,
            data["bias"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    setup_axes(ax2, xlabel="S/N", ylabel="|Bias| (%)", title="B. Bias")
    ax2.set_yscale("log")
    ax2.legend(fontsize=LEGEND_SIZE - 1)
    ax2.grid(True, alpha=0.3, which="both")

    # Panel C: RMSE
    ax3 = axes[2]
    for i, (name, data) in enumerate(results.items()):
        rmse = [np.sqrt(b**2 + s**2) for b, s in zip(data["bias"], data["std"])]
        ax3.plot(
            snr_values,
            rmse,
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    setup_axes(ax3, xlabel="S/N", ylabel="RMSE (%)", title="C. Total error")
    ax3.set_yscale("log")
    ax3.legend(fontsize=LEGEND_SIZE - 1)
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "The Unifying Formula: p = 1 + c/S/N", fontsize=16, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig("figures/fig18_unifying_formula.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 19: Centroid Regression vs Matched Filtering
# =============================================================================
def generate_fig19():
    """Compare centroid regression to matched filtering (if we knew the pulse shape)."""
    print("Generating fig19_centroid_vs_matched.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0
    freq_ref = freqs.max()
    pulse_width = 0.005

    snr_values = [5, 7, 10, 15, 20, 30]
    n_trials = 50

    results_centroid = {"bias": [], "std": []}
    results_matched = {"bias": [], "std": []}

    for snr in snr_values:
        biases_centroid = []
        biases_matched = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, pulse_width, snr, freqs, times)

            # Centroid regression (our method)
            est = StreamingDMEstimator(freqs, times, weight_power=3)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r:
                biases_centroid.append((r.dm - dm_true) / dm_true * 100)

            # "Matched filtering" - find arrival time at each frequency via
            # cross-correlation with known Gaussian template
            t_arrivals = []
            t_weights = []
            template = np.exp(-0.5 * ((times - times.mean()) / pulse_width) ** 2)

            for i, freq in enumerate(freqs):
                # Cross-correlate
                cc = np.correlate(
                    ds[i, :] - ds[i, :].mean(), template - template.mean(), mode="same"
                )
                # Find peak
                peak_idx = np.argmax(cc)
                t_arrivals.append(times[peak_idx])
                t_weights.append(cc[peak_idx])  # Weight by correlation strength

            t_arrivals = np.array(t_arrivals)
            t_weights = np.array(t_weights)
            t_weights = np.maximum(t_weights, 0)  # Non-negative

            if t_weights.sum() > 0:
                # Weighted linear regression
                x = freqs**-2 - freq_ref**-2
                w = t_weights
                W = w.sum()
                Sx = (w * x).sum()
                St = (w * t_arrivals).sum()
                Sxx = (w * x * x).sum()
                Sxt = (w * x * t_arrivals).sum()

                denom = K_DM * (W * Sxx - Sx**2)
                if abs(denom) > 1e-10:
                    dm_matched = (W * Sxt - Sx * St) / denom
                    biases_matched.append((dm_matched - dm_true) / dm_true * 100)

        results_centroid["bias"].append(
            np.mean(np.abs(biases_centroid)) if biases_centroid else np.nan
        )
        results_centroid["std"].append(
            np.std(biases_centroid) if biases_centroid else np.nan
        )
        results_matched["bias"].append(
            np.mean(np.abs(biases_matched)) if biases_matched else np.nan
        )
        results_matched["std"].append(
            np.std(biases_matched) if biases_matched else np.nan
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bias comparison
    ax1 = axes[0]
    ax1.plot(
        snr_values,
        results_centroid["bias"],
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        label="Centroid regression (ours)",
    )
    ax1.plot(
        snr_values,
        results_matched["bias"],
        "s--",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        label="Matched filter (known template)",
    )
    setup_axes(
        ax1,
        xlabel="S/N",
        ylabel="|Bias| (%)",
        title="A. Bias: centroid vs matched filter",
    )
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B: Std comparison
    ax2 = axes[1]
    ax2.plot(
        snr_values,
        results_centroid["std"],
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        label="Centroid regression (ours)",
    )
    ax2.plot(
        snr_values,
        results_matched["std"],
        "s--",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        label="Matched filter (known template)",
    )
    setup_axes(
        ax2,
        xlabel="S/N",
        ylabel="Std dev (%)",
        title="B. Precision: centroid vs matched filter",
    )
    ax2.set_yscale("log")
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "What We Actually Are: Centroid Regression",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig("figures/fig19_centroid_vs_matched.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 20: Correct CRB for Template-Free Estimation
# =============================================================================
def generate_fig20():
    """Compute the correct CRB when pulse shape is unknown."""
    print("Generating fig20_correct_crb.png...")

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0
    freq_ref = freqs.max()
    pulse_width = 0.005

    x = freqs**-2 - freq_ref**-2
    delta_x = x.max() - x.min()
    dt = times[1] - times[0]

    snr_values = [5, 7, 10, 15, 20, 30, 50]
    n_trials = 80

    results = {
        "empirical_std": [],
        "crb_matched": [],  # CRB if we knew pulse shape
        "crb_centroid": [],  # CRB for centroid estimation (template-free)
        "efficiency_matched": [],
        "efficiency_centroid": [],
    }

    for snr in snr_values:
        dm_estimates = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, pulse_width, snr, freqs, times)
            est = StreamingDMEstimator(freqs, times, weight_power=3)
            est.process_spectrum(ds)
            r = est.get_estimate()
            if r and not np.isnan(r.dm):
                dm_estimates.append(r.dm)

        empirical_std = np.std(dm_estimates) if dm_estimates else np.nan
        results["empirical_std"].append(empirical_std)

        # CRB for MATCHED FILTER (known pulse shape)
        # σ_t^MF ≈ w / (√2 × S/N)  for Gaussian pulse
        sigma_t_matched = pulse_width / (np.sqrt(2) * snr)
        crb_matched = sigma_t_matched / (K_DM * delta_x) / np.sqrt(len(freqs))
        results["crb_matched"].append(crb_matched)

        # CRB for CENTROID (template-free)
        # For centroid of a Gaussian pulse in noise:
        # Var(t_centroid) ≈ w² / (12 × N_eff) + σ² × w × N_pix / S²
        # where N_eff ≈ w/dt is effective samples in pulse
        #
        # Simplified: σ_t^centroid ≈ w / √(6 × S/N) for well-sampled pulse
        # This is √3 worse than matched filter
        sigma_t_centroid = pulse_width / np.sqrt(6) / snr * np.sqrt(1 + 1 / snr)
        crb_centroid = sigma_t_centroid / (K_DM * delta_x) / np.sqrt(len(freqs))
        results["crb_centroid"].append(crb_centroid)

        # Efficiency
        if not np.isnan(empirical_std):
            results["efficiency_matched"].append(
                (crb_matched / empirical_std) ** 2 * 100
            )
            results["efficiency_centroid"].append(
                (crb_centroid / empirical_std) ** 2 * 100
            )
        else:
            results["efficiency_matched"].append(np.nan)
            results["efficiency_centroid"].append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Std comparison
    ax1 = axes[0]
    ax1.plot(
        snr_values,
        results["empirical_std"],
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        label="Our method (empirical)",
    )
    ax1.plot(
        snr_values,
        results["crb_matched"],
        "s--",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        label="CRB (matched filter)",
    )
    ax1.plot(
        snr_values,
        results["crb_centroid"],
        "^--",
        color="#4daf4a",
        linewidth=2,
        markersize=8,
        label="CRB (centroid, template-free)",
    )
    setup_axes(
        ax1,
        xlabel="S/N",
        ylabel="Std dev of DM (pc/cm³)",
        title="A. Precision vs theoretical limits",
    )
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B: Efficiency vs correct CRB
    ax2 = axes[1]
    ax2.plot(
        snr_values,
        results["efficiency_matched"],
        "s-",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        label="vs matched filter CRB",
    )
    ax2.plot(
        snr_values,
        results["efficiency_centroid"],
        "^-",
        color="#4daf4a",
        linewidth=2,
        markersize=8,
        label="vs centroid CRB (correct!)",
    )
    ax2.axhline(100, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.axhspan(80, 120, alpha=0.1, color="green", label="Near-optimal")
    setup_axes(
        ax2,
        xlabel="S/N",
        ylabel="Efficiency (%)",
        title="B. Efficiency vs correct bound",
    )
    ax2.set_ylim(0, 150)
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "The Correct Cramér-Rao Bound (Template-Free)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig("figures/fig20_correct_crb.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 21: The Pairwise Consistency Estimator
# =============================================================================
def generate_fig21():
    """Test the pairwise consistency estimator against centroid limitations."""
    print("Generating fig21_pairwise_estimator.png...")

    # Import the new estimator
    from streaming_dm_estimator import PairwiseDMEstimator

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0
    freq_ref = freqs.max()

    snr_values = [5, 7, 10, 15, 20]
    n_trials = 40

    results_centroid = {"bias": [], "std": []}
    results_pairwise = {"bias": [], "std": []}

    for snr in snr_values:
        biases_centroid = []
        biases_pairwise = []

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)

            # Centroid method
            est_c = StreamingDMEstimator(freqs, times, weight_power=3)
            est_c.process_spectrum(ds)
            r_c = est_c.get_estimate()
            if r_c:
                biases_centroid.append((r_c.dm - dm_true) / dm_true * 100)

            # Pairwise method
            est_p = PairwiseDMEstimator(freqs, times, dm_range=(100, 500), n_bins=200)
            est_p.process_spectrum(ds)
            r_p = est_p.get_estimate()
            if r_p:
                biases_pairwise.append((r_p.dm - dm_true) / dm_true * 100)

        results_centroid["bias"].append(
            np.mean(np.abs(biases_centroid)) if biases_centroid else np.nan
        )
        results_centroid["std"].append(
            np.std(biases_centroid) if biases_centroid else np.nan
        )
        results_pairwise["bias"].append(
            np.mean(np.abs(biases_pairwise)) if biases_pairwise else np.nan
        )
        results_pairwise["std"].append(
            np.std(biases_pairwise) if biases_pairwise else np.nan
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bias comparison
    ax1 = axes[0]
    ax1.plot(
        snr_values,
        results_centroid["bias"],
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        label="Centroid (p=3)",
    )
    ax1.plot(
        snr_values,
        results_pairwise["bias"],
        "s-",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        label="Pairwise (offline)",
    )
    setup_axes(ax1, xlabel="S/N", ylabel="|Bias| (%)", title="A. Bias comparison")
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, alpha=0.3, which="both")

    # Std comparison
    ax2 = axes[1]
    ax2.plot(
        snr_values,
        results_centroid["std"],
        "o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        label="Centroid (p=3)",
    )
    ax2.plot(
        snr_values,
        results_pairwise["std"],
        "s-",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        label="Pairwise (offline)",
    )
    setup_axes(ax2, xlabel="S/N", ylabel="Std dev (%)", title="B. Precision comparison")
    ax2.set_yscale("log")
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Pairwise Consistency Estimator vs Centroid",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig("figures/fig21_pairwise_estimator.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 22: Robustness to Scattering (Asymmetric Pulses)
# =============================================================================
def generate_fig22():
    """Test robustness to scattering (asymmetric pulses)."""
    print("Generating fig22_scattering_robustness.png...")

    from streaming_dm_estimator import PairwiseDMEstimator

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.8, 400)
    dm_true = 300.0
    freq_ref = freqs.max()
    base_width = 0.005

    scatter_factors = [0, 1, 2, 3, 5]
    n_trials = 30
    snr = 15

    results_centroid = {"bias": [], "std": []}
    results_pairwise = {"bias": [], "std": []}

    for scatter in scatter_factors:
        biases_centroid = []
        biases_pairwise = []

        for _ in range(n_trials):
            # Generate scattered pulse
            ds = np.random.randn(len(freqs), len(times))

            for i, freq in enumerate(freqs):
                delay = K_DM * dm_true * (freq**-2 - freq_ref**-2)
                t_arrival = 0.1 + delay
                scatter_time = scatter * base_width

                for j, t in enumerate(times):
                    dt = t - t_arrival
                    if scatter_time > 0 and dt > -3 * base_width:
                        # Asymmetric pulse (scattered)
                        if dt < 0:
                            ds[i, j] += snr * np.exp(-0.5 * (dt / base_width) ** 2)
                        else:
                            ds[i, j] += snr * np.exp(-dt / scatter_time)
                    elif scatter_time == 0:
                        ds[i, j] += snr * np.exp(
                            -0.5 * ((t - t_arrival) / base_width) ** 2
                        )

            # Centroid method
            est_c = StreamingDMEstimator(freqs, times, weight_power=3)
            est_c.process_spectrum(ds)
            r_c = est_c.get_estimate()
            if r_c:
                biases_centroid.append((r_c.dm - dm_true) / dm_true * 100)

            # Pairwise method
            est_p = PairwiseDMEstimator(freqs, times, dm_range=(100, 500), n_bins=200)
            est_p.process_spectrum(ds)
            r_p = est_p.get_estimate()
            if r_p:
                biases_pairwise.append((r_p.dm - dm_true) / dm_true * 100)

        results_centroid["bias"].append(
            np.mean(biases_centroid) if biases_centroid else np.nan
        )
        results_centroid["std"].append(
            np.std(biases_centroid) if biases_centroid else np.nan
        )
        results_pairwise["bias"].append(
            np.mean(biases_pairwise) if biases_pairwise else np.nan
        )
        results_pairwise["std"].append(
            np.std(biases_pairwise) if biases_pairwise else np.nan
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        scatter_factors,
        results_centroid["bias"],
        yerr=results_centroid["std"],
        fmt="o-",
        color="#377eb8",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Centroid (p=3)",
    )
    ax.errorbar(
        scatter_factors,
        results_pairwise["bias"],
        yerr=results_pairwise["std"],
        fmt="s-",
        color="#e41a1c",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Pairwise (offline)",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    setup_axes(
        ax,
        xlabel="Scattering time / pulse width",
        ylabel="Bias (%)",
        title="Robustness to scattering (asymmetric pulses)",
    )
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig22_scattering_robustness.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 23: Multiple Pulse Detection
# =============================================================================
def generate_fig23():
    """Test ability to detect multiple pulses at different DMs."""
    print("Generating fig23_multiple_pulses.png...")

    from streaming_dm_estimator import PairwiseDMEstimator

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 1.0, 512)
    freq_ref = freqs.max()

    # Two pulses at different DMs
    dm1, dm2 = 200, 350
    snr1, snr2 = 15, 12

    # Generate spectrum with two pulses
    ds = np.random.randn(len(freqs), len(times))

    for dm, snr, t0 in [(dm1, snr1, 0.1), (dm2, snr2, 0.15)]:
        for i, freq in enumerate(freqs):
            delay = K_DM * dm * (freq**-2 - freq_ref**-2)
            t_arrival = t0 + delay
            for j, t in enumerate(times):
                ds[i, j] += snr * np.exp(-0.5 * ((t - t_arrival) / 0.005) ** 2)

    # Run pairwise estimator
    est = PairwiseDMEstimator(freqs, times, dm_range=(100, 500), n_bins=200)
    est.process_spectrum(ds)
    dm_centers, histogram = est.get_histogram()

    # Detect multiple pulses
    detected = est.detect_multiple_pulses(min_separation=30, min_height_ratio=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Dynamic spectrum
    ax1 = axes[0]
    im = ax1.imshow(
        ds,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        extent=[times[0] * 1000, times[-1] * 1000, freqs[-1], freqs[0]],
    )
    ax1.set_xlabel("Time (ms)", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Frequency (MHz)", fontsize=LABEL_SIZE)
    ax1.set_title(
        "A. Two pulses at different DMs", fontsize=TITLE_SIZE, fontweight="bold"
    )
    ax1.tick_params(axis="both", labelsize=TICK_SIZE)

    # Draw true DM curves
    for dm, color in [(dm1, "red"), (dm2, "cyan")]:
        t_arr = 0.1 + K_DM * dm * (freqs**-2 - freq_ref**-2)
        ax1.plot(t_arr * 1000, freqs, "--", color=color, linewidth=2, alpha=0.7)

    # Panel B: DM histogram
    ax2 = axes[1]
    ax2.plot(dm_centers, histogram / histogram.max(), "b-", linewidth=2)
    ax2.axvline(
        dm1, color="red", linestyle="--", linewidth=2, label=f"True DM$_1$ = {dm1}"
    )
    ax2.axvline(
        dm2, color="cyan", linestyle="--", linewidth=2, label=f"True DM$_2$ = {dm2}"
    )

    # Mark detected peaks
    for dm_det, strength in detected:
        ax2.axvline(dm_det, color="green", linestyle=":", linewidth=2, alpha=0.7)
        ax2.text(
            dm_det,
            strength + 0.05,
            f"{dm_det:.0f}",
            ha="center",
            fontsize=10,
            color="green",
        )

    setup_axes(
        ax2,
        xlabel="DM (pc/cm³)",
        ylabel="Histogram (normalized)",
        title="B. Pairwise DM histogram",
    )
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Multiple Pulse Detection", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig("figures/fig23_multiple_pulses.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 24: The Streaming Differential Median Estimator
# =============================================================================
def generate_fig24():
    """Compare all three methods: centroid, differential median, pairwise."""
    print("Generating fig24_differential_median.png...")

    from streaming_dm_estimator import (
        StreamingDMEstimator,
        StreamingDifferentialDMEstimator,
        PairwiseDMEstimator,
    )

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.5, 256)
    dm_true = 300.0

    snr_values = [5, 7, 10, 15, 20, 30]
    n_trials = 40

    results = {
        "Centroid (streaming)": {"bias": [], "std": []},
        "Differential median (streaming)": {"bias": [], "std": []},
        "Pairwise mode (offline)": {"bias": [], "std": []},
    }

    for snr in snr_values:
        biases = {k: [] for k in results.keys()}

        for _ in range(n_trials):
            ds = generate_test_spectrum(dm_true, 0.1, 0.005, snr, freqs, times)

            # Centroid method
            est1 = StreamingDMEstimator(freqs, times, weight_power=3)
            est1.process_spectrum(ds)
            r1 = est1.get_estimate()
            if r1:
                biases["Centroid (streaming)"].append((r1.dm - dm_true) / dm_true * 100)

            # Differential median method
            est2 = StreamingDifferentialDMEstimator(freqs, times, weight_power=2)
            est2.process_spectrum(ds)
            r2 = est2.get_estimate()
            if r2:
                biases["Differential median (streaming)"].append(
                    (r2.dm - dm_true) / dm_true * 100
                )

            # Pairwise method
            est3 = PairwiseDMEstimator(freqs, times, dm_range=(100, 500), n_bins=200)
            est3.process_spectrum(ds)
            r3 = est3.get_estimate()
            if r3:
                biases["Pairwise mode (offline)"].append(
                    (r3.dm - dm_true) / dm_true * 100
                )

        for name in results.keys():
            if biases[name]:
                results[name]["bias"].append(np.mean(np.abs(biases[name])))
                results[name]["std"].append(np.std(biases[name]))
            else:
                results[name]["bias"].append(np.nan)
                results[name]["std"].append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#377eb8", "#4daf4a", "#e41a1c"]
    markers = ["o", "s", "^"]

    # Panel A: Bias
    ax1 = axes[0]
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(
            snr_values,
            data["bias"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    ax1.axhline(1, color="gray", linestyle="--", alpha=0.5)
    setup_axes(ax1, xlabel="S/N", ylabel="|Bias| (%)", title="A. Bias comparison")
    ax1.set_yscale("log")
    ax1.legend(fontsize=LEGEND_SIZE - 1)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B: Std dev
    ax2 = axes[1]
    for i, (name, data) in enumerate(results.items()):
        ax2.plot(
            snr_values,
            data["std"],
            f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=name,
        )
    setup_axes(ax2, xlabel="S/N", ylabel="Std dev (%)", title="B. Precision comparison")
    ax2.set_yscale("log")
    ax2.legend(fontsize=LEGEND_SIZE - 1)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Three Estimators: Centroid vs Differential Median vs Pairwise",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig("figures/fig24_differential_median.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 25: Scattering Robustness Comparison (All Methods)
# =============================================================================
def generate_fig25():
    """Test scattering robustness of all three streaming-capable methods."""
    print("Generating fig25_scattering_all_methods.png...")

    from streaming_dm_estimator import (
        StreamingDMEstimator,
        StreamingDifferentialDMEstimator,
    )

    np.random.seed(42)
    freqs = np.linspace(1500, 1100, 64)
    times = np.linspace(0, 0.8, 400)
    dm_true = 300.0
    freq_ref = freqs.max()
    base_width = 0.005
    snr = 15

    scatter_factors = [0, 1, 2, 3, 5]
    n_trials = 30

    results = {
        "Centroid (p=3)": {"bias": [], "std": []},
        "Differential median": {"bias": [], "std": []},
    }

    for scatter in scatter_factors:
        biases = {k: [] for k in results.keys()}

        for _ in range(n_trials):
            # Generate scattered pulse
            ds = np.random.randn(len(freqs), len(times))

            for i, freq in enumerate(freqs):
                delay = K_DM * dm_true * (freq**-2 - freq_ref**-2)
                t_arrival = 0.1 + delay
                scatter_time = scatter * base_width

                for j, t in enumerate(times):
                    dt = t - t_arrival
                    if scatter_time > 0 and dt > -3 * base_width:
                        if dt < 0:
                            ds[i, j] += snr * np.exp(-0.5 * (dt / base_width) ** 2)
                        else:
                            ds[i, j] += snr * np.exp(-dt / scatter_time)
                    elif scatter_time == 0:
                        ds[i, j] += snr * np.exp(
                            -0.5 * ((t - t_arrival) / base_width) ** 2
                        )

            # Centroid
            est1 = StreamingDMEstimator(freqs, times, weight_power=3)
            est1.process_spectrum(ds)
            r1 = est1.get_estimate()
            if r1:
                biases["Centroid (p=3)"].append((r1.dm - dm_true) / dm_true * 100)

            # Differential median
            est2 = StreamingDifferentialDMEstimator(freqs, times, weight_power=2)
            est2.process_spectrum(ds)
            r2 = est2.get_estimate()
            if r2:
                biases["Differential median"].append((r2.dm - dm_true) / dm_true * 100)

        for name in results.keys():
            if biases[name]:
                results[name]["bias"].append(np.mean(biases[name]))
                results[name]["std"].append(np.std(biases[name]))
            else:
                results[name]["bias"].append(np.nan)
                results[name]["std"].append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#377eb8", "#4daf4a"]
    markers = ["o", "s"]

    for i, (name, data) in enumerate(results.items()):
        ax.errorbar(
            scatter_factors,
            data["bias"],
            yerr=data["std"],
            fmt=f"{markers[i]}-",
            color=colors[i],
            linewidth=2,
            markersize=8,
            capsize=4,
            label=name,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    setup_axes(
        ax,
        xlabel="Scattering time / pulse width",
        ylabel="Bias (%)",
        title="Scattering robustness: streaming methods only",
    )
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig25_scattering_all_methods.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# FIGURE 26: Memory Comparison
# =============================================================================
def generate_fig26():
    """Visualize memory usage of all methods."""
    print("Generating fig26_memory_comparison.png...")

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [
        "Centroid\n(StreamingDMEstimator)",
        "Differential Median\n(StreamingDifferential)",
        "Pairwise Mode\n(PairwiseDMEstimator)",
        "Iterative\n(iterative_dm_estimate)",
    ]

    # Memory complexity (in units of "floats stored")
    # Centroid: 5 scalars
    # Differential: ~N_channels (previous channel + list of DM estimates)
    # Pairwise: N_bins + N_bright_pixels (grows!)
    # Iterative: Full image

    n_channels = 64
    n_times = 256
    n_bins = 200
    n_bright = n_channels * 10  # ~10 bright pixels per channel

    memory = [
        5,  # Centroid: 5 scalars
        n_channels + 5,  # Differential: O(N_channels) DM estimates + constants
        n_bins + n_bright,  # Pairwise: histogram + all bright pixels
        n_channels * n_times,  # Iterative: full image
    ]

    colors = ["#4daf4a", "#4daf4a", "#e41a1c", "#e41a1c"]
    patterns = ["", "", "///", "///"]

    bars = ax.bar(methods, memory, color=colors, edgecolor="black", linewidth=1.5)

    # Add hatching for non-streaming methods
    bars[2].set_hatch("///")
    bars[3].set_hatch("///")

    # Add labels
    for bar, mem in zip(bars, memory):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{mem:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Memory (floats)", fontsize=LABEL_SIZE)
    ax.set_title("Memory Complexity Comparison", fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_yscale("log")
    ax.set_ylim(1, 50000)

    # Add legend for streaming vs non-streaming
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#4daf4a", edgecolor="black", label="Streaming (O(1) or O(N_ch))"
        ),
        Patch(
            facecolor="#e41a1c",
            edgecolor="black",
            hatch="///",
            label="Non-streaming (grows with data)",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="upper left")

    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("figures/fig26_memory_comparison.png", dpi=150, facecolor="white")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generating all figures with LaTeX styling...")
    print("=" * 50)

    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    generate_fig8()
    generate_fig9()
    generate_fig10()
    generate_fig11()
    generate_fig12()
    generate_fig13()
    generate_fig14()
    generate_fig15()
    generate_fig16()
    generate_fig17()
    generate_fig18()
    generate_fig19()
    generate_fig20()
    generate_fig21()
    generate_fig22()
    generate_fig23()
    generate_fig24()
    generate_fig25()
    generate_fig26()

    print("=" * 50)
    print("All figures generated successfully!")
