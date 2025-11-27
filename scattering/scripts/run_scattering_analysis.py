#!/usr/bin/env python
"""
run_scattering_analysis.py
==========================

Standalone script that mirrors the scattering_analysis.ipynb notebook.
Saves all figures to PDF files instead of displaying them.

Includes validation checks to flag potential issues with figures/results.

Usage:
    python run_scattering_analysis.py [config_file]

    # Default: uses casey_chime.yaml
    python run_scattering_analysis.py

    # Custom config:
    python run_scattering_analysis.py scattering/configs/bursts/dsa/casey_dsa.yaml
"""

import os
import sys
import warnings
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# =============================================================================
# SETUP PATHS
# =============================================================================

# Find FLITS root (this script is in scattering/scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
FLITS_ROOT = SCRIPT_DIR.parent.parent

# Add to path
sys.path.insert(0, str(FLITS_ROOT))
sys.path.insert(0, str(FLITS_ROOT / "scattering"))

# Change to FLITS root for relative paths
os.chdir(FLITS_ROOT)

print(f"[SETUP] FLITS root: {FLITS_ROOT}")
print(f"[SETUP] Working directory: {Path.cwd()}")

# =============================================================================
# IMPORTS
# =============================================================================

from scat_analysis.burstfit_pipeline import BurstPipeline, BurstDataset
from scat_analysis.burstfit_interactive import InitialGuessWidget
from scat_analysis.config_utils import load_config
from scat_analysis.burstfit import FRBParams

print("[OK] Imports successful")


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


class FigureValidator:
    """
    Utility class to validate figures and flag potential issues.
    Prints warnings that can be parsed to detect problems.
    """

    def __init__(self, name: str):
        self.name = name
        self.issues = []

    def check_array(self, arr, label: str, allow_negative: bool = True):
        """Check array for common issues."""
        arr = np.asarray(arr)

        # Check for NaN
        nan_count = np.sum(np.isnan(arr))
        if nan_count > 0:
            pct = 100 * nan_count / arr.size
            self.issues.append(f"[WARN] {label}: {nan_count} NaN values ({pct:.1f}%)")

        # Check for Inf
        inf_count = np.sum(np.isinf(arr))
        if inf_count > 0:
            self.issues.append(f"[WARN] {label}: {inf_count} Inf values")

        # Check for all zeros
        finite = arr[np.isfinite(arr)]
        if len(finite) > 0:
            if np.all(finite == 0):
                self.issues.append(f"[WARN] {label}: All values are zero")

            # Check for suspiciously uniform data
            if np.std(finite) < 1e-10 * np.abs(np.mean(finite)):
                self.issues.append(f"[WARN] {label}: Suspiciously uniform (std ≈ 0)")

            # Check for negative values when not expected
            if not allow_negative and np.any(finite < 0):
                self.issues.append(f"[WARN] {label}: Contains negative values")
        else:
            self.issues.append(f"[WARN] {label}: No finite values!")

        # Print stats for debugging
        if len(finite) > 0:
            print(
                f"    [STATS] {label}: min={np.min(finite):.4g}, max={np.max(finite):.4g}, "
                f"mean={np.mean(finite):.4g}, std={np.std(finite):.4g}"
            )

    def check_chi2(self, chi2_reduced: float):
        """Check if chi-squared is reasonable."""
        if chi2_reduced < 0:
            self.issues.append(f"[ERROR] χ²/dof = {chi2_reduced:.2f} is negative!")
        elif chi2_reduced < 0.5:
            self.issues.append(
                f"[WARN] χ²/dof = {chi2_reduced:.2f} is suspiciously low (overfitting?)"
            )
        elif chi2_reduced > 100:
            self.issues.append(
                f"[WARN] χ²/dof = {chi2_reduced:.2f} is very high (poor fit)"
            )
        else:
            print(f"    [OK] χ²/dof = {chi2_reduced:.2f}")

    def check_mcmc_convergence(self, sampler, flat_chain):
        """Check MCMC convergence indicators."""
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            n_steps = sampler.iteration

            # Check if chain is long enough (should be >> autocorr time)
            min_ratio = np.min(n_steps / tau)
            if min_ratio < 10:
                self.issues.append(
                    f"[WARN] Chain may not be converged: n_steps/τ = {min_ratio:.1f} < 10"
                )
            else:
                print(f"    [OK] Chain convergence: n_steps/τ = {min_ratio:.1f}")

            print(f"    [STATS] Autocorrelation times: {tau}")

        except Exception as e:
            self.issues.append(f"[WARN] Could not compute autocorrelation: {e}")

        # Check effective sample size
        if flat_chain is not None:
            n_eff = flat_chain.shape[0]
            if n_eff < 100:
                self.issues.append(
                    f"[WARN] Only {n_eff} effective samples (want > 1000)"
                )
            else:
                print(f"    [OK] Effective samples: {n_eff}")

    def check_parameter_bounds(self, params: FRBParams, log_space: bool = True):
        """Check if parameters are physically reasonable.

        Args:
            params: FRBParams object (may be in log-space)
            log_space: If True, positive-definite params (c0, zeta, tau_1ghz) are in log-space
        """
        if log_space:
            # Parameters are in log-space - check that log values are reasonable
            # (extreme values might indicate poor convergence)
            print(f"    [INFO] Parameters are in log-space (will be exponentiated)")

            # c0: log(amplitude), reasonable range roughly [-5, 5]
            if abs(params.c0) > 10:
                self.issues.append(
                    f"[WARN] log(c0) = {params.c0:.4g} is extreme (|x| > 10)"
                )

            # zeta: log(pulse width), reasonable for ms timescale
            if params.zeta < -15 or params.zeta > 5:
                self.issues.append(
                    f"[WARN] log(zeta) = {params.zeta:.4g} outside reasonable range"
                )

            # tau_1ghz: log(scattering time)
            if params.tau_1ghz < -15 or params.tau_1ghz > 5:
                self.issues.append(
                    f"[WARN] log(tau_1ghz) = {params.tau_1ghz:.4g} outside reasonable range"
                )

            # alpha is NOT in log-space, check typical range (3-5 for Kolmogorov)
            alpha = getattr(params, "alpha", 4.0)
            if alpha < 1 or alpha > 8:
                self.issues.append(
                    f"[WARN] alpha = {alpha:.2f} outside expected range [1, 8]"
                )
        else:
            # Linear space checks
            if params.c0 <= 0:
                self.issues.append(f"[WARN] c0 = {params.c0:.4g} should be positive")
            if params.zeta < 0:
                self.issues.append(
                    f"[WARN] zeta = {params.zeta:.4g} is negative (non-physical)"
                )
            if params.tau_1ghz < 0:
                self.issues.append(
                    f"[WARN] tau_1ghz = {params.tau_1ghz:.4g} is negative"
                )
            alpha = getattr(params, "alpha", 4.0)
            if alpha < 2 or alpha > 6:
                self.issues.append(
                    f"[WARN] alpha = {alpha:.2f} outside typical range [2, 6]"
                )

    def report(self):
        """Print summary of all issues found."""
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT: {self.name}")
        print(f"{'='*60}")

        if not self.issues:
            print("[✓] All checks passed!")
        else:
            print(f"[!] Found {len(self.issues)} potential issues:\n")
            for issue in self.issues:
                print(f"    {issue}")

        print(f"{'='*60}\n")
        return len(self.issues) == 0


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def create_data_overview_figure(dataset, output_path: Path, validator: FigureValidator):
    """
    Create a 3-panel overview of the data.
    Similar to what the widget would show.
    """
    print("\n[FIGURE] Creating data overview...")

    # Validate data
    validator.check_array(dataset.data, "Dynamic spectrum")
    validator.check_array(dataset.time, "Time axis")
    validator.check_array(dataset.freq, "Frequency axis")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Dynamic spectrum
    ax = axes[0]
    vmin, vmax = np.nanpercentile(dataset.data, [1, 99])
    im = ax.imshow(
        dataset.data,
        aspect="auto",
        origin="lower",
        extent=[dataset.time[0], dataset.time[-1], dataset.freq[0], dataset.freq[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [GHz]")
    ax.set_title(
        f"Dynamic Spectrum\n({dataset.data.shape[0]} ch × {dataset.data.shape[1]} bins)"
    )
    plt.colorbar(im, ax=ax, label="Intensity")

    # Panel 2: Time profile
    ax = axes[1]
    time_profile = np.nansum(dataset.data, axis=0)
    validator.check_array(time_profile, "Time profile")
    ax.plot(dataset.time, time_profile, "k-", lw=1)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Flux (summed)")
    ax.set_title("Time Profile")

    # Add peak marker
    peak_idx = np.argmax(time_profile)
    ax.axvline(
        dataset.time[peak_idx],
        color="r",
        ls=":",
        alpha=0.7,
        label=f"Peak @ {dataset.time[peak_idx]:.2f} ms",
    )
    ax.legend(fontsize=8)

    # Panel 3: Frequency spectrum
    ax = axes[2]
    freq_spec = np.nansum(dataset.data, axis=1)
    validator.check_array(freq_spec, "Frequency spectrum")
    ax.plot(freq_spec, dataset.freq, "k-", lw=1)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Flux (summed)")
    ax.set_ylabel("Frequency [GHz]")
    ax.set_title("Frequency Spectrum")

    plt.tight_layout()

    # Add sanity check annotations
    data_max = np.nanmax(dataset.data)
    data_min = np.nanmin(dataset.data)
    peak_time = dataset.time[np.argmax(np.nansum(dataset.data, axis=0))]
    sanity_text = (
        f"[SANITY CHECK]\n"
        f"Data range: [{data_min:.3g}, {data_max:.3g}]\n"
        f"Peak @ t={peak_time:.2f} ms\n"
        f"Shape: {dataset.data.shape}"
    )
    fig.text(
        0.99,
        0.01,
        sanity_text,
        fontsize=7,
        ha="right",
        va="bottom",
        family="monospace",
        alpha=0.7,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
    )

    # Save
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")
    print(
        f"    [SANITY] Data range: [{data_min:.3g}, {data_max:.3g}], Peak @ {peak_time:.2f} ms"
    )


def create_initial_guess_figure(
    dataset,
    params: FRBParams,
    model_key: str,
    output_path: Path,
    validator: FigureValidator,
):
    """
    Create a figure showing the initial guess vs data.
    This is what the interactive widget would show.
    """
    print("\n[FIGURE] Creating initial guess comparison...")

    from scat_analysis.burstfit import FRBModel

    # Create model
    model = FRBModel(
        data=dataset.data,
        time=dataset.time,
        freq=dataset.freq,
        dm_init=getattr(dataset, "dm_init", 0.0),
        df_MHz=getattr(dataset, "df_MHz", None),
    )

    # Generate model prediction
    model_dyn = model(params, model_key)
    residual = dataset.data - model_dyn

    # Validate
    validator.check_array(model_dyn, "Model prediction")
    validator.check_array(residual, "Residual")

    # Chi-squared
    chi2 = np.nansum(residual**2) / np.nansum(dataset.data**2)
    print(f"    Initial guess χ² (normalized): {chi2:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Data
    vmin, vmax = np.nanpercentile(dataset.data, [1, 99])
    im0 = axes[0, 0].imshow(
        dataset.data,
        aspect="auto",
        origin="lower",
        extent=[dataset.time[0], dataset.time[-1], dataset.freq[0], dataset.freq[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    axes[0, 0].set_title("Data", fontweight="bold")
    axes[0, 0].set_ylabel("Frequency [GHz]")
    plt.colorbar(im0, ax=axes[0, 0])

    # Model
    im1 = axes[0, 1].imshow(
        model_dyn,
        aspect="auto",
        origin="lower",
        extent=[dataset.time[0], dataset.time[-1], dataset.freq[0], dataset.freq[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    axes[0, 1].set_title(f"Model ({model_key})", fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 1])

    # Residual
    res_std = np.nanstd(residual)
    im2 = axes[1, 0].imshow(
        residual,
        aspect="auto",
        origin="lower",
        extent=[dataset.time[0], dataset.time[-1], dataset.freq[0], dataset.freq[-1]],
        vmin=-3 * res_std,
        vmax=3 * res_std,
        cmap="PuOr",
    )
    axes[1, 0].set_title(f"Residual (χ² = {chi2:.4f})", fontweight="bold")
    axes[1, 0].set_xlabel("Time [ms]")
    axes[1, 0].set_ylabel("Frequency [GHz]")
    plt.colorbar(im2, ax=axes[1, 0])

    # Time profiles
    time_data = np.nansum(dataset.data, axis=0)
    time_model = np.nansum(model_dyn, axis=0)
    axes[1, 1].plot(dataset.time, time_data, "k-", lw=1.5, alpha=0.7, label="Data")
    axes[1, 1].plot(dataset.time, time_model, "m-", lw=2, label="Model")
    axes[1, 1].set_xlabel("Time [ms]")
    axes[1, 1].set_ylabel("Intensity")
    axes[1, 1].set_title("Time Profile Comparison", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Add parameter text
    param_text = (
        f"Initial Guess Parameters:\n"
        f"  c0 = {params.c0:.4g}\n"
        f"  t0 = {params.t0:.4f} ms\n"
        f"  γ = {params.gamma:.4f}\n"
        f"  ζ = {params.zeta:.4f} ms\n"
        f"  τ_1GHz = {params.tau_1ghz:.4f} ms\n"
        f"  α = {getattr(params, 'alpha', 4.0):.2f}"
    )
    fig.text(
        0.02,
        0.02,
        param_text,
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Sanity check for initial guess quality
    model_max = np.nanmax(model_dyn)
    data_max = np.nanmax(dataset.data)
    residual_rms = np.sqrt(np.nanmean(residual**2))
    correlation = np.corrcoef(dataset.data.flatten(), model_dyn.flatten())[0, 1]

    sanity_text = (
        f"[SANITY CHECK]\n"
        f"χ² (norm): {chi2:.4f}\n"
        f"Data max: {data_max:.3g}\n"
        f"Model max: {model_max:.3g}\n"
        f"Correlation: {correlation:.3f}\n"
        f"{'✓ Good' if correlation > 0.5 else '✗ Poor'} initial guess"
    )
    fig.text(
        0.99,
        0.02,
        sanity_text,
        fontsize=8,
        ha="right",
        va="bottom",
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="lightgreen" if correlation > 0.5 else "lightcoral",
            alpha=0.5,
        ),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")
    print(
        f"    [SANITY] χ²={chi2:.4f}, Corr={correlation:.3f} {'✓' if correlation > 0.5 else '✗'}"
    )


def create_results_summary_figure(
    results: dict, dataset, output_path: Path, validator: FigureValidator
):
    """
    Create a summary figure of the MCMC results.
    """
    print("\n[FIGURE] Creating results summary...")

    best_params = results.get("best_params")
    best_key = results.get("best_key")
    flat_chain = results.get("flat_chain")
    param_names = list(results.get("param_names", []))
    gof = results.get("goodness_of_fit", {})

    # Validate - assume log-space if any positive-definite param is negative
    if best_params is not None:
        # Detect if params are in log-space (c0, zeta, tau_1ghz should be positive in linear space)
        log_space = (
            best_params.c0 < 0 or best_params.zeta < 0 or best_params.tau_1ghz < 0
        )
        validator.check_parameter_bounds(best_params, log_space=log_space)

    if gof:
        validator.check_chi2(gof.get("chi2_reduced", -1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel 1: Parameter posteriors (histograms)
    ax = axes[0, 0]
    if flat_chain is not None and len(param_names) > 0:
        n_params = min(len(param_names), flat_chain.shape[1])
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_params))
        for i in range(n_params):
            samples = flat_chain[:, i]
            validator.check_array(samples, f"Posterior {param_names[i]}")
            ax.hist(
                samples,
                bins=50,
                alpha=0.5,
                color=colors[i],
                label=f"{param_names[i]}: {np.median(samples):.3g}",
            )
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Count")
        ax.set_title("Parameter Posteriors")
        ax.legend(fontsize=7, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No chain data", ha="center", va="center")
        ax.set_axis_off()

    # Panel 2: Chain traces (first 2 params)
    ax = axes[0, 1]
    sampler = results.get("sampler")
    if sampler is not None:
        chain = sampler.get_chain()  # shape: (n_steps, n_walkers, n_params)
        n_show = min(2, chain.shape[2])
        for i in range(n_show):
            ax.plot(chain[:, ::5, i], alpha=0.3, lw=0.5)  # Plot every 5th walker
        ax.set_xlabel("Step")
        ax.set_ylabel("Parameter value")
        ax.set_title(f"MCMC Traces (first {n_show} params)")

        # Add burn-in line
        burn = results.get("chain_stats", {}).get("burn_in", 0)
        ax.axvline(burn, color="r", ls="--", label=f"Burn-in ({burn})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No sampler data", ha="center", va="center")
        ax.set_axis_off()

    # Panel 3: Model vs Data profiles
    ax = axes[0, 2]
    model_instance = results.get("model_instance")
    if model_instance is not None and best_params is not None:
        model_dyn = model_instance(best_params, best_key)
        time_data = np.nansum(dataset.data, axis=0)
        time_model = np.nansum(model_dyn, axis=0)

        ax.plot(dataset.time, time_data, "k-", lw=1.5, label="Data")
        ax.plot(dataset.time, time_model, "m--", lw=2, label=f"Model ({best_key})")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Intensity")
        ax.set_title("Best-fit Time Profile")
        ax.legend()
        ax.grid(alpha=0.3)

        # Check residual RMS
        resid = time_data - time_model
        rms = np.sqrt(np.nanmean(resid**2))
        signal = np.nanmax(time_data) - np.nanmin(time_data)
        snr = signal / rms if rms > 0 else 0
        print(f"    Time profile residual SNR: {snr:.1f}")
        if snr < 3:
            validator.issues.append(f"[WARN] Low time profile SNR: {snr:.1f}")
    else:
        ax.text(0.5, 0.5, "No model data", ha="center", va="center")
        ax.set_axis_off()

    # Panel 4: Residual histogram
    ax = axes[1, 0]
    if model_instance is not None and best_params is not None:
        model_dyn = model_instance(best_params, best_key)
        residual = dataset.data - model_dyn
        noise_std = model_instance.noise_std

        # Normalize residuals
        res_norm = residual / noise_std[:, None]
        res_flat = res_norm.flatten()
        res_flat = res_flat[np.isfinite(res_flat)]

        ax.hist(
            res_flat, bins=100, density=True, alpha=0.7, color="gray", label="Residuals"
        )

        # Overlay N(0,1)
        x = np.linspace(-4, 4, 100)
        ax.plot(x, scipy_stats.norm.pdf(x), "r-", lw=2, label="N(0,1)")

        ax.set_xlabel("Normalized residual")
        ax.set_ylabel("Density")
        ax.set_title("Residual Distribution")
        ax.legend()
        ax.set_xlim(-5, 5)

        # Check if residuals are Gaussian
        res_mean = np.mean(res_flat)
        res_std = np.std(res_flat)
        if abs(res_mean) > 0.1:
            validator.issues.append(
                f"[WARN] Residual mean = {res_mean:.3f} (should be ≈ 0)"
            )
        if abs(res_std - 1.0) > 0.3:
            validator.issues.append(
                f"[WARN] Residual std = {res_std:.3f} (should be ≈ 1)"
            )
        print(f"    Residual stats: mean={res_mean:.3f}, std={res_std:.3f}")
    else:
        ax.text(0.5, 0.5, "No model data", ha="center", va="center")
        ax.set_axis_off()

    # Panel 5: Goodness of fit summary (text)
    ax = axes[1, 1]
    ax.set_axis_off()

    summary_text = f"RESULTS SUMMARY\n{'='*40}\n\n"
    summary_text += f"Best model: {best_key}\n\n"

    if gof:
        summary_text += f"Goodness of fit:\n"
        summary_text += f"  χ²/dof = {gof.get('chi2_reduced', 'N/A'):.2f}\n"
        summary_text += f"  χ² = {gof.get('chi2', 'N/A'):.1f}\n"
        summary_text += f"  dof = {gof.get('ndof', 'N/A')}\n\n"

    if best_params is not None:
        summary_text += f"Best-fit parameters:\n"
        for name, val in asdict(best_params).items():
            if flat_chain is not None and name in param_names:
                idx = param_names.index(name)
                median = np.median(flat_chain[:, idx])
                std = np.std(flat_chain[:, idx])
                summary_text += f"  {name}: {median:.4g} ± {std:.4g}\n"
            else:
                summary_text += f"  {name}: {val:.4g}\n"

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Panel 6: Residual ACF
    ax = axes[1, 2]
    if gof and "residual_autocorr" in gof:
        acf = gof["residual_autocorr"]
        n_lags = len(acf)
        lags = np.arange(n_lags) - n_lags // 2

        ax.plot(lags, acf, "k-", lw=1)
        ax.axhline(0, color="gray", ls="--")
        ax.axhline(2 / np.sqrt(n_lags), color="r", ls=":", alpha=0.7, label="95% CI")
        ax.axhline(-2 / np.sqrt(n_lags), color="r", ls=":", alpha=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Residual ACF")
        ax.legend(fontsize=8)
        ax.set_xlim(-50, 50)
    else:
        ax.text(0.5, 0.5, "No ACF data", ha="center", va="center")
        ax.set_axis_off()

    plt.suptitle(f"MCMC Analysis Results - {best_key}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================


def main(config_file: str | None = None):
    """Main analysis function."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Default config
    if config_file is None:
        config_file = "scattering/configs/bursts/chime/casey_chime.yaml"

    print(f"\n{'='*60}")
    print(f"FRB SCATTERING ANALYSIS")
    print(f"{'='*60}")
    print(f"Config: {config_file}")
    print(f"Time: {timestamp}")
    print(f"{'='*60}\n")

    # Create validator
    validator = FigureValidator(f"Analysis {timestamp}")

    # -------------------------------------------------------------------------
    # 1. LOAD CONFIGURATION
    # -------------------------------------------------------------------------
    print("[1/6] Loading configuration...")

    try:
        config = load_config(config_file)
        print(f"    Data path: {config.path}")
        print(f"    Telescope: {config.telescope.name}")
        print(f"    DM init: {config.dm_init}")
        print(
            f"    Downsampling: {config.pipeline.f_factor}x freq, {config.pipeline.t_factor}x time"
        )
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return False

    # Setup output directory
    output_dir = config.path.parent / f"analysis_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    print(f"    Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # 2. CREATE PIPELINE AND LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[2/6] Creating pipeline and loading data...")

    try:
        pipe = BurstPipeline(
            inpath=config.path,
            outpath=output_dir,
            name=config.path.stem.split("_")[0],
            dm_init=config.dm_init,
            telescope=config.telescope,
            sampler=config.sampler,
            f_factor=config.pipeline.f_factor,
            t_factor=config.pipeline.t_factor,
            steps=config.pipeline.steps,
            nproc=4,
        )

        # Create dataset manually (for initial guess visualization)
        pipe.dataset = BurstDataset(
            inpath=pipe.inpath,
            outpath=pipe.outpath,
            name=pipe.name,
            telescope=config.telescope,
            sampler=config.sampler,
            f_factor=config.pipeline.f_factor,
            t_factor=config.pipeline.t_factor,
        )
        pipe.dataset.dm_init = pipe.dm_init
        pipe.dataset.model.dm_init = pipe.dm_init

        print(f"    Data shape: {pipe.dataset.data.shape}")
        print(
            f"    Time range: {pipe.dataset.time[0]:.2f} - {pipe.dataset.time[-1]:.2f} ms"
        )
        print(
            f"    Freq range: {pipe.dataset.freq[0]:.3f} - {pipe.dataset.freq[-1]:.3f} GHz"
        )

    except Exception as e:
        print(f"[ERROR] Failed to create pipeline: {e}")
        import traceback

        traceback.print_exc()
        return False

    # -------------------------------------------------------------------------
    # 3. CREATE DATA OVERVIEW FIGURE
    # -------------------------------------------------------------------------
    print("\n[3/6] Creating data overview figure...")

    create_data_overview_figure(
        pipe.dataset, output_dir / "01_data_overview.pdf", validator
    )

    # -------------------------------------------------------------------------
    # 4. GET INITIAL GUESS AND CREATE COMPARISON FIGURE
    # -------------------------------------------------------------------------
    print("\n[4/6] Computing initial guess...")

    try:
        # Use the widget's data-driven guess
        widget = InitialGuessWidget(pipe.dataset, model_key="M3")
        initial_params = widget.get_params()

        print(f"    Initial guess:")
        print(f"      c0 = {initial_params.c0:.4g}")
        print(f"      t0 = {initial_params.t0:.4f} ms")
        print(f"      γ = {initial_params.gamma:.4f}")
        print(f"      ζ = {initial_params.zeta:.4f}")
        print(f"      τ_1GHz = {initial_params.tau_1ghz:.4f}")
        print(f"      α = {getattr(initial_params, 'alpha', 4.0):.2f}")

        # Save to pipeline
        pipe.seed_single = initial_params

        # Create comparison figure
        create_initial_guess_figure(
            pipe.dataset,
            initial_params,
            "M3",
            output_dir / "02_initial_guess.pdf",
            validator,
        )

    except Exception as e:
        print(f"[WARN] Failed to get initial guess: {e}")
        import traceback

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # 5. RUN MCMC FITTING
    # -------------------------------------------------------------------------
    print("\n[5/6] Running MCMC fitting...")
    print("    (This may take several minutes...)")

    try:
        results = pipe.run_full(
            model_scan=True,
            diagnostics=False,  # Skip for speed
            plot=True,
            show=False,
            save=True,
        )

        print(f"\n    Best model: {results['best_key']}")
        print(f"    χ²/dof: {results['goodness_of_fit']['chi2_reduced']:.2f}")

        # Validate MCMC results
        sampler = results.get("sampler")
        flat_chain = results.get("flat_chain")
        if sampler is not None and flat_chain is not None:
            validator.check_mcmc_convergence(sampler, flat_chain)

    except Exception as e:
        print(f"[ERROR] MCMC fitting failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # -------------------------------------------------------------------------
    # 6. CREATE RESULTS SUMMARY FIGURE
    # -------------------------------------------------------------------------
    print("\n[6/6] Creating results summary figure...")

    create_results_summary_figure(
        results, pipe.dataset, output_dir / "03_results_summary.pdf", validator
    )

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------
    validator.report()  # Print validation results (informational)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Output files saved to: {output_dir}")
    print(f"  - 01_data_overview.pdf")
    print(f"  - 02_initial_guess.pdf")
    print(f"  - 03_results_summary.pdf")
    print(
        f"  + Pipeline diagnostic plots (FRB_four_panel.pdf, FRB_comp_diagnostics.pdf)"
    )
    print(f"{'='*60}\n")

    # Return True (success) if we got this far - validation issues are informational
    return True


if __name__ == "__main__":
    # Get config file from command line or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else None

    success = main(config_file)
    sys.exit(0 if success else 1)
