"""
burstfit_pipeline.py
====================

Object-oriented orchestrator for the BurstFit pipeline. This module connects
the data loading, preprocessing, fitting, diagnostics, and plotting modules
into a coherent, runnable sequence.
"""

from __future__ import annotations

import os
import logging
import warnings
import argparse
import inspect
import contextlib
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy as sp
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from .burstfit import (
    FRBModel,
    FRBFitter,
    FRBParams,
    build_priors,
    plot_dynamic,
    goodness_of_fit,
    downsample,
    gelman_rubin,
)
from .burstfit_modelselect import fit_models_bic
from .burstfit_robust import (
    subband_consistency,
    leave_one_out_influence,
    plot_influence,
    fit_subband_profiles,
    plot_subband_profiles,
    plot_subband_profiles,
    plot_subband_profiles,
    plot_subband_profiles,
    dm_optimization_check,
)
from flits.fitting.diagnostics import analyze_residuals, ResidualDiagnostics
from flits.sampler import MCMCDiagnostics
from .burstfit_nested import fit_models_evidence
from .config_utils import SamplerConfig, TelescopeConfig, load_telescope_block
from .pool_utils import build_pool
import dataclasses

# --- Setup Logging ---
log = logging.getLogger("burstfit.pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s | %(name)s] %(message)s")


###############################################################################
# 0. PLOTTING FUNCTIONS
###############################################################################


def create_four_panel_plot(
    dataset: "BurstDataset",
    results: Dict[str, Any],
    *,
    save: bool = True,
    show: bool = True,
):
    """Creates a four-panel diagnostic plot comparing data, model, and residuals."""
    log.info("Generating four-panel diagnostic plot...")

    best_p, best_key = results["best_params"], results["best_key"]
    model_instance = results["model_instance"]
    param_names = results.get("param_names", [])
    flat_chain = results.get("flat_chain")

    data, time, freq = dataset.data, dataset.time, dataset.freq
    time_centered = time - (time[0] + (time[-1] - time[0]) / 2)
    extent = [time_centered[0], time_centered[-1], freq[0], freq[-1]]

    clean_model = model_instance(best_p, best_key)
    synthetic_noise = np.random.normal(
        0.0, model_instance.noise_std[:, None], size=data.shape
    )
    synthetic_data = clean_model + synthetic_noise
    residual = data - synthetic_data

    def _normalize_panel_data(arr_2d, off_pulse_data):
        mean_off, std_off = np.nanmean(off_pulse_data), np.nanstd(off_pulse_data)
        if std_off < 1e-9:
            return arr_2d
        arr_norm = (arr_2d - mean_off) / std_off
        peak = np.nanmax(arr_norm)
        return arr_norm / peak if peak > 0 else arr_norm

    q = data.shape[1] // 4
    data_off_pulse = data[:, np.r_[0:q, -q:0]]
    
    # Calculate global normalization stats from Data
    mean_off = np.nanmean(data_off_pulse)
    std_off = np.nanstd(data_off_pulse)
    if std_off < 1e-9: std_off = 1.0 # Prevent div/0
    
    # Pre-normalize data to get peak S/N
    data_snr = (data - mean_off) / std_off
    peak_snr = np.nanmax(data_snr)
    if peak_snr <= 0: peak_snr = 1.0
    
    def _apply_norm(arr, subtract_mean=True):
        if subtract_mean:
            return (arr - mean_off) / std_off / peak_snr
        else:
            return arr / std_off / peak_snr

    data_norm = _apply_norm(data, subtract_mean=True)
    model_norm = _apply_norm(clean_model, subtract_mean=True)
    synthetic_norm = _apply_norm(synthetic_data, subtract_mean=True)
    residual_norm = _apply_norm(residual, subtract_mean=False) # Residuals already centered

    # Calculate global Y-limits for Time Series (Top Panels)
    # We want valid limits covering min/max of everything
    all_ts = [np.nansum(p, axis=0) for p in [data_norm, model_norm, synthetic_norm, residual_norm]]
    ts_min = min(np.min(t) for t in all_ts if t.size > 0)
    ts_max = max(np.max(t) for t in all_ts if t.size > 0)
    y_range = ts_max - ts_min
    ts_ylim = (ts_min - 0.05 * y_range, ts_max + 0.05 * y_range)

    # Calculate global X-limits for Spectrum (Side Panels)
    # These plot Flux (x) vs Freq (y), so we need shared X-limits
    all_sp = [np.nansum(p, axis=1) for p in [data_norm, model_norm, synthetic_norm, residual_norm]]
    sp_min = min(np.min(s) for s in all_sp if s.size > 0)
    sp_max = max(np.max(s) for s in all_sp if s.size > 0)
    x_range = sp_max - sp_min
    sp_xlim = (sp_min - 0.05 * x_range, sp_max + 0.05 * x_range)

    # Create 2 rows, 10 columns: Data, Model, Synth, Resid, FLOW logic
    # Logical Structure: 5 panels. Each panel uses 2 grid columns (Left for Time/WF, Right for Spec/Empty).
    # Total Columns = 5 * 2 = 10.
    # Create 2 rows, 8 columns: Data, Model, Synth, Resid
    # For waterfall + spectrum, we use GridSpec or nested:
    # 4 pairs of columns = 8 columns
    fig, axes = plt.subplots(
        nrows=2,
        ncols=8,
        gridspec_kw={"height_ratios": [1, 2.5], "width_ratios": [2, 0.5] * 4},
        figsize=(24, 8),
    )

    panel_data = [
        (data_norm, "Data", r"$\mathbf{I}_{\rm data}$"),
        (model_norm, "Model", r"$\mathbf{I}_{\rm model}$"),
        (synthetic_norm, "Model + Noise", r"$\mathbf{I}_{\rm model} + \mathbf{N}$"),
        (residual_norm, "Residual", r"$\mathbf{I}_{\rm residual}$"),
    ]

    for i, (panel_ds, title, label) in enumerate(panel_data):
        col_idx = i * 2
        ax_ts, ax_sp, ax_wf = axes[0, col_idx], axes[1, col_idx + 1], axes[1, col_idx]

        ts = np.nansum(panel_ds, axis=0)
        sp = np.nansum(panel_ds, axis=1)

        ax_ts.step(time_centered, ts, where="mid", c="k", lw=1.5, label=label)
        ax_ts.legend(loc="upper right", fontsize=14, frameon=False)
        ax_ts.set_ylim(ts_ylim) # Enforce shared Y-limits

        cmap = "coolwarm" if title == "Residual" else "plasma"
        if title == "Residual":
            vmax = np.nanmax(np.abs(panel_ds)) # Symmetric around 0
            vmin = -vmax
        else:
             vmin = np.nanpercentile(panel_ds, 1)
             vmax = np.nanpercentile(panel_ds, 99.5)
             
        ax_wf.imshow(
            panel_ds,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
            origin="lower",
        )

        ax_sp.step(sp, freq, where="mid", c="k", lw=1.5)
        ax_sp.set_xlim(sp_xlim) # Enforce shared X-limits

        ax_ts.set_yticks([]) 
        ax_ts.tick_params(axis="x", labelbottom=False)
        ax_ts.set_xlim(extent[0], extent[1])
        ax_sp.set_xticks([])
        ax_sp.tick_params(axis="y", labelleft=False)
        ax_sp.set_ylim(extent[2], extent[3])
        ax_wf.set_xlabel("Time [ms]")
        if i == 0:
            ax_wf.set_ylabel("Frequency [GHz]")
        else:
            ax_wf.tick_params(axis="y", labelleft=False)
        axes[0, col_idx + 1].axis("off")

    # Adjust top to make room for header without cutoff
    # Tighten wspace to align with hspace (user requested equivalent margins)
    plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.83, bottom=0.15, left=0.05, right=0.98)
    
    # --- Detailed Header Table Implementation (Clean/Scientific) ---
    # Aligned with panels, no boxes, clean text.
    
    # 1. Metadata
    fname = dataset.inpath.name
    tns_name = "FRB 20190425A" if "casey" in dataset.name.lower() else "FRB (Unknown)"
    person_name = dataset.name.split('_')[0].upper() # e.g. CASEY
    observatory = "CHIME/FRB" if "chime" in fname.lower() else "DSA-110"
    
    meta_text = (
        f"{tns_name} / {person_name}\n"
        f"Observatory: {observatory}"
    )
    
    # 2. Model Selection
    model_text = "Model Selection:\n"
    if "all_results" in results:
        res_all = results["all_results"]
        keys = list(res_all.keys())
        keys.sort(reverse=True) # M3, M2, M1
        best_z = max(float(res_all[k].log_evidence) for k in keys) if keys else 0
        
        for k in keys:
            z = float(res_all[k].log_evidence)
            dz = z - best_z
            # Clean format: M3: logZ = -562 (d=0)
            mark = r"$\mathbf{\ast}$" if k == best_key else " "
            # Use LaTeX for cleaner look if possible, or just text
            model_text += f"{mark} {k}: $\ln{{Z}}={z:.0f}$ ($\Delta={dz:.0f}$)\n"
    else:
        model_text += f"{best_key} (Selected)\n(Comparison N/A)"

    # 3. Goodness of Fit
    gof = results.get("goodness_of_fit", {})
    chi2 = gof.get("chi2_reduced", np.nan)
    r2 = gof.get("r_squared", np.nan)
    quality = gof.get("quality_flag", "UNKNOWN")
    
    # Use color only for the status word
    gof_header = "Goodness of Fit:"
    gof_body = (
        f"$\chi^2_\\nu = {chi2:.2f}$\n"
        f"$R^2   = {r2:.3f}$"
    )
    
    # 4. Parameters
    param_header = "Best Fit Parameters:"
    param_lines = []
    for i, name in enumerate(param_names):
        # Handle nan chain
        vals = flat_chain[:, i]
        if np.all(np.isnan(vals)):
            val, err = getattr(best_p, name, np.nan), 0.0
        else:
            val = np.median(vals)
            err = np.std(vals)
        
        # Smart Formatting
        if abs(val) < 0.001 and val != 0:
            s_val = f"{val:.1e}"
        else:
            s_val = f"{val:.4g}"
            
        param_lines.append(f"{name} = ${s_val} \pm {err:.1g}$")
    param_text = "\n".join(param_lines)

    # --- Rendering Text Blocks ---
    # Y position for all blocks (moved down slightly to ensure safe margin)
    header_y = 0.94 
    body_y = 0.91
    fontsize_head = 12
    fontsize_body = 10
    
    # Block 1 (Left): Meta (Larger, bold)
    fig.text(0.05, header_y, meta_text, fontsize=14, weight='bold', va='top', fontfamily='sans-serif')
    
    # Block 2: Model
    fig.text(0.28, header_y, "Model Selection", fontsize=fontsize_head, weight='bold', va='top')
    fig.text(0.28, body_y, model_text.replace("Model Selection:\n", ""), fontsize=fontsize_body, va='top')
    
    # Block 3: GoF
    fig.text(0.52, header_y, gof_header, fontsize=fontsize_head, weight='bold', va='top')
    fig.text(0.52, body_y, gof_body, fontsize=fontsize_body, va='top')
    # Status below params or specifically colored
    q_color = "green" if quality == "PASS" else ("red" if quality == "FAIL" else "orange")
    fig.text(0.52, body_y - 0.04, f"Status: {quality}", fontsize=fontsize_body, weight='bold', color=q_color, va='top')
             
    # Block 4: Params
    fig.text(0.75, header_y, param_header, fontsize=fontsize_head, weight='bold', va='top')
    fig.text(0.75, body_y, param_text, fontsize=fontsize_body, va='top', fontfamily='monospace')
             
    # No separator line requested.



    if save:
        output_path = os.path.join(dataset.outpath, f"{dataset.name}_four_panel.pdf")
        log.info(f"Saving 4-panel plot to {output_path}")
        fig.savefig(output_path)  # , bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def create_sixteen_panel_plot(
    dataset: "BurstDataset",
    results: Dict[str, Any],
    *,
    save: bool = True,
    show: bool = True,
):
    """Creates a comprehensive 16-panel diagnostic plot summarizing the fit."""
    log.info("Generating 16-panel comprehensive diagnostics plot...")

    best_key, best_p = results["best_key"], results["best_params"]
    sampler, gof = results.get("sampler"), results.get("goodness_of_fit")
    chain_stats, flat_chain = results.get("chain_stats", {}), results["flat_chain"]
    param_names, diag_results = results["param_names"], results.get("diagnostics", {})
    model_instance = results["model_instance"]

    model_dyn = model_instance(best_p, best_key)
    residual = dataset.data - model_dyn

    fig, axes = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True)
    ax = axes.ravel()

    # Panels 0-2: Data, Model, Residual
    vmin, vmax = np.percentile(dataset.data, [1, 99])
    plot_dynamic(
        ax[0],
        dataset.data,
        dataset.time,
        dataset.freq,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    ax[0].set_title("Data")
    plot_dynamic(
        ax[1],
        model_dyn,
        dataset.time,
        dataset.freq,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    ax[1].set_title(f"Model ({best_key})")
    res_std = np.std(residual)
    plot_dynamic(
        ax[2],
        residual,
        dataset.time,
        dataset.freq,
        vmin=-3 * res_std,
        vmax=3 * res_std,
        cmap="coolwarm",
    )
    ax[2].set_title("Residual")

    # Panel 3: Residual Histogram
    res_norm = residual / model_instance.noise_std[:, None]
    # Filter out NaNs and Infs (from masked channels with noise_std=0)
    res_norm_clean = res_norm[np.isfinite(res_norm)]
    ax[3].hist(
        res_norm_clean, bins=100, density=True, color="gray", label="Residuals"
    )
    x_pdf = np.linspace(-4, 4, 100)
    ax[3].plot(x_pdf, sp.stats.norm.pdf(x_pdf), "m-", lw=2, label="N(0,1)")
    ax[3].set_title("Residual Distribution")
    ax[3].legend()

    # Panel 4-5: Time Profile and Spectrum
    ax[4].plot(dataset.time, np.sum(dataset.data, axis=0), "k-", label="Data")
    ax[4].plot(dataset.time, np.sum(model_dyn, axis=0), "m--", lw=2, label="Model")
    ax[4].set_title("Time Profile")
    ax[4].legend()
    ax[4].set_xlabel("Time [ms]")
    ax[5].plot(dataset.freq, np.sum(dataset.data, axis=1), "k-", label="Data")
    ax[5].plot(dataset.freq, np.sum(model_dyn, axis=1), "m--", lw=2, label="Model")
    ax[5].set_title("Frequency Spectrum")
    ax[5].legend()
    ax[5].set_xlabel("Frequency [GHz]")

    # Panel 6-8: Diagnostics
    if diag_results.get("influence") is not None:
        plot_influence(ax[6], diag_results["influence"], dataset.freq)
    else:
        ax[6].text(0.5, 0.5, "Influence\nNot Run", ha="center", va="center")
        ax[6].set_axis_off()

    if diag_results.get("subband_2d") is not None:
        p_name, s_res, _ = diag_results["subband_2d"]
        if p_name:
            valid = [
                (i, v, e) for i, (v, e) in enumerate(s_res) if np.isfinite(v) and e > 0
            ]
            if valid:
                idx, vals, errs = zip(*valid)
                edges = np.linspace(0, dataset.freq.size, len(s_res) + 1, dtype=int)
                band_centers = np.array(
                    [
                        dataset.freq[edges[i] : edges[i + 1]].mean()
                        for i in range(len(s_res))
                    ]
                )[list(idx)]
                ax[7].errorbar(
                    band_centers,
                    vals,
                    yerr=errs,
                    fmt="o",
                    c="k",
                    capsize=3,
                    label=f"Sub-band {p_name}",
                )
                global_val = getattr(best_p, p_name)
                ax[7].axhline(global_val, color="m", ls="--", label="Global Fit")
                ax[7].legend()
                ax[7].set_ylabel(p_name)
    else:
        ax[7].text(0.5, 0.5, "2D Sub-band\nNot Run", ha="center", va="center")
        ax[7].set_axis_off()

    if diag_results.get("profile1d") is not None:
        plot_subband_profiles(ax[8], *diag_results["profile1d"], best_p, fontsize=10)
    else:
        ax[8].text(0.5, 0.5, "1D Profile\nNot Run", ha="center", va="center")
        ax[8].set_axis_off()

    # Panel 9-10: MCMC chains
    if sampler is not None:
        chain = sampler.get_chain()
        burn = chain_stats.get("burn_in", 0)
        ax[9].plot(chain[:, ::10, 0], "k", alpha=0.3)
        ax[9].axvline(burn, color="m", ls="--")
        ax[9].set_title(f"Trace: {param_names[0]}")
    else:
        ax[9].text(0.5, 0.5, "MCMC Traces\nN/A (Nested Sampling)", ha="center", va="center")
        ax[9].set_axis_off()
    ax[9].set_title(f"Trace: {param_names[0]}")
    ax[9].set_xlabel("Step")
    if len(param_names) > 1:
        corr = np.corrcoef(flat_chain.T)
        np.fill_diagonal(corr, 0)
        i, j = np.unravel_index(np.abs(corr).argmax(), corr.shape)
        ax[10].scatter(flat_chain[:, i], flat_chain[:, j], s=1, alpha=0.1, c="k")
        ax[10].set_xlabel(param_names[i])
        ax[10].set_ylabel(param_names[j])
        ax[10].set_title(f"Highest Correlation (ρ={corr[i,j]:.2f})")

    # Panel 11-12: ACF and DM Check
    if gof:
        lags_ms = (
            np.arange(len(gof["residual_autocorr"]))
            - len(gof["residual_autocorr"]) // 2
        ) * dataset.dt_ms
        ax[11].plot(lags_ms, gof["residual_autocorr"], "k-", label="Data")
        # Posterior predictive residual ACF envelope (fast PPC via noise-only)
        try:
            n_ppc = 50
            noise_std = model_instance.noise_std
            acfs = []
            for _ in range(n_ppc):
                noise = np.random.normal(
                    0.0, noise_std[:, None], size=dataset.data.shape
                )
                resid_ppc = np.sum(noise, axis=0)
                resid_ppc -= np.mean(resid_ppc)
                acf_ppc = np.correlate(resid_ppc, resid_ppc, mode="same")
                center_val = acf_ppc[len(acf_ppc) // 2]
                if center_val > 0:
                    acf_ppc = acf_ppc / center_val
                acfs.append(acf_ppc)
            acfs = np.asarray(acfs)
            ppc_med = np.median(acfs, axis=0)
            ppc_lo = np.percentile(acfs, 5, axis=0)
            ppc_hi = np.percentile(acfs, 95, axis=0)
            ax[11].plot(lags_ms, ppc_med, color="m", lw=1.5, label="PPC median")
            ax[11].fill_between(
                lags_ms, ppc_lo, ppc_hi, color="m", alpha=0.15, label="PPC 5–95%"
            )
        except Exception as e:
            log.warning(f"PPC ACF generation failed: {e}")
        ax[11].set_title("Residual ACF")
        ax[11].set_xlabel("Lag [ms]")
        ax[11].legend(loc="upper right", fontsize=9)
    if diag_results.get("dm_check") is not None:
        dms, snrs = diag_results["dm_check"]
        ax[12].plot(dms, snrs, "o-k")
        ax[12].set_title("DM Optimization")
        ax[12].set_xlabel(r"ΔDM (pc cm$^{-3}$)")

    # Panels 13-15: Text Summaries
    for i in [13, 14, 15]:
        ax[i].set_axis_off()
    if gof:
        # Format validation summary
        quality = gof.get('quality_flag', 'UNKNOWN')
        chi2 = gof.get('chi2_reduced', 0.0)
        r2 = gof.get('r_squared', 0.0)
        p_norm = gof.get('normality_pvalue', 0.0)
        bias = gof.get('bias_nsigma', 0.0)
        dw = gof.get('durbin_watson', 0.0)
        
        gof_text = (
            f"V&V STATUS: {quality}\n"
            f"-----------------\n"
            f"χ²/dof = {chi2:.2f}\n"
            f"R²     = {r2:.3f}\n"
            f"Normality p = {p_norm:.1e}\n"
            f"Bias σ      = {bias:.1f}\n"
            f"Durbin-Watson = {dw:.2f}"
        )
        # Determine color based on quality
        color = "black"
        if quality == "FAIL": color = "red"
        elif quality == "MARGINAL": color = "orange"
        elif quality == "PASS": color = "green"

        ax[13].text(
            0.05,
            0.95,
            gof_text,
            va="top",
            fontfamily="monospace",
            fontsize=11,
            color=color,
            weight="bold" if quality != "UNKNOWN" else "normal"
        )
    p_summary = "Best Fit (Median & 1σ):\n" + "\n".join(
        [
            f"{n}: {np.median(flat_chain[:,i]):.3f} ± {np.std(flat_chain[:,i]):.3f}"
            for i, n in enumerate(param_names)
        ]
    )
    ax[14].text(0.05, 0.95, p_summary, va="top", fontfamily="monospace", fontsize=12)
    # Δν_d diagnostic from τ (report only)
    try:
        alpha = getattr(best_p, "alpha", 4.4)
        tau1 = getattr(best_p, "tau_1ghz", 0.0)
        if tau1 > 0:
            f = dataset.freq
            tau_ms = tau1 * (f / 1.0) ** (-alpha)
            tau_sec = tau_ms * 1e-3
            dnu_hz = 1.16 / (2.0 * np.pi * np.clip(tau_sec, 1e-12, None))
            dnu_mhz = dnu_hz / 1e6
            dnu_center = float(np.median(dnu_mhz))
            dnu_min, dnu_max = float(np.min(dnu_mhz)), float(np.max(dnu_mhz))
            diag_text = (
                f"Δν_d (implied, MHz)\n"
                f"center: {dnu_center:.3g}\n"
                f"band: [{dnu_min:.3g}, {dnu_max:.3g}]\n"
                f"(2π τ Δν_d ≈ 1.16)"
            )
        else:
            diag_text = "Δν_d (implied): n/a"
    except Exception as e:
        diag_text = f"Δν_d diagnostic failed: {e}"
    ax[15].text(
        0.05,
        0.95,
        f"File:\n{dataset.inpath.name}\n\n{diag_text}",
        va="top",
        fontfamily="monospace",
        fontsize=12,
    )

    fig.suptitle(
        f"Comprehensive Fit Diagnostics: {dataset.inpath.name}",
        fontsize=24,
        weight="bold",
    )
    if save:
        comp_diag_path = os.path.join(
            dataset.outpath, f"{dataset.name}_comp_diagnostics.pdf"
        )
        fig.savefig(comp_diag_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


###############################################################################
# 1. DATASET LOADER
###############################################################################
class BurstDataset:
    """Loads and preprocesses a burst from a .npy file."""

    def __init__(
        self,
        inpath: str | Path,
        outpath: str | Path,
        *,
        name: str = "FRB",
        telescope: TelescopeConfig | None = None,
        sampler: SamplerConfig | None = None,
        f_factor: int = 1,
        t_factor: int = 1,
        outer_trim: float = 0.45,
        smooth_ms: float = 0.1,
        center_burst: bool = True,
        flip_freq: bool = True,
        lazy: bool = False,
    ):
        self.inpath = Path(inpath)
        self.outpath = Path(outpath)
        self.name = name
        if telescope is None:
            raise ValueError("telescope configuration must be provided")
        self.telescope = telescope
        self.sampler = sampler
        self.f_factor, self.t_factor = f_factor, t_factor
        self.outer_trim = outer_trim if outer_trim is not None else 0.45
        self.smooth_ms = smooth_ms
        self.center_burst, self.flip_freq = center_burst, flip_freq
        self.data = self.freq = self.time = self.df_MHz = self.dt_ms = self.model = None
        if not lazy:
            self.load()

    def load(self):
        if self.data is not None:
            return
        raw = self._load_raw()
        if self.flip_freq:
            raw = np.flipud(raw)

        # Build axes for the raw data to use in preprocessing
        raw_freq, raw_time, _, _ = self._build_axes(raw.shape, f_factor=1, t_factor=1)
        ds = self._bandpass_correct(raw, raw_time)
        ds = self._trim_buffer(ds)
        self.data = self._downsample_and_renormalize(ds)

        # Re-build axes for final downsampled data shape
        self.freq, self.time, self.df_MHz, self.dt_ms = self._build_axes(
            self.data.shape
        )

        if self.center_burst:
            self._centre_burst()

        self.model = FRBModel(
            time=self.time, freq=self.freq, data=self.data, df_MHz=self.df_MHz
        )

    def _load_raw(self):
        if not self.inpath.exists():
            raise FileNotFoundError(f"Data not found: {self.inpath}")
        try:
            data = np.load(self.inpath)
            return np.nan_to_num(data.astype(np.float64))
        except Exception as e:
            raise IOError(f"Failed to load {self.inpath}: {e}")

    def _build_axes(self, shape, f_factor=None, t_factor=None):
        f_factor = f_factor if f_factor is not None else self.f_factor
        t_factor = t_factor if t_factor is not None else self.t_factor

        # Get raw shape from config, not from current array shape
        p = self.telescope
        n_ch_raw = p.n_ch_raw if p.n_ch_raw is not None else shape[0] * f_factor

        df_MHz = p.df_MHz_raw * f_factor
        dt_ms = p.dt_ms_raw * t_factor

        final_n_ch = shape[0]
        final_n_t = shape[1]

        # Check if data has high freq at row 0 (freq_descending)
        freq_desc = getattr(p, 'freq_descending', False)
        if freq_desc:
            # data[0] = f_max, data[-1] = f_min
            freq = np.linspace(p.f_max_GHz, p.f_min_GHz, final_n_ch)
        else:
            # data[0] = f_min, data[-1] = f_max (default/legacy)
            freq = np.linspace(p.f_min_GHz, p.f_max_GHz, final_n_ch)
        time = np.arange(final_n_t) * dt_ms
        return freq, time, df_MHz, dt_ms

    def _bandpass_correct(self, arr, time_axis):
        q = time_axis.size // 4
        off_pulse_idx = np.r_[0:q, -q:0]
        mu = np.nanmean(arr[:, off_pulse_idx], axis=1, keepdims=True)
        sig = np.nanstd(arr[:, off_pulse_idx], axis=1, keepdims=True)
        sig[sig < 1e-9] = np.nan
        return np.nan_to_num((arr - mu) / sig, nan=0.0)

    def _trim_buffer(self, arr):
        n_trim = int(self.outer_trim * arr.shape[1])
        return arr[:, n_trim:-n_trim] if n_trim > 0 else arr

    def _downsample_and_renormalize(self, arr):
        ds_arr = downsample(arr, self.f_factor, self.t_factor)
        # Do NOT normalize by peak. Keep units as S/N (z-score from bandpass_correct).
        # peak = np.nanmax(ds_arr)
        # return ds_arr / peak if peak > 0 else ds_arr
        return ds_arr

    def _centre_burst(self):
        prof = np.nansum(self.data, axis=0)
        if self.smooth_ms > 0 and self.dt_ms > 0:
            sigma_samps = (self.smooth_ms / 2.355) / self.dt_ms
            prof = gaussian_filter1d(prof, sigma=sigma_samps)
        shift = self.data.shape[1] // 2 - np.argmax(prof)
        self.data = np.roll(self.data, shift, axis=1)


###############################################################################
# 2. DIAGNOSTICS WRAPPER
###############################################################################
class BurstDiagnostics:
    """A container for running and storing all post-fit diagnostic checks."""

    def __init__(self, dataset: "BurstDataset", results: Dict[str, Any]):
        self.dataset = dataset
        self.results_in = results
        self.diag_results: Dict[str, Any] = {}

    def run_all(self, sb_steps: int = 500, pool=None):
        log.info("Running all post-fit diagnostics...")
        best_p = self.results_in["best_params"]
        best_key = self.results_in["best_key"]
        dm_init = self.results_in["dm_init"]
        model_instance = self.results_in["model_instance"]
        model_dyn = model_instance(best_p, best_key)

        self.diag_results["influence"] = leave_one_out_influence(
            self.dataset.data, model_dyn
        )
        self.diag_results["dm_check"] = dm_optimization_check(
            self.dataset.data, self.dataset.freq, self.dataset.time, dm_init
        )
        self.diag_results["subband_2d"] = subband_consistency(
            self.dataset.data,
            self.dataset.freq,
            self.dataset.time,
            dm_init,
            self.dataset.df_MHz,
            best_p,
            model_key=best_key,
            n_steps=sb_steps,
            pool=pool,
        )
        self.diag_results["profile1d"] = fit_subband_profiles(
            self.dataset, best_p, dm_init
        )

        # --- NEW: Rigorous Residual Analysis ---
        log.info("Running rigorous residual analysis...")
        res_diag = analyze_residuals(
            data=self.dataset.data,
            model_pred=model_dyn,
            noise_std=model_instance.noise_std,
            output_path=str(self.dataset.outpath / f"{self.dataset.name}_residuals_detailed.png")
        )
        self.diag_results["residual_analysis"] = res_diag
        
        log.info("Diagnostics complete.")
        return self.diag_results


###############################################################################
# 3. PIPELINE FAÇADE
###############################################################################

def refine_initial_guess_mle(model: FRBModel, init_guess: FRBParams) -> FRBParams:
    """
    Use MLE (Nelder-Mead) to refine the initial guess before MCMC.

    Optimizes primarily for tau_1ghz, alpha, t0, and c0.
    Keeps nuisance parameters (gamma, delta_dm) fixed or tightly constrained.
    """
    log.info("Refining initial guess via MLE (Nelder-Mead)...")

    # Parameters to float: [tau, alpha, t0, c0]
    # We work in log-space for positive params to ensure positivity
    x0 = [
        np.log(max(init_guess.tau_1ghz, 1e-4)),  # log tau
        init_guess.alpha,                        # alpha (linear)
        init_guess.t0,                           # t0 (linear)
        np.log(max(init_guess.c0, 1e-4))         # log c0
    ]

    def obj_func(theta):
        ln_tau, alpha, t0, ln_c0 = theta
        
        # Constraints
        if not (0.1 < alpha < 8.0): return 1e20 # Reasonable alpha bounds
        
        tau_val = np.exp(ln_tau)
        c0_val = np.exp(ln_c0)
        
        # Build params
        p = FRBParams(
            c0=c0_val,
            t0=t0,
            gamma=init_guess.gamma, # Fixed
            zeta=init_guess.zeta,   # Fixed
            tau_1ghz=tau_val,
            alpha=alpha,
            delta_dm=init_guess.delta_dm # Fixed
        )
        
        # Negative Log Likelihood
        # Add simple priors to prevent runaway
        nll = -model.log_likelihood(p, "M3")
        return nll

    try:
        res = minimize(obj_func, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-2})
        
        if res.success or res.message:
            log.info(f"MLE Refinement finished: {res.message}")
            
            ln_tau, alpha, t0, ln_c0 = res.x
            refined_params = FRBParams(
                c0=np.exp(ln_c0),
                t0=t0,
                gamma=init_guess.gamma,
                zeta=init_guess.zeta,
                tau_1ghz=np.exp(ln_tau),
                alpha=alpha,
                delta_dm=init_guess.delta_dm
            )
            
            log.info(f"  tau:   {init_guess.tau_1ghz:.3f} -> {refined_params.tau_1ghz:.3f} ms")
            log.info(f"  alpha: {init_guess.alpha:.3f} -> {refined_params.alpha:.3f}")
            log.info(f"  t0:    {init_guess.t0:.3f} -> {refined_params.t0:.3f} ms")
            return refined_params
        else:
            log.warning("MLE refinement did not converge, using original guess.")
            return init_guess
            
    except Exception as e:
        log.warning(f"MLE refinement failed with error: {e}. using original guess.")
        return init_guess


class BurstPipeline:
    """Main orchestrator for the fitting pipeline."""

    def __init__(
        self,
        inpath: str | Path,
        outpath: str | Path,
        name: str,
        *,
        dm_init: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the pipeline.

        Args:
            name: FRB name
            inpath: Path to the input .npy data file.
            outpath: Path to the output files.
            dm_init: Initial dispersion measure for the data.
            **kwargs: Keyword arguments for pipeline configuration. These are
                      intelligently split between BurstDataset and the pipeline.
        """
        self.inpath = inpath
        self.outpath = Path(outpath)
        self.outpath.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.dm_init = dm_init

        # --- FIX: Intelligently separate kwargs for different components ---

        # Get the names of all valid arguments for the BurstDataset constructor
        dataset_params = inspect.signature(BurstDataset).parameters
        dataset_arg_names = list(dataset_params.keys())

        # Create a dictionary with only the kwargs that BurstDataset accepts
        self.dataset_kwargs = {
            k: v for k, v in kwargs.items() if k in dataset_arg_names
        }

        # Store the remaining kwargs for the pipeline itself (e.g., 'steps')
        self.pipeline_kwargs = {
            k: v for k, v in kwargs.items() if k not in dataset_arg_names
        }

        # Create the multiprocessing pool
        self.pool = build_pool(
            self.pipeline_kwargs.get("nproc"),
            auto_ok=self.pipeline_kwargs.get("yes", False),
        )

        # Optional seed init-guess
        self.seed_single: FRBParams | None = None
        self.seed_multi: dict[str, float] | None = None
        init_guess_path = self.pipeline_kwargs.get(
            "init_guess"
        ) or self.pipeline_kwargs.get("init_guess_path")
        if init_guess_path:
            try:
                import json

                with (
                    Path(init_guess_path).expanduser().open("r", encoding="utf-8") as fh
                ):
                    seed = json.load(fh)
                mk = seed.get("model_key", "M3")
                if mk == "M3":
                    self.seed_single = FRBParams(
                        c0=float(seed.get("c0", 1.0)),
                        t0=float(seed.get("t0", 0.0)),
                        gamma=float(seed.get("gamma", -1.6)),
                        zeta=float(seed.get("zeta", 0.1)),
                        tau_1ghz=float(seed.get("tau_1ghz", 0.1)),
                        alpha=float(seed.get("alpha", 4.4)),
                        delta_dm=float(seed.get("delta_dm", 0.0)),
                    )
                elif mk == "M3_multi":
                    shared = seed.get("shared", {})
                    comp = seed.get("components", [])
                    d: dict[str, float] = {}
                    d["gamma"] = float(shared.get("gamma", -1.6))
                    d["tau_1ghz"] = float(shared.get("tau_1ghz", 0.1))
                    d["alpha"] = float(shared.get("alpha", 4.4))
                    d["delta_dm"] = float(shared.get("delta_dm", 0.0))
                    for i, c in enumerate(comp, start=1):
                        d[f"c0_{i}"] = float(c.get("c0", 1.0))
                        d[f"t0_{i}"] = float(c.get("t0", 0.0))
                        d[f"zeta_{i}"] = float(c.get("zeta", 0.1))
                    self.seed_multi = d
                    # ensure ncomp matches seed
                    self.pipeline_kwargs["ncomp"] = max(1, len(comp))
            except Exception as e:
                warnings.warn(f"Failed to read init-guess '{init_guess_path}': {e}")
                
        # Resolve telescope config if it's a string
        if "telescope" in self.dataset_kwargs and isinstance(self.dataset_kwargs["telescope"], str):
            tel_name = self.dataset_kwargs["telescope"]
            telcfg_path = self.dataset_kwargs.get("telcfg_path", "scattering/configs/telescopes.yaml")
            try:
                self.dataset_kwargs["telescope"] = load_telescope_block(telcfg_path, tel_name)
            except Exception as e:
                raise ValueError(f"Failed to load telescope config for '{tel_name}' from '{telcfg_path}': {e}")

    def run_full(
        self,
        model_scan=True,
        diagnostics=True,
        plot=True,
        save=True,
        show=True,
        model_keys=("M0", "M1", "M2", "M3"),
        **kwargs,
    ):
        """Main pipeline execution flow."""
        with self.pool or contextlib.nullcontext(self.pool) as pool:
            # --- FIX: Use the filtered kwargs to instantiate BurstDataset ---
            self.dataset = BurstDataset(
                self.inpath, self.outpath, **self.dataset_kwargs
            )
            self.dataset.model.dm_init = self.dm_init

            # Optional DM refinement via phase-coherence method
            if self.pipeline_kwargs.get('refine_dm', False):
                log.info("DM refinement enabled, running phase-coherence estimation...")
                try:
                    from .dm_preprocessing import refine_dm_init
                    catalog_dm = self.dm_init  # Original value from config/bursts.yaml
                    
                    self.dm_init = refine_dm_init(
                        dataset=self.dataset,
                        catalog_dm=catalog_dm,
                        enable_dm_estimation=True,
                        dm_search_window=self.pipeline_kwargs.get('dm_search_window', 5.0),
                        dm_grid_resolution=self.pipeline_kwargs.get('dm_grid_resolution', 0.01),
                        n_bootstrap=self.pipeline_kwargs.get('dm_n_bootstrap', 200),
                    )
                    
                    # Update model's dm_init
                    self.dataset.model.dm_init = self.dm_init
                    log.info(f"✓ DM refined: {catalog_dm:.3f} → {self.dm_init:.3f} pc/cm³")
                except Exception as e:
                    log.error(f"DM refinement failed: {e}")
                    log.info(f"Continuing with catalog DM: {self.dm_init:.3f} pc/cm³")

            n_steps = self.pipeline_kwargs.get("steps", 2000)

            # Seed initial guess from file if provided
            if self.seed_single is not None:
                init_guess = self.seed_single
            else:
                init_guess = self._get_initial_guess(self.dataset.model)

            # Automated MLE Refinement of Initial Guess
            if self.pipeline_kwargs.get("auto_guess", True):
                init_guess = refine_initial_guess_mle(self.dataset.model, init_guess)

            # Configure priors/likelihood controls
            alpha_fixed = self.pipeline_kwargs.get("alpha_fixed")
            alpha_mu = self.pipeline_kwargs.get("alpha_mu", 4.4)
            alpha_sigma = self.pipeline_kwargs.get("alpha_sigma", 0.6)
            delta_dm_sigma = self.pipeline_kwargs.get("delta_dm_sigma", 0.1)
            likelihood_kind = self.pipeline_kwargs.get("likelihood", "gaussian")
            studentt_nu = float(self.pipeline_kwargs.get("studentt_nu", 5.0))
            sample_log_params = bool(
                self.pipeline_kwargs.get("sample_log_params", True)
            )

            # Components
            ncomp = int(self.pipeline_kwargs.get("ncomp", 1))
            auto_components = bool(self.pipeline_kwargs.get("auto_components", False))

            if model_scan and ncomp == 1:
                sampler_name = self.pipeline_kwargs.get("fitting_method", "emcee")
                log.info(f"DEBUG: fitting_method='{sampler_name}'")
                
                if sampler_name == "nested":
                    log.info("Starting model selection using Nested Sampling (dynesty)...")
                    best_key, ns_results = fit_models_evidence(
                        model=self.dataset.model,
                        init=init_guess,
                        model_keys=model_keys,
                        priors=None, # Will use defaults in nested module
                        nlive=n_steps // 4, # Heuristic mapping steps -> nlive
                        alpha_prior=(alpha_mu, alpha_sigma) if alpha_fixed is None else None,
                        alpha_fixed=alpha_fixed,
                        likelihood_kind=likelihood_kind,
                        student_nu=studentt_nu,
                    )
                    
                    # Convert NS result to pipeline format
                    from dynesty.utils import resample_equal
                    best_res = ns_results[best_key]
                    flat_chain = resample_equal(best_res.samples, best_res.weights)
                    
                    results = {
                        "best_key": best_key,
                        "best_params": best_res.get_best_params(),
                        "flat_chain": flat_chain,
                        "model_instance": self.dataset.model,
                        "param_names": best_res.param_names,
                        "goodness_of_fit": {"log_evidence": best_res.log_evidence, "log_evidence_err": best_res.log_evidence_err},
                        "dm_init": self.dm_init,
                        "loop_stats": {"ncall": best_res.ncall},
                        "all_results": ns_results
                    }
                    
                    # For nested, we don't have a 'sampler' object in the emcee sense
                    # So we construct a dummy sampler or skip steps that require it
                    sampler = None 
                    
                else: 
                    log.info("Starting model selection scan (BIC)...")
                    best_key, all_res = fit_models_bic(
                        model=self.dataset.model,
                        init=init_guess,
                        n_steps=n_steps // 2,
                        pool=pool,
                        model_keys=model_keys,
                        sample_log_params=sample_log_params,
                        alpha_prior=(
                            (alpha_mu, alpha_sigma)
                            if alpha_fixed is None
                            else (alpha_fixed, None)
                        ),
                        likelihood_kind=likelihood_kind,
                        student_nu=studentt_nu,
                        walker_width_frac=self.pipeline_kwargs.get(
                            "walker_width_frac", 0.01
                        ),
                    )
                    sampler = all_res[best_key][0]
                    # Pack results for emcee (legacy) path
                    chain = sampler.get_chain(flat=True)
                    log.info(f"Emcee chain shape: {chain.shape}")
                    param_names = FRBFitter._ORDER[best_key]
                    best_params_vec = np.median(chain, axis=0) # crude estimate
                    
                    results = {
                        "best_key": best_key,
                        "best_params": FRBParams.from_sequence(best_params_vec, best_key),
                         "param_names": param_names,
                         "goodness_of_fit": {}, # Populated later
                         "dm_init": self.dm_init,
                         "loop_stats": {},
                    }


            elif ncomp == 1:
                # Direct single model fit code... (keeping existing structure but wrapping in else)
                best_key = "M3"
                if self.pipeline_kwargs.get("sampler") == "nested":
                     log.info("Fitting model M3 directly with Nested Sampling...")
                     # Implement direct nested fit logic here if needed, or fall through
                     pass
                     
                log.info(f"Fitting model {best_key} directly...")
                # right before sampling
                priors, use_logw = build_priors(
                    init_guess,
                    scale=6.0,
                    abs_max={"tau_1ghz": 5e4, "zeta": 5e4},
                    log_weight_pos=True,
                )  # Jeffreys weighting in prior weight
                # give generous bounds to the two broadening params
                priors["tau_1ghz"] = (1e-6, 5e4)  # ms
                priors["zeta"] = (1e-6, 5e4)  # ms
                # alpha prior bounds
                if alpha_fixed is not None:
                    priors["alpha"] = (float(alpha_fixed), float(alpha_fixed))
                    alpha_prior = (float(alpha_fixed), None)
                else:
                    lo_a = max(0.1, float(alpha_mu) - 6.0 * float(alpha_sigma))
                    hi_a = float(alpha_mu) + 6.0 * float(alpha_sigma)
                    priors["alpha"] = (lo_a, hi_a)
                    alpha_prior = (float(alpha_mu), float(alpha_sigma))
                # delta_dm bounds (top-hat prior)
                dm_w = float(delta_dm_sigma)
                priors["delta_dm"] = (-3.0 * dm_w, 3.0 * dm_w)

                fitter = FRBFitter(
                    self.dataset.model,
                    priors,
                    n_steps=n_steps,
                    pool=pool,
                    log_weight_pos=use_logw,
                    sample_log_params=sample_log_params,
                    alpha_prior=alpha_prior,
                    likelihood_kind=likelihood_kind,
                    student_nu=studentt_nu,
                    walker_width_frac=self.pipeline_kwargs.get(
                        "walker_width_frac", 0.01
                    ),
                )
                
                # UPDATED: Unpack tuple return (sampler, diagnostics)
                sampler, mcmc_diag = fitter.sample(init_guess, model_key=best_key)
            else:
                # Multi-component with shared PBF
                K = ncomp
                log.info(f"Fitting multi-component model with K={K} (shared PBF)...")
                # Build initial multi guess
                if self.seed_multi is not None:
                    init_multi = self.seed_multi
                else:
                    init_multi = self._get_initial_guess_multi(
                        self.dataset.model, K, base=init_guess
                    )
                # Build priors for shared + component params
                priors = {}
                # shared from build_priors around base guess
                shared_priors, use_logw = build_priors(
                    init_guess,
                    scale=6.0,
                    abs_max={"tau_1ghz": 5e4, "zeta": 5e4},
                    log_weight_pos=True,
                )
                priors.update(
                    {
                        k: v
                        for k, v in shared_priors.items()
                        if k in ("gamma", "tau_1ghz")
                    }
                )
                # alpha, delta_dm
                if alpha_fixed is not None:
                    priors["alpha"] = (float(alpha_fixed), float(alpha_fixed))
                    alpha_prior = (float(alpha_fixed), None)
                else:
                    lo_a = max(0.1, float(alpha_mu) - 6.0 * float(alpha_sigma))
                    hi_a = float(alpha_mu) + 6.0 * float(alpha_sigma)
                    priors["alpha"] = (lo_a, hi_a)
                    alpha_prior = (float(alpha_mu), float(alpha_sigma))
                dm_w = float(delta_dm_sigma)
                priors["delta_dm"] = (-3.0 * dm_w, 3.0 * dm_w)

                # per-component bounds
                tmin, tmax = float(self.dataset.time.min()), float(
                    self.dataset.time.max()
                )
                for i in range(1, K + 1):
                    priors[f"c0_{i}"] = (1e-6, 1e9)
                    priors[f"t0_{i}"] = (tmin, tmax)
                    priors[f"zeta_{i}"] = (1e-6, 5e4)

                fitter = FRBFitter(
                    self.dataset.model,
                    priors,
                    n_steps=n_steps,
                    pool=pool,
                    log_weight_pos=use_logw,
                    sample_log_params=sample_log_params,
                    alpha_prior=alpha_prior,
                    likelihood_kind=likelihood_kind,
                    student_nu=studentt_nu,
                    walker_width_frac=self.pipeline_kwargs.get(
                        "walker_width_frac", 0.01
                    ),
                )
                names = fitter.build_multicomp_order(K)
                
                # UPDATED: Unpack tuple return (sampler, diagnostics)
                sampler, mcmc_diag = fitter.sample(init_multi, model_key="M3_multi")
                best_key = "M3_multi"

            if sampler is not None:
                log.info("Processing MCMC chains...")
                burn, thin, convergence_info = auto_burn_thin(sampler)
                flat_chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
                if flat_chain.shape[0] == 0:
                    raise RuntimeError(
                        "MCMC chain is empty after burn-in and thinning. Check sampler settings or increase n_steps."
                    )

                if best_key == "M3_multi":
                    # keep theta_best and names for downstream
                    idx_best = int(
                        np.argmax(sampler.get_log_prob(discard=burn, thin=thin, flat=True))
                    )
                    theta_best = flat_chain[idx_best]
                    
                    param_names = list(fitter.custom_order["M3_multi"])  # type: ignore[attr-defined]
                    results = {
                        "best_key": best_key,
                        "sampler": sampler,
                        "flat_chain": flat_chain,
                        "param_names": param_names,
                        "dm_init": self.dm_init,
                        "model_instance": self.dataset.model,
                        "chain_stats": {
                            "burn_in": burn,
                            "thin": thin,
                            "convergence": convergence_info,
                        },
                        "is_multi": True,
                        "K": K,
                        "theta_best": theta_best,
                        "mcmc_diagnostics": mcmc_diag,  # Add MCMC diagnostics
                    }
                else:
                    best_params = FRBParams.from_sequence(
                        flat_chain[
                            np.argmax(
                                sampler.get_log_prob(discard=burn, thin=thin, flat=True)
                            )
                        ],
                        best_key,
                    )
                    
                    param_names = (
                        FRBFitter._ORDER[best_key]
                        if best_key in FRBFitter._ORDER
                        else [] # Should not happen for standard BIC scan
                    )
                    
                    # Calculate goodness of fit
                    # Fix: pass correct arguments matching definition
                    gof = goodness_of_fit(
                        self.dataset.data,
                        self.dataset.model(best_params, best_key),
                        self.dataset.model.noise_std,
                        len(param_names)
                    )
                    loop_stats = {
                        "burn_in": burn,
                        "thin": thin,
                        "convergence": convergence_info,
                    }
                    
                    results = {
                        "best_key": best_key,
                        "best_params": best_params,
                        "param_names": param_names,
                        "goodness_of_fit": gof,
                        "dm_init": self.dm_init,
                        "loop_stats": loop_stats,
                        "flat_chain": flat_chain,
                        "sampler": sampler,
                        "model_instance": self.dataset.model,
                        "is_multi": False,
                        "mcmc_diagnostics": mcmc_diag,  # Add MCMC diagnostics
                    }

            # Diagnostics and plotting should happen after results is definitely populated
            if results is None:
                raise RuntimeError("Results dictionary was not populated by any fitting path.")

            if diagnostics:
                # Skip diagnostics if chain is badly non-converged (R̂ > 5)
                # Only applicable if sampler is not None (i.e., emcee)
                if sampler is not None:
                    max_rhat = results["chain_stats"].get("convergence", {}).get("max_rhat", 1.0)
                    if max_rhat > 5.0:
                        log.warning(
                            f"Skipping diagnostics: chain not converged (R̂ = {max_rhat:.2f} > 5)"
                        )
                        results["diagnostics"] = {
                            "skipped": True,
                            "reason": f"R̂ = {max_rhat:.2f} too high",
                        }
                    else:
                        try:
                            diag_runner = BurstDiagnostics(self.dataset, results)
                            results["diagnostics"] = diag_runner.run_all(
                                sb_steps=n_steps // 4, pool=pool
                            )
                        except Exception as e:
                            log.warning(f"Diagnostics failed: {e}")
                            results["diagnostics"] = {"skipped": True, "reason": str(e)}
                else: # Nested sampling path, no Rhat
                    try:
                        diag_runner = BurstDiagnostics(self.dataset, results)
                        results["diagnostics"] = diag_runner.run_all(
                            sb_steps=n_steps // 4, pool=pool
                        )
                    except Exception as e:
                        log.warning(f"Diagnostics failed: {e}")
                        results["diagnostics"] = {"skipped": True, "reason": str(e)}

            if best_key == "M3_multi":
                model_dyn = self._build_multi_model(results)
                results["goodness_of_fit"] = goodness_of_fit(
                    self.dataset.data,
                    model_dyn,
                    self.dataset.model.noise_std,
                    len(results["param_names"]),
                )
            else:
                results["goodness_of_fit"] = goodness_of_fit(
                    self.dataset.data,
                    self.dataset.model(results["best_params"], best_key),
                    self.dataset.model.noise_std,
                    len(results["param_names"]),
                )
            log.info(
                f"Best model: {best_key} | χ²/dof = {results['goodness_of_fit']['chi2_reduced']:.2f}"
            )

            if save:
                import json
                
                # Helper to convert numpy types
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super().default(obj)

                # Prepare safe dict
                best_params = results.get("best_params")
                if dataclasses.is_dataclass(best_params):
                    best_params = dataclasses.asdict(best_params)
                elif hasattr(best_params, "__dict__"):
                    best_params = best_params.__dict__
                    
                safe_results = {
                    "best_model": results.get("best_key"),
                    "best_params": best_params,
                    "param_names": results.get("param_names"),
                    "goodness_of_fit": results.get("goodness_of_fit"),
                    "dm_init": results.get("dm_init"),
                    "convergence": results.get("loop_stats")
                }
                
                json_path = os.path.join(self.outpath, f"{self.name}_fit_results.json")
                with open(json_path, "w") as f:
                    json.dump(safe_results, f, indent=4, cls=NumpyEncoder)
                log.info(f"Saved fit results to {json_path}")

            if plot:
                p_path_sixt = os.path.join(self.outpath, f"{self.name}_diagnostics.pdf")
                create_sixteen_panel_plot(self.dataset, results, save=save, show=False)
                p_path_four = os.path.join(self.outpath, f"{self.name}_fullmodel.pdf")
                create_four_panel_plot(self.dataset, results, save=save, show=False)

            return results



    def _get_initial_guess(self, model: "FRBModel") -> "FRBParams":
        """Generate data-driven initial guess for MCMC.
        
        Uses the burstfit_init module to extract parameter estimates
        directly from the data instead of hardcoded values.
        """
        log.info("Finding data-driven initial guess for MCMC...")
        
        # Try data-driven estimation first
        try:
            from .burstfit_init import data_driven_initial_guess
            
            result = data_driven_initial_guess(
                data=model.data,
                freq=model.freq,
                time=model.time,
                dm=self.dm_init,
                verbose=True,
            )
            
            init_guess = result.params
            log.info(f"Data-driven initial guess:")
            log.info(f"  c0      = {init_guess.c0:.2f}")
            log.info(f"  t0      = {init_guess.t0:.3f} ms")
            log.info(f"  gamma   = {init_guess.gamma:.2f}")
            log.info(f"  zeta    = {init_guess.zeta:.3f} ms")
            log.info(f"  tau_1ghz= {init_guess.tau_1ghz:.3f} ms")
            log.info(f"  alpha   = {init_guess.alpha:.2f}")
            
            # Store diagnostics for later inspection
            self._init_guess_diagnostics = result.diagnostics
            
            return init_guess
            
        except Exception as e:
            log.warning(f"Data-driven guess failed: {e}. Falling back to optimization.")
        
        # Fallback: quick optimization-based guess
        f_ds = 1
        t_ds = 1

        # Build down-sampled arrays
        data_ds = model.data[::f_ds, ::t_ds]
        time_ds = model.time[::t_ds]
        freq_ds = model.freq[::f_ds]

        model_ds = FRBModel(
            data=data_ds,
            time=time_ds,
            freq=freq_ds,
            dm_init=self.dm_init,
            df_MHz=model.df_MHz,
        )

        prof = np.nansum(model_ds.data, axis=0)
        if np.all(prof == 0):
            return FRBParams(c0=0, t0=model_ds.time.mean(), gamma=0, zeta=0, tau_1ghz=0)
        
        # Data-derived rough guess (better than pure hardcodes)
        t0_idx = np.argmax(prof)
        t0 = model_ds.time[t0_idx]
        c0 = np.sum(prof)
        
        # Estimate spectral index from data
        spectrum = np.nansum(model_ds.data, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            log_freq = np.log(freq_ds)
            log_flux = np.log(np.maximum(spectrum, 1e-10))
            mask = np.isfinite(log_flux) & np.isfinite(log_freq)
        if mask.sum() > 3:
            try:
                gamma = np.polyfit(log_freq[mask], log_flux[mask], 1)[0]
                gamma = np.clip(gamma, -5, 2)
            except Exception:
                gamma = -1.6
        else:
            gamma = -1.6
        
        # Estimate width from profile variance
        weights = np.maximum(prof - np.percentile(prof, 10), 0)
        weights /= np.sum(weights) + 1e-30
        t_var = np.sum((model_ds.time - t0) ** 2 * weights)
        width = 2.355 * np.sqrt(max(t_var, 1e-6))
        
        # Initial zeta and tau: split observed width
        zeta = max(0.1, width * 0.4)
        tau_1ghz = max(0.1, width * 0.4)
        
        rough_guess = FRBParams(
            c0=c0,
            t0=t0,
            gamma=gamma,
            zeta=zeta,
            tau_1ghz=tau_1ghz,
            alpha=4.0,  # Thin screen default
        )
        
        # Refine with L-BFGS-B
        priors, use_logw = build_priors(
            rough_guess,
            scale=1.5,
            abs_max={"tau_1ghz": 5e4, "zeta": 5e4},
            log_weight_pos=True,
        )
        model_key = "M3"
        x0 = rough_guess.to_sequence(model_key)
        bounds = [priors[n] for n in FRBFitter._ORDER[model_key]]

        def nll(theta):
            p = FRBParams.from_sequence(theta, model_key)
            ll = model_ds.log_likelihood(p, model_key)
            return -ll if np.isfinite(ll) else np.inf

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-7},
        )
        if not res.success:
            warnings.warn("Initial guess optimization failed. Using rough guess.")
            return rough_guess
        log.info("Refined initial guess found via optimization.")
        return FRBParams.from_sequence(res.x, model_key)

    def _get_initial_guess_multi(
        self, model: "FRBModel", K: int, base: "FRBParams"
    ) -> dict[str, float]:
        # Smooth profile and find K peaks
        prof = np.nansum(model.data, axis=0)
        if model.dt > 0:
            sigma_samps = max(1, int((0.1 / 2.355) / model.dt))
            if sigma_samps > 1:
                prof = sp.ndimage.gaussian_filter1d(prof, sigma_samps)
        idxs = np.argpartition(prof, -K)[-K:]
        idxs = np.sort(idxs)
        # initial guesses
        total = np.sum(prof)
        init: dict[str, float] = {
            "gamma": base.gamma,
            "tau_1ghz": max(base.tau_1ghz, 1e-3),
            "alpha": getattr(base, "alpha", 4.4),
            "delta_dm": 0.0,
        }
        for j, ix in enumerate(idxs, start=1):
            init[f"t0_{j}"] = model.time[ix]
            init[f"c0_{j}"] = max(total / K, 1e-3)
            init[f"zeta_{j}"] = max(getattr(base, "zeta", 0.05), 1e-3)
        return init

    def _build_multi_model(self, results: Dict[str, Any]):
        names = results["param_names"]
        theta = results["theta_best"]
        K = int(results["K"])
        model = results["model_instance"]

        # helper
        def get(name):
            return theta[names.index(name)] if name in names else None

        gamma = get("gamma")
        tau1 = get("tau_1ghz")
        alpha = get("alpha")
        delta_dm = get("delta_dm")
        model_sum = np.zeros_like(model.data)
        for i in range(1, K + 1):
            c0 = get(f"c0_{i}")
            t0 = get(f"t0_{i}")
            zeta = get(f"zeta_{i}")
            p = FRBParams(
                c0=c0,
                t0=t0,
                gamma=gamma,
                zeta=zeta,
                tau_1ghz=tau1,
                alpha=alpha,
                delta_dm=delta_dm,
            )
            model_sum = model_sum + model(p, "M3")
        return model_sum


def auto_burn_thin(sampler, safety_factor_burn=3.0, safety_factor_thin=0.5):
    """Automatically determine burn-in and thinning based on autocorrelation time.

    Also computes Gelman-Rubin R̂ for convergence diagnostics.

    Returns
    -------
    tuple
        (burn, thin, convergence_info) where convergence_info is a dict with R̂ values.
    """
    burn = sampler.iteration // 4  # default fallback
    thin = 1
    convergence_info = {}

    try:
        tau = sampler.get_autocorr_time(tol=0.01)
        burn = int(safety_factor_burn * np.nanmax(tau))
        thin = max(1, int(safety_factor_thin * np.nanmin(tau)))
        burn = min(burn, sampler.iteration // 2)
        log.info(f"Auto-determined burn-in: {burn}, thinning: {thin}")
        convergence_info["autocorr_time"] = tau.tolist()
    except Exception as e:
        warnings.warn(f"Could not estimate autocorr time: {e}. Using defaults.")

    # Compute Gelman-Rubin R̂
    try:
        rhat_results = gelman_rubin(sampler, discard=burn)
        convergence_info.update(rhat_results)
        if rhat_results["converged"]:
            log.info(f"Gelman-Rubin R̂ max = {rhat_results['max_rhat']:.4f} (CONVERGED)")
        else:
            log.warning(
                f"Gelman-Rubin R̂ max = {rhat_results['max_rhat']:.4f} (NOT CONVERGED - consider more steps)"
            )
    except Exception as e:
        warnings.warn(f"Could not compute Gelman-Rubin: {e}")
        convergence_info["gelman_rubin_error"] = str(e)

    return burn, thin, convergence_info


###############################################################################
# 4. CLI WRAPPER
###############################################################################
def _main():
    p = argparse.ArgumentParser(description="Run BurstFit pipeline on a .npy file.")
    # Add all possible arguments here
    p.add_argument("inpath", type=Path, help="Input .npy file")
    p.add_argument("--frb", type=str, help="Event name")
    p.add_argument("--outpath", type=Path, help="Output filepath")
    p.add_argument("--dm_init", type=float, default=0.0)
    p.add_argument("--telescope", default="CHIME")
    p.add_argument("--telcfg", default="telescopes.yaml")
    p.add_argument("--sampcfg", default="sampler.yaml")
    p.add_argument("--nproc", type=int, default=None)
    p.add_argument("--yes", action="store_true", help="Bypass pool confirmation")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--f_factor", type=int, default=1)
    p.add_argument("--t_factor", type=int, default=1)
    p.add_argument(
        "--outer-trim",
        dest="outer_trim",
        type=float,
        help="Fraction to trim from each time edge (0-0.49)",
    )
    p.add_argument(
        "--flip-freq",
        dest="flip_freq",
        action="store_true",
        help="Flip frequency axis (high at top)",
    )
    p.add_argument(
        "--no-flip-freq",
        dest="flip_freq",
        action="store_false",
        help="Do not flip frequency axis",
    )
    # New modeling controls
    p.add_argument(
        "--alpha-fixed",
        type=float,
        default=None,
        help="Fix alpha frequency scaling exponent",
    )
    p.add_argument(
        "--alpha-mu", type=float, default=4.4, help="Gaussian prior mean for alpha"
    )
    p.add_argument(
        "--alpha-sigma", type=float, default=0.6, help="Gaussian prior sigma for alpha"
    )
    p.add_argument(
        "--delta-dm-sigma",
        type=float,
        default=0.1,
        help="Top-hat prior sigma for delta DM (pc cm^-3)",
    )
    p.add_argument(
        "--likelihood", type=str, choices=["gaussian", "studentt"], default="gaussian"
    )
    p.add_argument(
        "--studentt-nu", type=float, default=5.0, help="Student-t degrees of freedom"
    )
    p.add_argument(
        "--no-logspace",
        dest="sample_log_params",
        action="store_false",
        help="Disable log-space sampling for positive params",
    )
    # Seeding / walkers
    p.add_argument(
        "--init-guess",
        type=Path,
        default=None,
        help="Path to JSON seed for initial guess (single or multi)",
    )
    p.add_argument(
        "--walker-width-frac",
        type=float,
        default=0.01,
        help="Initial walker cloud width as fraction of prior span",
    )
    # Multi-component controls
    p.add_argument(
        "--ncomp",
        type=int,
        default=1,
        help="Number of Gaussian components (shared PBF)",
    )
    p.add_argument(
        "--auto-components",
        action="store_true",
        help="Greedy BIC-based component selection (placeholder)",
    )
    # Earmarks / placeholders
    p.add_argument(
        "--anisotropy-enabled",
        action="store_true",
        help="Earmark: enable anisotropy option (not implemented)",
    )
    p.add_argument(
        "--anisotropy-axial-ratio",
        type=float,
        default=1.0,
        help="Earmark: anisotropy axial ratio (not implemented)",
    )
    p.add_argument(
        "--baseline-order",
        type=int,
        default=0,
        help="Earmark: polynomial baseline order to marginalize (not implemented)",
    )
    p.add_argument(
        "--correlated-resid",
        action="store_true",
        help="Earmark: AR(1)/GP residual model (not implemented)",
    )
    p.add_argument(
        "--fitting-method",
        dest="fitting_method",
        type=str,
        choices=["emcee", "nested"],
        default="emcee",
        help="Sampler choice (emcee or nested)",
    )
    # Add flags for boolean pipeline controls
    p.add_argument("--no-scan", dest="model_scan", action="store_false")
    p.add_argument("--no-diag", dest="diagnostics", action="store_false")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.set_defaults(model_scan=True, diagnostics=True, plot=True, flip_freq=True)
    args = p.parse_args()

    # --- FIX: Pass all arguments as a dict to the pipeline constructor ---
    # The new __init__ will sort them out automatically.
    pipeline_kwargs = vars(args)

    # Extract required args and provide sensible defaults
    inpath = pipeline_kwargs.pop("inpath")
    outpath = pipeline_kwargs.pop("outpath") or inpath.parent
    name = pipeline_kwargs.pop("frb") or inpath.stem
    dm_init = pipeline_kwargs.pop("dm_init")

    # Harmonize config key names for dataset constructor
    telcfg_cli = pipeline_kwargs.pop("telcfg", None)
    if telcfg_cli is not None:
        pipeline_kwargs["telcfg_path"] = telcfg_cli
    sampcfg_cli = pipeline_kwargs.pop("sampcfg", None)
    if sampcfg_cli is not None:
        pipeline_kwargs["sampcfg_path"] = sampcfg_cli

    pipe = BurstPipeline(
        name=name,
        inpath=inpath,  # positional arg extracted above
        outpath=outpath,
        dm_init=dm_init,
        **pipeline_kwargs,
    )

    pipe.run_full(
        model_scan=args.model_scan, diagnostics=args.diagnostics, plot=args.plot
    )


if __name__ == "__main__":
    _main()
