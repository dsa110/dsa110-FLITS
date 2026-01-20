"""
Diagnostic plotting and validation for the BurstFit pipeline.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp

from ..burstfit import plot_dynamic
from ..burstfit_robust import (
    subband_consistency,
    leave_one_out_influence,
    plot_influence,
    fit_subband_profiles,
    dm_optimization_check,
)
from flits.fitting.diagnostics import analyze_residuals

log = logging.getLogger(__name__)

class BurstDiagnostics:
    """A container for running and storing all post-fit diagnostic checks."""

    def __init__(self, dataset, results: Dict[str, Any]):
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
            output_path=str(
                self.dataset.outpath / f"{self.dataset.name}_residuals_detailed.png"
            ),
        )
        self.diag_results["residual_analysis"] = res_diag

        log.info("Diagnostics complete.")
        return self.diag_results


def create_four_panel_plot(
    dataset,
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
    if std_off < 1e-9:
        std_off = 1.0  # Prevent div/0

    # Pre-normalize data to get peak S/N
    data_snr = (data - mean_off) / std_off
    peak_snr = np.nanmax(data_snr)
    if peak_snr <= 0:
        peak_snr = 1.0

    def _apply_norm(arr, subtract_mean=True):
        if subtract_mean:
            return (arr - mean_off) / std_off / peak_snr
        else:
            return arr / std_off / peak_snr

    data_norm = _apply_norm(data, subtract_mean=True)
    model_norm = _apply_norm(clean_model, subtract_mean=True)
    synthetic_norm = _apply_norm(synthetic_data, subtract_mean=True)
    residual_norm = _apply_norm(
        residual, subtract_mean=False
    )  # Residuals already centered

    # Calculate global Y-limits for Time Series (Top Panels)
    # We want valid limits covering min/max of everything
    all_ts = [
        np.nansum(p, axis=0)
        for p in [data_norm, model_norm, synthetic_norm, residual_norm]
    ]
    ts_min = min(np.min(t) for t in all_ts if t.size > 0)
    ts_max = max(np.max(t) for t in all_ts if t.size > 0)
    y_range = ts_max - ts_min
    ts_ylim = (ts_min - 0.05 * y_range, ts_max + 0.05 * y_range)

    # Calculate global X-limits for Spectrum (Side Panels)
    # These plot Flux (x) vs Freq (y), so we need shared X-limits
    all_sp = [
        np.nansum(p, axis=1)
        for p in [data_norm, model_norm, synthetic_norm, residual_norm]
    ]
    sp_min = min(np.min(s) for s in all_sp if s.size > 0)
    sp_max = max(np.max(s) for s in all_sp if s.size > 0)
    x_range = sp_max - sp_min
    sp_xlim = (sp_min - 0.05 * x_range, sp_max + 0.05 * x_range)

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
        ax_ts.set_ylim(ts_ylim)  # Enforce shared Y-limits

        cmap = "coolwarm" if title == "Residual" else "plasma"
        if title == "Residual":
            vmax = np.nanmax(np.abs(panel_ds))  # Symmetric around 0
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
        ax_sp.set_xlim(sp_xlim)  # Enforce shared X-limits

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
    plt.subplots_adjust(
        hspace=0.05, wspace=0.05, top=0.83, bottom=0.15, left=0.05, right=0.98
    )

    # --- Detailed Header Table Implementation (Clean/Scientific) ---
    # Aligned with panels, no boxes, clean text.

    # 1. Metadata
    fname = dataset.inpath.name
    tns_name = "FRB 20190425A" if "casey" in dataset.name.lower() else "FRB (Unknown)"
    person_name = dataset.name.split("_")[0].upper()  # e.g. CASEY
    observatory = "CHIME/FRB" if "chime" in fname.lower() else "DSA-110"

    meta_text = f"{tns_name} / {person_name}\nObservatory: {observatory}"

    # 2. Model Selection
    model_text = "Model Selection:\n"
    if "all_results" in results:
        res_all = results["all_results"]
        keys = list(res_all.keys())
        keys.sort(reverse=True)  # M3, M2, M1

        def get_logz(res):
            if isinstance(res, dict):
                return res.get("log_evidence", 0)
            return getattr(res, "log_evidence", 0)

        best_z = max(float(get_logz(res_all[k])) for k in keys) if keys else 0

        for k in keys:
            z = float(get_logz(res_all[k]))
            dz = z - best_z
            # Clean format: M3: logZ = -562 (d=0)
            mark = r"$\mathbf{\ast}$" if k == best_key else " "
            # Use LaTeX for cleaner look if possible, or just text
            model_text += rf"{mark} {k}: $\ln{{Z}}={z:.0f}$ ($\Delta={dz:.0f}$)\n"
    else:
        model_text += f"{best_key} (Selected)\n(Comparison N/A)"

    # 3. Goodness of Fit
    gof = results.get("goodness_of_fit", {})
    chi2 = gof.get("chi2_reduced", np.nan)
    r2 = gof.get("r_squared", np.nan)
    quality = gof.get("quality_flag", "UNKNOWN")

    # Use color only for the status word
    gof_header = "Goodness of Fit:"
    reduced_chi2_label = r"$\chi^2_r$"
    gof_body = f"{reduced_chi2_label} = {chi2:.2f}\n$R^2 = {r2:.3f}$"

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

        msg = rf"{name} = ${s_val} \pm {err:.1g}$"
        param_lines.append(msg)
    param_text = "\n".join(param_lines)

    # --- Rendering Text Blocks ---
    # Y position for all blocks (moved down slightly to ensure safe margin)
    header_y = 0.94
    body_y = 0.91
    fontsize_head = 12
    fontsize_body = 10

    # Block 1 (Left): Meta (Larger, bold)
    fig.text(
        0.05,
        header_y,
        meta_text,
        fontsize=14,
        weight="bold",
        va="top",
        fontfamily="sans-serif",
    )

    # Block 2: Model
    fig.text(
        0.28,
        header_y,
        "Model Selection",
        fontsize=fontsize_head,
        weight="bold",
        va="top",
    )
    fig.text(
        0.28,
        body_y,
        model_text.replace("Model Selection:\n", ""),
        fontsize=fontsize_body,
        va="top",
    )

    # Block 3: GoF
    fig.text(
        0.52, header_y, gof_header, fontsize=fontsize_head, weight="bold", va="top"
    )
    fig.text(0.52, body_y, gof_body, fontsize=fontsize_body, va="top")
    # Status below params or specifically colored
    q_color = (
        "green" if quality == "PASS" else ("red" if quality == "FAIL" else "orange")
    )
    fig.text(
        0.52,
        body_y - 0.04,
        f"Status: {quality}",
        fontsize=fontsize_body,
        weight="bold",
        color=q_color,
        va="top",
    )

    # Block 4: Params
    fig.text(
        0.75, header_y, param_header, fontsize=fontsize_head, weight="bold", va="top"
    )
    fig.text(
        0.75,
        body_y,
        param_text,
        fontsize=fontsize_body,
        va="top",
        fontfamily="monospace",
    )

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


def create_fit_summary_plot(
    dataset,
    results: Dict[str, Any],
    *,
    save: bool = True,
    show: bool = True,
):
    """Creates an instructive and logically grouped fit summary page."""
    log.info("Generating enhanced fit diagnostics page...")

    best_key, best_p = results["best_key"], results["best_params"]
    sampler, gof = results.get("sampler"), results.get("goodness_of_fit")
    chain_stats, flat_chain = results.get("chain_stats", {}), results["flat_chain"]
    param_names, diag_results = results["param_names"], results.get("diagnostics", {})
    model_instance = results["model_instance"]

    model_dyn = model_instance(best_p, best_key)
    residual = dataset.data - model_dyn

    # Set up GridSpec (5 rows, 4 columns)
    fig = plt.figure(figsize=(18, 22), constrained_layout=True)
    gs = gridspec.GridSpec(5, 4, figure=fig, height_ratios=[0.1, 1, 0.8, 0.8, 0.8])

    # Title row
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_axis_off()
    ax_title.text(
        0.5,
        0.5,
        f"Fit Diagnostics - {dataset.name} ({best_key})",
        fontsize=24,
        ha="center",
        va="center",
        weight="bold",
    )

    # --- SECTION 1: Fit Overview (Top Row) ---
    ax_wf_data = fig.add_subplot(gs[1, 0])
    ax_wf_model = fig.add_subplot(gs[1, 1])
    ax_wf_res = fig.add_subplot(gs[1, 2])
    ax_prof = fig.add_subplot(gs[1, 3])

    vmin, vmax = np.nanpercentile(dataset.data, [1, 99])
    plot_dynamic(ax_wf_data, dataset.data, dataset.time, dataset.freq, vmin=vmin, vmax=vmax, cmap="plasma")
    ax_wf_data.set_title("1. Data", fontweight="bold")
    
    plot_dynamic(ax_wf_model, model_dyn, dataset.time, dataset.freq, vmin=vmin, vmax=vmax, cmap="plasma")
    ax_wf_model.set_title(f"2. Model ({best_key})", fontweight="bold")
    
    res_std = np.nanstd(residual)
    plot_dynamic(ax_wf_res, residual, dataset.time, dataset.freq, vmin=-3 * res_std, vmax=3 * res_std, cmap="coolwarm")
    ax_wf_res.set_title("3. Residuals", fontweight="bold")

    ax_prof.plot(dataset.time, np.nansum(dataset.data, axis=0), "k-", alpha=0.5, label="Data")
    ax_prof.plot(dataset.time, np.nansum(model_dyn, axis=0), "m-", lw=2, label="Model")
    ax_prof.set_title("Time Profile", fontweight="bold")
    ax_prof.legend(fontsize=9)
    ax_prof.set_xlabel("Time [ms]")

    # --- SECTION 2: Residual Analysis (Statistical Quality) ---
    ax_res_hist = fig.add_subplot(gs[2, 0:2])
    ax_res_acf = fig.add_subplot(gs[2, 2:4])

    res_norm = residual / model_instance.noise_std[:, None]
    res_norm_clean = res_norm[np.isfinite(res_norm)]
    ax_res_hist.hist(res_norm_clean, bins=100, density=True, color="gray", alpha=0.7, label="Residuals")
    x_pdf = np.linspace(-4, 4, 100)
    ax_res_hist.plot(x_pdf, sp.stats.norm.pdf(x_pdf), "m--", lw=2, label="N(0,1)")
    ax_res_hist.set_title("Residual Distribution (S/N units)", fontweight="bold")
    ax_res_hist.legend()
    ax_res_hist.set_xlabel("Residual (σ)")

    if gof:
        lags_ms = (np.arange(len(gof["residual_autocorr"])) - len(gof["residual_autocorr"]) // 2) * dataset.dt_ms
        ax_res_acf.plot(lags_ms, gof["residual_autocorr"], "k-", label="Data")
        try:
            n_ppc = 50
            acfs = []
            for _ in range(n_ppc):
                noise = np.random.normal(0.0, model_instance.noise_std[:, None], size=dataset.data.shape)
                resid_ppc = np.nansum(noise, axis=0)
                resid_ppc -= np.nanmean(resid_ppc)
                acf_ppc = np.correlate(resid_ppc, resid_ppc, mode="same")
                cv = acf_ppc[len(acf_ppc) // 2]
                if cv > 0: acf_ppc = acf_ppc / cv
                acfs.append(acf_ppc)
            acfs = np.asarray(acfs)
            ax_res_acf.fill_between(lags_ms, np.percentile(acfs, 5, axis=0), np.percentile(acfs, 95, axis=0), color="m", alpha=0.15, label="Expected (90% CI)")
        except Exception: pass
        ax_res_acf.axhline(0, color='gray', lw=0.5)
        ax_res_acf.set_title("Residual ACF (Temporal Correlations)", fontweight="bold")
        ax_res_acf.set_xlabel("Lag [ms]")
        ax_res_acf.legend(loc="upper right", fontsize=9)

    # --- SECTION 3: Consistency checks ---
    ax_sub_consist = fig.add_subplot(gs[3, 0:2])
    ax_dm_opt = fig.add_subplot(gs[3, 2])
    ax_influence = fig.add_subplot(gs[3, 3])

    if diag_results.get("subband_2d") is not None:
        p_name, s_res, _ = diag_results["subband_2d"]
        if p_name:
            valid = [(i, v, e) for i, (v, e) in enumerate(s_res) if np.isfinite(v) and e > 0]
            if valid:
                idx, vals, errs = zip(*valid)
                edges = np.linspace(0, dataset.freq.size, len(s_res) + 1, dtype=int)
                bc = np.array([dataset.freq[edges[i]:edges[i+1]].mean() for i in range(len(s_res))])[list(idx)]
                ax_sub_consist.errorbar(bc, vals, yerr=errs, fmt="o", c="k", capsize=3, label=f"Sub-band {p_name}")
                global_val = getattr(best_p, p_name)
                ax_sub_consist.axhline(global_val, color="m", ls="--", label="Global Fit")
                ax_sub_consist.set_ylabel(p_name)
                ax_sub_consist.set_title(f"Parameter Consistency: {p_name}", fontweight="bold")
                ax_sub_consist.legend(fontsize=9)
    else:
        ax_sub_consist.text(0.5, 0.5, "Sub-band Diagnostics\nNot Run", ha="center", va="center")
        ax_sub_consist.set_axis_off()

    if diag_results.get("dm_check") is not None:
        dms, snrs = diag_results["dm_check"]
        ax_dm_opt.plot(dms, snrs, "o-k")
        ax_dm_opt.set_title("DM Tuning SNR", fontweight="bold")
        ax_dm_opt.set_xlabel("ΔDM")
    else:
        ax_dm_opt.set_axis_off()

    if diag_results.get("influence") is not None:
        plot_influence(ax_influence, diag_results["influence"], dataset.freq)
        ax_influence.set_title("Channel Influence", fontweight="bold")
    else:
        ax_influence.set_axis_off()

    # --- SECTION 4: Summary & Verdict (Reviewer Guidance) ---
    ax_verdict = fig.add_subplot(gs[4, 0:2])
    ax_params = fig.add_subplot(gs[4, 2:4])
    ax_verdict.set_axis_off()
    ax_params.set_axis_off()

    if gof:
        quality = gof.get("quality_flag", "UNKNOWN")
        color = {"PASS": "green", "MARGINAL": "orange", "FAIL": "red"}.get(quality, "black")
        
        verdict_text = (
            f"V&V VERDICT: {quality}\n"
            f"---------------------------------\n"
            f"χ²/dof:      {gof.get('chi2_reduced', 0.0):.2f}\n"
            f"R²:          {gof.get('r_squared', 0.0):.3f}\n"
            f"Normality p: {gof.get('normality_pvalue', 0.0):.1e}\n"
            f"Bias σ:      {gof.get('bias_nsigma', 0.0):.1f}\n"
            f"DW Stat:     {gof.get('durbin_watson', 0.0):.2f}\n\n"
            f"GUIDANCE:\n"
        )
        if quality == "PASS":
            verdict_text += "• Robust convergence, clean residuals.\n• Model statistically matches data."
        elif quality == "FAIL":
            verdict_text += "• Persistent structure in residuals!\n• Check for unmodelled components/RFI."
        else:
            verdict_text += "• Fit acceptable but check for mild biases\n  or autocorrelation in residuals."

        ax_verdict.text(0.05, 0.95, verdict_text, va="top", fontfamily="monospace", fontsize=12, color=color, fontweight="bold")

    p_lines = [f"{'Param':<10} | {'Value':>10} | {'Unc (1σ)':>10}", "-" * 35]
    for name in param_names:
        val = getattr(best_p, name)
        try:
            idx = param_names.index(name)
            unc = np.std(flat_chain[:, idx])
            p_lines.append(f"{name:<10} | {val:>10.4f} | {unc:>10.4f}")
        except:
            p_lines.append(f"{name:<10} | {val:>10.4f} | {'N/A':>10}")

    ax_params.text(0.05, 0.95, "\n".join(p_lines), va="top", fontfamily="monospace", fontsize=11)
    ax_params.set_title("Best Fit Parameters", fontweight="bold", loc="left")

    if save:
        spath = dataset.outpath / f"{dataset.name}_fit_summary.png"
        fig.savefig(spath, dpi=150, bbox_inches="tight")
        log.info(f"Saved enhanced diagnostics to {spath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
