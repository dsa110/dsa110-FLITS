#!/usr/bin/env python3
"""
Generate diagnostic plots from scattering fit results.

This script replicates the exact preprocessing pipeline used during fitting
to create accurate diagnostic visualizations showing data, model, and residuals.

Usage:
    python -m scattering.scat_analysis.visualization <results.json> <data.npy> <telescope> [options]

Example:
    python -m scattering.scat_analysis.visualization \\
        freya_chime_I_912_4067_32000b_cntr_bpc_fit_results.json \\
        data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy \\
        chime \\
        --t-factor 4 \\
        --f-factor 32 \\
        --output freya_diagnostic.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


from scattering.scat_analysis.burstfit import FRBModel, FRBParams, downsample
from flits.utils.reporting import print_fit_summary, get_fit_summary_lines


def load_telescope_config(telescope_name: str, config_path: Path = None) -> dict:
    """Load telescope configuration from YAML."""
    if config_path is None:
        # Default location
        config_path = Path(__file__).parent.parent / "configs" / "telescopes.yaml"
    
    with open(config_path) as f:
        configs = yaml.safe_load(f)
    
    if telescope_name not in configs:
        raise ValueError(f"Telescope '{telescope_name}' not found in {config_path}")
    
    return configs[telescope_name]


def preprocess_data(
    raw_data: np.ndarray,
    config: dict,
    t_factor: int = 4,
    f_factor: int = 32,
    outer_trim: float = 0.45,
    smooth_ms: float = 0.1,
    center_burst: bool = True
):
    """
    Preprocess data exactly as the pipeline does.
    
    Parameters
    ----------
    raw_data : ndarray
        Raw dynamic spectrum (freq × time)
    config : dict
        Telescope configuration
    t_factor : int
        Time downsampling factor
    f_factor : int
        Frequency downsampling factor
    outer_trim : float
        Fraction to trim from each end (0.45 = keep central 10%)
    smooth_ms : float
        Smoothing width in ms for burst detection
    center_burst : bool
        Whether to center the burst in the array
    
    Returns
    -------
    data : ndarray
        Preprocessed data
    freq : ndarray
        Frequency axis in GHz
    time : ndarray
        Time axis in ms
    dt_ms : float
        Time resolution in ms
    df_MHz : float
        Frequency resolution in MHz
    """
    # Bandpass correction
    raw_data = np.nan_to_num(raw_data.astype(np.float64))
    n_t_raw = raw_data.shape[1]
    q = n_t_raw // 4
    off_pulse_idx = np.r_[0:q, -q:0]
    
    mu = np.nanmean(raw_data[:, off_pulse_idx], axis=1, keepdims=True)
    sig = np.nanstd(raw_data[:, off_pulse_idx], axis=1, keepdims=True)
    sig[sig < 1e-9] = np.nan
    raw_corr = np.nan_to_num((raw_data - mu) / sig, nan=0.0)
    
    # Downsample
    data = downsample(raw_corr, f_factor, t_factor)
    
    # Apply outer trim
    n_trim = int(outer_trim * data.shape[1])
    if n_trim > 0:
        data = data[:, n_trim:-n_trim]
    
    # Build axes
    n_ch, n_t = data.shape
    dt_ms = config["dt_ms_raw"] * t_factor
    df_MHz = config["df_MHz_raw"] * f_factor
    
    # Handle frequency ordering
    # Force output to be ascending frequency (Low -> High) for standardized plotting
    freq_desc = config.get("freq_descending", False)
    
    if freq_desc:
        # Data is High -> Low. Flip to make it Low -> High.
        data = np.flip(data, axis=0)
        # Verify: original data[0] was High Freq. Now data[-1] is High Freq.
        # So Index 0 is Low Freq.
    
    # Always generate ascending frequency axis
    freq = np.linspace(config["f_min_GHz"], config["f_max_GHz"], n_ch)
    
    time = np.arange(n_t) * dt_ms
    
    # Center burst if requested
    if center_burst:
        prof = np.sum(data, axis=0)
        sigma_samps = (smooth_ms / 2.355) / dt_ms
        prof_smooth = gaussian_filter1d(prof, sigma=sigma_samps)
        burst_idx = np.argmax(prof_smooth)
        shift = n_t // 2 - burst_idx
        data = np.roll(data, shift, axis=1)
    
    return data, freq, time, dt_ms, df_MHz


def plot_scattering_diagnostic(
    data: np.ndarray,
    model: np.ndarray,
    freq: np.ndarray,
    time: np.ndarray,
    params: FRBParams,
    results: dict,
    output_path: Path,
    burst_name: str = "FRB"
):
    """
    Create 4-panel diagnostic plot matching create_four_panel_plot layout.
    
    Each of the 4 panels shows:
    - Top row: Time series profile
    - Bottom-left: Dynamic spectrum (2D waterfall)
    - Bottom-right: Frequency spectrum
    
    Parameters
    ----------
    data : ndarray
        Preprocessed data (freq × time)
    model : ndarray
        Model dynamic spectrum
    freq : ndarray
        Frequency axis in GHz
    time : ndarray
        Time axis in ms
    params : FRBParams
        Best-fit parameters
    results : dict
        Full results dictionary (must contain 'goodness_of_fit', 'best_key', 'param_names', 'flat_chain')
    output_path : Path
        Output file path
    burst_name : str
        Name of the burst for title
    """
    # Generate synthetic data with noise for comparison
    # Extract noise from results if available, otherwise estimate from off-pulse
    q = data.shape[1] // 4
    data_off_pulse = data[:, np.r_[0:q, -q:0]]
    noise_std = np.nanstd(data_off_pulse, axis=1)
    
    synthetic_noise = np.random.normal(0.0, noise_std[:, None], size=data.shape)
    synthetic_data = model + synthetic_noise
    residual = data - synthetic_data
    
    # Normalize all panels consistently
    mean_off = np.nanmean(data_off_pulse)
    std_off = np.nanstd(data_off_pulse)
    if std_off < 1e-9: 
        std_off = 1.0
    
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
    model_norm = _apply_norm(model, subtract_mean=True)
    synthetic_norm = _apply_norm(synthetic_data, subtract_mean=True)
    residual_norm = _apply_norm(residual, subtract_mean=False)
    
    # Calculate global Y-limits for Time Series
    all_ts = [np.nansum(p, axis=0) for p in [data_norm, model_norm, synthetic_norm, residual_norm]]
    ts_min = min(np.min(t) for t in all_ts if t.size > 0)
    ts_max = max(np.max(t) for t in all_ts if t.size > 0)
    y_range = ts_max - ts_min
    ts_ylim = (ts_min - 0.05 * y_range, ts_max + 0.05 * y_range)
    
    # Calculate global X-limits for Spectrum
    all_sp = [np.nansum(p, axis=1) for p in [data_norm, model_norm, synthetic_norm, residual_norm]]
    sp_min = min(np.min(s) for s in all_sp if s.size > 0)
    sp_max = max(np.max(s) for s in all_sp if s.size > 0)
    x_range = sp_max - sp_min
    sp_xlim = (sp_min - 0.05 * x_range, sp_max + 0.05 * x_range)
    
    # Create figure: 2 rows, 8 columns
    fig, axes = plt.subplots(
        nrows=2,
        ncols=8,
        gridspec_kw={"height_ratios": [1, 2.5], "width_ratios": [2, 0.5] * 4},
        figsize=(24, 8),
    )
    
    time_centered = time - (time[0] + (time[-1] - time[0]) / 2)
    extent = [time_centered[0], time_centered[-1], freq[0], freq[-1]]
    
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
        ax_ts.set_ylim(ts_ylim)
        
        cmap = "coolwarm" if title == "Residual" else "plasma"
        if title == "Residual":
            vmax = np.nanmax(np.abs(panel_ds))
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
        ax_sp.set_xlim(sp_xlim)
        
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
    
    plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.83, bottom=0.15, left=0.05, right=0.98)
    
    # --- Create Header ---
    best_key = results.get('best_key', results.get('best_model', 'Unknown'))
    param_names = results.get('param_names', [])
    flat_chain = results.get('flat_chain', np.array([]))
    
    # 1. Metadata
    fname = output_path.name
    tns_name = "FRB 20190425A" if "casey" in burst_name.lower() else f"FRB ({burst_name})"
    observatory = "CHIME/FRB" if "chime" in fname.lower() else "DSA-110"
    
    meta_text = (
        f"{tns_name} / {burst_name.upper()}\n"
        f"Observatory: {observatory}"
    )
    
    # 2. Model Selection
    model_text = "Model Selection:\n"
    if "all_results" in results:
        res_all = results["all_results"]
        keys = list(res_all.keys())
        keys.sort(reverse=True)
        best_z = max(float(res_all[k].log_evidence) for k in keys) if keys else 0
        
        for k in keys:
            z = float(res_all[k].log_evidence)
            dz = z - best_z
            mark = r"$\mathbf{\ast}$" if k == best_key else " "
            model_text += f"{mark} {k}: $\\ln{{Z}}={z:.0f}$ ($\\Delta={dz:.0f}$)\n"
    else:
        model_text += f"{best_key} (Selected)\n(Comparison N/A)"
    
    # 3. Goodness of Fit
    gof = results.get("goodness_of_fit", {})
    chi2 = gof.get("chi2_reduced", np.nan)
    r2 = gof.get("r_squared", np.nan)
    quality = gof.get("quality_flag", "UNKNOWN")
    
    gof_body = (
        f"$\\chi^2_{{\\nu}} = {chi2:.2f}$\n"
        f"$R^2   = {r2:.3f}$"
    )
    
    # 4. Parameters
    param_lines = []
    for i, name in enumerate(param_names):
        if flat_chain.size > 0 and flat_chain.ndim == 2 and i < flat_chain.shape[1]:
            vals = flat_chain[:, i]
            if not np.all(np.isnan(vals)):
                val = np.median(vals)
                err = np.std(vals)
            else:
                val, err = getattr(params, name, np.nan), 0.0
        else:
            val, err = getattr(params, name, np.nan), 0.0
        
        if abs(val) < 0.001 and val != 0:
            s_val = f"{val:.1e}"
        else:
            s_val = f"{val:.4g}"
        
        param_lines.append(f"{name} = ${s_val} \\pm {err:.1g}$")
    param_text = "\n".join(param_lines)
    
    # Render Header
    header_y = 0.94
    body_y = 0.91
    fontsize_head = 12
    fontsize_body = 10
    
    fig.text(0.05, header_y, meta_text, fontsize=14, weight='bold', va='top', fontfamily='sans-serif')
    fig.text(0.28, header_y, "Model Selection", fontsize=fontsize_head, weight='bold', va='top')
    fig.text(0.28, body_y, model_text.replace("Model Selection:\n", ""), fontsize=fontsize_body, va='top')
    fig.text(0.52, header_y, "Goodness of Fit", fontsize=fontsize_head, weight='bold', va='top')
    fig.text(0.52, body_y, gof_body, fontsize=fontsize_body, va='top')
    
    q_color = "green" if quality == "PASS" else ("red" if quality == "FAIL" else "orange")
    fig.text(0.52, body_y - 0.04, f"Status: {quality}", fontsize=fontsize_body, weight='bold', color=q_color, va='top')
    
    fig.text(0.75, header_y, "Best Fit Parameters", fontsize=fontsize_head, weight='bold', va='top')
    fig.text(0.75, body_y, param_text, fontsize=fontsize_body, va='top', fontfamily='monospace')
    
    # Save
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots from scattering fit results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to fit results JSON file"
    )
    parser.add_argument(
        "data_npy",
        type=Path,
        help="Path to raw data .npy file"
    )
    parser.add_argument(
        "telescope",
        type=str,
        help="Telescope name (must be in telescopes.yaml)"
    )
    parser.add_argument(
        "--t-factor",
        type=int,
        default=4,
        help="Time downsampling factor (default: 4)"
    )
    parser.add_argument(
        "--f-factor",
        type=int,
        default=32,
        help="Frequency downsampling factor (default: 32)"
    )
    parser.add_argument(
        "--outer-trim",
        type=float,
        default=0.45,
        help="Fraction to trim from each end (default: 0.45)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: auto-generated from input)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to telescopes.yaml (default: auto-detect)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_json}")
    with open(args.results_json) as f:
        results = json.load(f)
    
    best_params = results["best_params"]
    best_params = results["best_params"]
    
    # --- CONSOLIDATED FIT REPORTING ---
    print_fit_summary(results)
    # ----------------------------------
    
    # Load telescope config
    print(f"Loading telescope config for '{args.telescope}'")
    config = load_telescope_config(args.telescope, args.config)
    
    # Load raw data
    print(f"Loading data from {args.data_npy}")
    raw_data = np.load(args.data_npy)
    print(f"Raw data shape: {raw_data.shape}")
    
    # Preprocess data
    print(f"Preprocessing (t_factor={args.t_factor}, f_factor={args.f_factor}, "
          f"outer_trim={args.outer_trim})")
    data, freq, time, dt_ms, df_MHz = preprocess_data(
        raw_data, config, args.t_factor, args.f_factor, args.outer_trim
    )
    print(f"Processed shape: {data.shape}")
    print(f"Time range: {time[0]:.3f} to {time[-1]:.3f} ms")
    print(f"Freq range: {freq[0]:.4f} to {freq[-1]:.4f} GHz")
    
    # Generate model
    print("Generating model...")
    model_obj = FRBModel(time=time, freq=freq, data=data, df_MHz=df_MHz)
    params = FRBParams(**best_params)
    model = model_obj(params, results["best_model"])
    
    # Determine output path
    if args.output is None:
        output_path = args.results_json.parent / args.results_json.name.replace(
            "_fit_results.json", "_diagnostic.png"
        )
    else:
        output_path = args.output
    
    # Extract burst name from filename
    burst_name = args.data_npy.stem.split('_')[0].capitalize()
    
    # Create plot
    print("Creating diagnostic plot...")
    fig = plot_scattering_diagnostic(
        data, model, freq, time, params, results, output_path, burst_name
    )
    
    if args.show:
        plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()
