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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import yaml
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from .burst_metadata import load_tns_name


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


def _draw_diagnostic_header(fig, results, tns_name, burst_name, observatory, best_key):
    """Draw the elegant header at the top of the diagnostic plot."""
    # Design Configuration
    C_BG = "#F4F6F7"
    C_TEXT_PRIMARY = "#333333"
    C_TEXT_SECONDARY = "#777777"
    C_HIGHLIGHT_BLUE = "#0056b3"
    C_STATUS_RED = "#d9534f"
    C_STATUS_GREEN = "#28a745"
    C_DIVIDER = "#E0E0E0"
    
    FONT_SANS = 'DejaVu Sans'
    KW_TITLE = dict(fontname=FONT_SANS, fontsize=9, color=C_TEXT_SECONDARY, weight='bold', ha='left', va='top')
    
    # Add background rectangle
    header_rect = mpatches.Rectangle((0.05, 0.89), 0.93, 0.10, 
                                    transform=fig.transFigure,
                                    facecolor=C_BG, edgecolor='none', zorder=-1)
    fig.add_artist(header_rect)
    
    def add_divider(x_pos):
        line = mpatches.ConnectionPatch(
            xyA=(x_pos, 0.90), xyB=(x_pos, 0.98),
            coordsA='figure fraction', coordsB='figure fraction',
            color=C_DIVIDER, linewidth=1)
        fig.add_artist(line)

    # Panel 1: Event Information
    fig.text(0.06, 0.975, "EVENT", **KW_TITLE)
    fig.text(0.06, 0.95, tns_name, fontname=FONT_SANS, fontsize=13, weight='bold', color=C_TEXT_PRIMARY, va='top')
    fig.text(0.06, 0.93, burst_name.upper(), fontname=FONT_SANS, fontsize=10, color=C_TEXT_PRIMARY, va='top')
    fig.text(0.06, 0.905, f"Observatory: {observatory}", fontname=FONT_SANS, fontsize=8, color=C_TEXT_SECONDARY, va='top')
    
    add_divider(0.25)
    
    # Panel 2: Model Selection
    fig.text(0.27, 0.975, "MODEL SELECTION", **KW_TITLE)
    if "all_results" in results:
        res_all = results["all_results"]
        model_keys = ["M0", "M1", "M2", "M3"]
        x_positions = [0.27, 0.36]
        y_positions = [0.95, 0.93]
        
        for i, k in enumerate(model_keys):
            if k not in res_all: continue
            res_k = res_all[k]
            z = float(res_k.get('log_evidence', 0)) if isinstance(res_k, dict) else float(getattr(res_k, 'log_evidence', 0))
            bic = -2 * z
            x_pos = x_positions[i // 2]
            y_pos = y_positions[i % 2]
            
            is_best = (k == best_key)
            color = C_HIGHLIGHT_BLUE if is_best else C_TEXT_SECONDARY
            weight = 'bold' if is_best else 'normal'
            size = 9 if is_best else 8
            suffix = " ✓" if is_best else ""
            fig.text(x_pos, y_pos, f"{k}: BIC = {bic:.0f}{suffix}", fontname=FONT_SANS, 
                    fontsize=size, weight=weight, color=color, va='top')
    else:
        fig.text(0.27, 0.95, f"{best_key}: Selected ✓", fontname=FONT_SANS, fontsize=9, 
                weight='bold', color=C_HIGHLIGHT_BLUE, va='top')
    
    add_divider(0.45)
    
    # Panel 3: Fit Evaluation
    fig.text(0.47, 0.975, "FIT EVALUATION", **KW_TITLE)
    gof = results.get("goodness_of_fit", {})
    quality = gof.get("quality_flag", "UNKNOWN")
    chi2 = gof.get("chi2_reduced", np.nan)
    r2 = gof.get("r_squared", np.nan)
    
    status_color = C_STATUS_RED if quality == "FAIL" else C_STATUS_GREEN
    fig.text(0.47, 0.95, f"Status: {quality}", fontname=FONT_SANS, fontsize=9, weight='bold', color=status_color, va='top')
    fig.text(0.47, 0.92, f"χ²ᵣ = {chi2:.2f}", fontname=FONT_SANS, fontsize=9, color=C_TEXT_PRIMARY, va='top')
    fig.text(0.57, 0.92, f"R² = {r2:.3f}", fontname=FONT_SANS, fontsize=9, color=C_TEXT_PRIMARY, va='top')
    
    add_divider(0.67)
    
    # Panel 4: Best Fit Parameters
    fig.text(0.69, 0.975, "BEST FIT PARAMETERS", **KW_TITLE)
    param_names = results.get('param_names', [])
    param_strs = []
    flat_chain = results.get('flat_chain', np.array([]))
    
    for i, name in enumerate(param_names):
        val, err = np.nan, 0.0
        if flat_chain.size > 0 and flat_chain.ndim == 2 and i < flat_chain.shape[1]:
            vals = flat_chain[:, i]
            if not np.all(np.isnan(vals)):
                val, err = np.median(vals), np.std(vals)
        
        if np.isnan(val): # Fallback
            val = getattr(results.get('best_params_obj', {}), name, np.nan)

        if abs(val) < 0.001 and val != 0:
            param_strs.append(f"{name} = {val:.2e} ± {err:.1e}")
        else:
            param_strs.append(f"{name} = {val:.3g} ± {err:.1g}")

    x_cols = [0.69, 0.79, 0.89]
    for i, s in enumerate(param_strs):
        col, row = i // 2, i % 2
        if col < len(x_cols):
            fig.text(x_cols[col], 0.95 - (row * 0.018), s, fontname=FONT_SANS, fontsize=8, color=C_TEXT_PRIMARY, va='top')

def plot_scattering_diagnostic(
    data: np.ndarray,
    model: np.ndarray,
    freq: np.ndarray,
    time: np.ndarray,
    params: FRBParams,
    results: dict,
    output_path: Path,
    burst_name: str = "FRB",
    telescope: str = None
):
    """Create 4-panel diagnostic plot with elegant header."""
    # Data Preparation
    q = data.shape[1] // 4
    data_off = data[:, np.r_[0:q, -q:0]]
    noise_std = np.nanstd(data_off, axis=1)
    
    synth_data = model + np.random.normal(0.0, noise_std[:, None], size=data.shape)
    residual = data - synth_data
    
    m_off, s_off = np.nanmean(data_off), np.nanstd(data_off)
    if s_off < 1e-9: s_off = 1.0
    
    p_snr = np.nanmax((data - m_off) / s_off) or 1.0
    norm = lambda x, sub=True: ((x - m_off) if sub else x) / s_off / p_snr
    
    d_norm, m_norm, s_norm, r_norm = norm(data), norm(model), norm(synth_data), norm(residual, False)
    
    # Configure Style
    plt.rcParams.update({
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.labelsize': 16,
    })
    
    # Create Figure
    fig, axes = plt.subplots(nrows=2, ncols=8, gridspec_kw={"height_ratios": [1, 2.5], "width_ratios": [2, 0.5] * 4}, figsize=(24, 8.5))
    
    time_centered = time - (time[0] + (time[-1] - time[0]) / 2)
    extent = [time_centered[0], time_centered[-1], freq[0], freq[-1]]
    
    # shared limits
    ts_list = [np.nansum(x, axis=0) for x in [d_norm, m_norm, s_norm, r_norm]]
    ts_ylim = (min(np.min(t) for t in ts_list) * 1.05, max(np.max(t) for t in ts_list) * 1.05)
    
    panels = [(d_norm, "Data", r"$\mathbf{I}_{\rm data}$"), (m_norm, "Model", r"$\mathbf{I}_{\rm model}$"),
              (s_norm, "Model + Noise", r"$\mathbf{I}_{\rm model} + \mathbf{N}$"), (r_norm, "Residual", r"$\mathbf{I}_{\rm residual}$")]
    
    for i, (ds, title, label) in enumerate(panels):
        ax_ts, ax_wf, ax_sp = axes[0, i*2], axes[1, i*2], axes[1, i*2+1]
        ax_ts.step(time_centered, np.nansum(ds, axis=0), where="mid", c="k", lw=1.5, label=label)
        ax_ts.legend(loc="upper right", fontsize=14, frameon=False)
        ax_ts.set_ylim(ts_ylim); ax_ts.set_yticks([]); ax_ts.set_xlim(extent[0], extent[1])
        ax_ts.tick_params(labelbottom=False)
        
        cmap = "coolwarm" if title == "Residual" else "plasma"
        vmin = -np.nanmax(np.abs(ds)) if title=="Residual" else np.nanpercentile(ds, 1)
        vmax = np.nanmax(np.abs(ds)) if title=="Residual" else np.nanpercentile(ds, 99.5)
        
        ax_wf.imshow(ds, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto", origin="lower")
        ax_wf.set_xlabel("Time [ms]", fontsize=16)
        if i == 0: ax_wf.set_ylabel("Frequency [GHz]", fontsize=16)
        else: ax_wf.tick_params(labelleft=False)
        
        sp = np.nansum(ds, axis=1)
        ax_sp.step(sp, freq, where="mid", c="k", lw=1.5)
        ax_sp.set_yticks([]); ax_sp.set_xticks([]); ax_sp.set_ylim(extent[2], extent[3])
        axes[0, i*2+1].axis("off")

    plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.88, bottom=0.08, left=0.05, right=0.98)

    # Observatory & Metadata
    obs_map = {'chime': 'CHIME/FRB', 'dsa': 'DSA-110', 'dsa110': 'DSA-110'}
    observatory = telescope and obs_map.get(telescope.lower()) or ('CHIME/FRB' if 'chime' in str(output_path).lower() else 'DSA-110')
    tns_name = load_tns_name(burst_name)
    best_key = results.get('best_model', results.get('best_key', 'M3'))
    
    # Ensure best_params is accessible to header helper
    results['best_params_obj'] = params
    
    _draw_diagnostic_header(fig, results, tns_name, burst_name, observatory, best_key)
    
    fig.savefig(output_path, dpi=150)
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
        data, model, freq, time, params, results, output_path, burst_name, args.telescope
    )
    
    if args.show:
        plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()
