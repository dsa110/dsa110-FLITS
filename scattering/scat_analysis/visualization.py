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
    Create 4-panel diagnostic plot.
    
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
        Full results dictionary
    output_path : Path
        Output file path
    burst_name : str
        Name of the burst for title
    """
    # Set generic plotting style to match reference
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'lines.linewidth': 1.5
    })

    # Scale model to match data for visualization
    data_peak = np.max(np.sum(data, axis=0))
    model_peak = np.max(np.sum(model, axis=0))
    scale = data_peak / max(model_peak, 1e-10)
    model_scaled = model * scale
    
    # Calculate residuals
    residual = data - model_scaled
    
    # Create figure: 1 row, 5 columns (Data, Model, Resid, Profile, Stats)
    # Aspect ratio: Standard 4-panel is wide but 2 rows. Here 1 row.
    # Each plot needs width. Let's try 20x5.
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    
    # Extent for imshow: [left, right, bottom, top]
    extent = [time[0], time[-1], freq[0], freq[-1]]
    
    # Colormap limits
    vmax = np.percentile(data, 99.5)
    
    # Helper to clean up axes
    def format_ax(ax, title, ylabel=True, cbar_im=None, cbar_label="S/N"):
        ax.set_title(title, pad=10)
        ax.set_xlabel("Time (ms)")
        if ylabel:
            ax.set_ylabel("Freq (GHz)")
        else:
            ax.set_yticklabels([])
    for ax in axes[:4]: # Only for plot axes
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Panel 5: Stats Table
    from flits.utils.reporting import get_fit_summary_lines
    stats_lines = get_fit_summary_lines(results, table_format="ascii")
    
    # Clean up lines for plot display (remove separators like --- or === if desired, or keep as monospace block)
    # User "Markdown style" might mean formatted. But Matplotlib text is easier with monospace block.
    # Let's keep the block but remove top/bottom empty lines
    stats_text = "\n".join([l for l in stats_lines if l.strip() != ""])
    
    axes[4].axis('off')
    # Using a monospaced font to preserve table alignment
    # Place text roughly centered vertically, or top aligned?
    # "Format it so that its height ... spans an equivalent height"
    # Vertical alignment 'center' might be best if we want it to span.
    # Or 'top' with adjusted bounds.
    axes[4].text(0.0, 0.5, stats_text, 
                 family='monospace', fontsize=10,
                 va='center', ha='left')
    axes[4].set_title("Fit Statistics", pad=10)

    # Output parameters title
    gof = results['goodness_of_fit']
    title_text = (
        f"{burst_name} - {results.get('best_model', results.get('best_key', 'Unknown'))} Fit (χ²/dof={gof['chi2_reduced']:.2f}, R²={gof['r_squared']:.2f})\n"
    )
    if hasattr(params, 'tau_1ghz') and params.tau_1ghz > 1e-6:
        title_text += (
            f"τ(1GHz)={params.tau_1ghz:.3f}ms, α={params.alpha:.1f}, t₀={params.t0:.2f}ms"
        )
    
    plt.suptitle(title_text, fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
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
