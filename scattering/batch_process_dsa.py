#!/usr/bin/env python3
"""
batch_process_dsa.py
====================

Batch process all DSA-110 bursts with ULTRA_FAST fitting and high-resolution
diagnostic plot generation.

Usage:
    python batch_process_dsa.py                    # Process all bursts
    python batch_process_dsa.py --test --bursts freya  # Test on single burst
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scattering.scat_analysis.burstfit_pipeline import BurstPipeline, BurstDataset
from scattering.scat_analysis.burstfit import FRBModel, FRBParams, goodness_of_fit
from scattering.scat_analysis.config_utils import load_telescope_block

# Burst list
DSA_BURSTS = [
    "casey", "chromatica", "freya", "hamilton", "isha", "johndoeII",
    "mahi", "oran", "phineas", "whitney", "wilhelm", "zach"
]

def create_highres_diagnostic_plot(
    data: np.ndarray,
    model: np.ndarray,
    freq: np.ndarray,
    time: np.ndarray,
    params: FRBParams,
    gof_dict: Dict[str, Any],
    burst_name: str,
    output_path: Path,
):
    """Create 1x4 diagnostic plot with header."""
    
    # Generate synthetic data and residuals
    noise_std = np.nanstd(data[:, :data.shape[1]//4], axis=1)
    synthetic = model + np.random.normal(0, noise_std[:, None], size=data.shape)
    residual = data - synthetic
    
    # Normalization
    q = data.shape[1] // 4
    off_pulse = data[:, np.r_[0:q, -q:0]]
    mean_off, std_off = np.nanmean(off_pulse), np.nanstd(off_pulse)
    if std_off < 1e-9:
        std_off = 1.0
    
    peak_snr = np.nanmax((data - mean_off) / std_off)
    if peak_snr <= 0:
        peak_snr = 1.0
    
    def norm(arr, subtract=True):
        if subtract:
            return (arr - mean_off) / std_off / peak_snr
        return arr / std_off / peak_snr
    
    data_n = norm(data)
    model_n = norm(model)
    synth_n = norm(synthetic)
    resid_n = norm(residual, False)
    
    # Setup figure
    fig, axes = plt.subplots(
        nrows=2, ncols=8,
        gridspec_kw={"height_ratios": [1, 2.5], "width_ratios": [2, 0.5] * 4},
        figsize=(24, 8.5)
    )
    
    time_centered = time - (time[0] + (time[-1] - time[0]) / 2)
    extent = [time_centered[0], time_centered[-1], freq[0], freq[-1]]
    
    # Calculate shared Y-limits for time series
    all_ts = [np.nansum(x, axis=0) for x in [data_n, model_n, synth_n, resid_n]]
    ts_min = min(np.nanmin(t) for t in all_ts if t.size > 0)
    ts_max = max(np.nanmax(t) for t in all_ts if t.size > 0)
    ts_ylim = (ts_min * 1.05, ts_max * 1.05)
    
    # Render 4 panels
    panels = [
        (data_n, "Data", r"$\mathbf{I}_{\rm data}$"),
        (model_n, "Model", r"$\mathbf{I}_{\rm model}$"),
        (synth_n, "Model + Noise", r"$\mathbf{I}_{\rm model} + \mathbf{N}$"),
        (resid_n, "Residual", r"$\mathbf{I}_{\rm residual}$"),
    ]
    
    for i, (ds, title, label) in enumerate(panels):
        col_idx = i * 2
        ax_ts = axes[0, col_idx]
        ax_wf = axes[1, col_idx]
        ax_sp = axes[1, col_idx + 1]
        
        # Time series
        ts = np.nansum(ds, axis=0)
        ax_ts.step(time_centered, ts, where="mid", c="k", lw=1.5, label=label)
        ax_ts.legend(loc="upper right", fontsize=14, frameon=False)
        ax_ts.set_ylim(ts_ylim)
        ax_ts.set_yticks([])
        ax_ts.set_xlim(extent[0], extent[1])
        ax_ts.tick_params(labelbottom=False)
        
        # Waterfall
        cmap = "coolwarm" if title == "Residual" else "plasma"
        if title == "Residual":
            vmax = np.nanmax(np.abs(ds))
            vmin = -vmax
        else:
            vmin = np.nanpercentile(ds, 1)
            vmax = np.nanpercentile(ds, 99.5)
        
        ax_wf.imshow(ds, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap,
                     aspect="auto", origin="lower")
        ax_wf.set_xlabel("Time [ms]", fontsize=16)
        if i == 0:
            ax_wf.set_ylabel("Frequency [GHz]", fontsize=16)
        else:
            ax_wf.tick_params(labelleft=False)
        
        # Spectrum
        sp = np.nansum(ds, axis=1)
        ax_sp.step(sp, freq, where="mid", c="k", lw=1.5)
        ax_sp.set_yticks([])
        ax_sp.set_xticks([])
        ax_sp.set_ylim(extent[2], extent[3])
        
        # Hide top-right corner
        axes[0, col_idx + 1].axis("off")
    
    plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.88, bottom=0.08,
                        left=0.05, right=0.98)
    
    # Add header
    _add_header(fig, burst_name, params, gof_dict, data.shape)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _add_header(fig, burst_name: str, params: FRBParams, gof: Dict, data_shape: tuple):
    """Add informative header to diagnostic plot."""
    
    # Header configuration
    header_y = 0.94
    body_y = 0.91
    fontsize_head = 12
    fontsize_body = 10
    
    # Block 1: Burst Info
    tns_name = f"FRB (DSA-110)"  # Could load from metadata
    fig.text(0.05, header_y, f"{burst_name.upper()}", fontsize=14,
             weight="bold", va="top")
    fig.text(0.05, body_y, f"Observatory: DSA-110\nData: {data_shape[0]}×{data_shape[1]}",
             fontsize=10, va="top")
    
    # Block 2: Fit Mode
    fig.text(0.25, header_y, "Fit Mode", fontsize=fontsize_head, weight="bold", va="top")
    fig.text(0.25, body_y, "ULTRA_FAST\n(Low-res fit,\nHigh-res plot)",
             fontsize=fontsize_body, va="top")
    
    # Block 3: Goodness of Fit
    chi2 = gof.get("chi2_reduced", np.nan)
    r2 = gof.get("r_squared", np.nan)
    quality = gof.get("quality_flag", "UNKNOWN")
    
    fig.text(0.42, header_y, "Goodness of Fit", fontsize=fontsize_head,
             weight="bold", va="top")
    fig.text(0.42, body_y, f"χ²/dof = {chi2:.2f}\nR² = {r2:.3f}",
             fontsize=fontsize_body, va="top")
    
    q_color = "green" if quality == "PASS" else ("red" if quality == "FAIL" else "orange")
    fig.text(0.42, body_y - 0.04, f"Status: {quality}", fontsize=fontsize_body,
             weight="bold", color=q_color, va="top")
    
    # Block 4: Parameters
    fig.text(0.65, header_y, "Best Fit Parameters", fontsize=fontsize_head,
             weight="bold", va="top")
    
    param_text = (
        f"τ@1GHz = {params.tau_1ghz:.4f} ms\n"
        f"α = {params.alpha:.2f}\n"
        f"ζ = {params.zeta:.4f} ms\n"
        f"γ = {params.gamma:.2f}"
    )
    fig.text(0.65, body_y, param_text, fontsize=fontsize_body, va="top",
             fontfamily="monospace")


def process_burst(
    burst_name: str,
    data_dir: Path,
    output_dir: Path,
    telcfg_path: Path,
) -> Dict[str, Any]:
    """Process a single burst: ULTRA_FAST fit + high-res diagnostic plot."""
    
    print(f"\nProcessing {burst_name}...")
    start_time = time.time()
    
    # Find data file
    data_file = data_dir / f"{burst_name}_dsa_I_*_2500b_cntr_bpc.npy"
    matches = list(data_dir.glob(f"{burst_name}_dsa_*.npy"))
    
    if not matches:
        return {"success": False, "error": "Data file not found"}
    
    data_path = matches[0]
    print(f"  Data: {data_path.name}")
    
    try:
        # Step 1: ULTRA_FAST fit
        print(f"  [1/3] Running ULTRA_FAST fit...")
        pipeline = BurstPipeline(
            inpath=data_path,
            outpath=output_dir,
            name=f"{burst_name}_ultrafast",
            dm_init=0.0,
            telescope="dsa",
            telcfg_path=str(telcfg_path),
            t_factor=8,
            f_factor=64,
            steps=200,
            fitting_method="nested",
            likelihood="studentt",
            alpha_fixed=4.0,
            yes=True,
        )
        
        results = pipeline.run_full(
            model_scan=True,
            diagnostics=False,
            plot=False,
            save=False,
            show=False,
            model_keys=["M3"],
        )
        
        best_params = results["best_params"]
        fit_time = time.time() - start_time
        print(f"  ✓ Fit complete ({fit_time:.1f}s): τ={best_params.tau_1ghz:.4f} ms")
        
        # Step 2: Load high-res data
        print(f"  [2/3] Loading high-resolution data...")
        telescope = load_telescope_block(telcfg_path, "dsa")
        
        highres_dataset = BurstDataset(
            inpath=data_path,
            outpath=output_dir,
            name=f"{burst_name}_highres",
            telescope=telescope,
            t_factor=4,
            f_factor=32,
        )
        
        print(f"  ✓ High-res data: {highres_dataset.data.shape}")
        
        # Step 3: Generate high-res model and plot
        print(f"  [3/3] Generating diagnostic plot...")
        highres_model = FRBModel(
            time=highres_dataset.time,
            freq=highres_dataset.freq,
            data=highres_dataset.data,
            df_MHz=highres_dataset.df_MHz,
            dm_init=0.0,
        )
        
        model_highres = highres_model(best_params, "M3")
        
        # Calculate GoF on high-res
        gof = goodness_of_fit(
            highres_dataset.data,
            model_highres,
            highres_model.noise_std,
            n_params=7,
        )
        
        # Create plot
        plot_path = output_dir / f"{burst_name}_diagnostic.png"
        create_highres_diagnostic_plot(
            highres_dataset.data,
            model_highres,
            highres_dataset.freq,
            highres_dataset.time,
            best_params,
            gof,
            burst_name,
            plot_path,
        )
        
        total_time = time.time() - start_time
        print(f"  ✓ Complete in {total_time:.1f}s → {plot_path.name}")
        
        return {
            "success": True,
            "burst_name": burst_name,
            "tau_1ghz": float(best_params.tau_1ghz),
            "alpha": float(best_params.alpha),
            "zeta": float(best_params.zeta),
            "gamma": float(best_params.gamma),
            "t0": float(best_params.t0),
            "delta_dm": float(best_params.delta_dm),
            "chi2_reduced_highres": float(gof["chi2_reduced"]),
            "r_squared_highres": float(gof["r_squared"]),
            "quality_highres": gof["quality_flag"],
            "fit_time_sec": fit_time,
            "total_time_sec": total_time,
            "plot_file": str(plot_path.name),
        }
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "burst_name": burst_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch process DSA-110 bursts")
    parser.add_argument("--test", action="store_true", help="Test mode (single burst)")
    parser.add_argument("--bursts", nargs="+", help="Specific bursts to process")
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "dsa"
    output_dir = base_dir / "scattering" / "dsa_diagnostics"
    output_dir.mkdir(exist_ok=True)
    
    telcfg_path = base_dir / "scattering" / "configs" / "telescopes.yaml"
    
    # Determine burst list
    if args.bursts:
        bursts_to_process = args.bursts
    elif args.test:
        bursts_to_process = ["freya"]
    else:
        bursts_to_process = DSA_BURSTS
    
    print("="*60)
    print(f"DSA-110 Batch Processing")
    print("="*60)
    print(f"Mode: {'TEST' if args.test else 'FULL BATCH'}")
    print(f"Bursts: {len(bursts_to_process)}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Process bursts
    results = []
    for burst in tqdm(bursts_to_process, desc="Processing bursts"):
        result = process_burst(burst, data_dir, output_dir, telcfg_path)
        results.append(result)
    
    # Save summary
    summary_file = output_dir / "dsa_fitting_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Batch Processing Complete")
    print(f"{'='*60}")
    
    # Print summary
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f.get('burst_name', 'unknown')}: {f.get('error', 'unknown error')}")
    
    print(f"\nResults saved to: {summary_file}")
    print(f"Diagnostic plots in: {output_dir}/")
    
    # Create CSV summary
    if successful:
        import csv
        csv_file = output_dir / "dsa_fitting_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ["burst_name", "tau_1ghz", "alpha", "zeta", "gamma",
                         "chi2_reduced_highres", "r_squared_highres", "fit_time_sec"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in successful:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        
        print(f"CSV summary: {csv_file}")


if __name__ == "__main__":
    main()
