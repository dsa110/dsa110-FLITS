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
from scattering.scat_analysis.visualization import plot_scattering_diagnostic

# Burst list
DSA_BURSTS = [
    "casey", "chromatica", "freya", "hamilton", "isha", "johndoeII",
    "mahi", "oran", "phineas", "whitney", "wilhelm", "zach"
]


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
        
        # Step 3: Generate high-res model and plot using standard visualization
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
        
        # Create plot using standard function
        plot_path = output_dir / f"{burst_name}_diagnostic.png"
        
        # Package results dictionary for plot_scattering_diagnostic
        results_for_plot = {
            "best_params": best_params,
            "best_key": "M3",
            "goodness_of_fit": gof,
            "param_names": ["c0", "t0", "gamma", "zeta", "tau_1ghz", "delta_dm"],
            "best_model": "M3",
        }
        
        # Use the standard plotting function
        plot_scattering_diagnostic(
            data=highres_dataset.data,
            model=model_highres,
            freq=highres_dataset.freq,
            time=highres_dataset.time,
            params=best_params,
            results=results_for_plot,
            output_path=plot_path,
            burst_name=burst_name,
            telescope="dsa",
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
