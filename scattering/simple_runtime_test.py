#!/usr/bin/env python3
"""
simple_runtime_test.py
======================
Quick test of runtime with minimal configuration.
"""

import sys
import time
import signal
import logging
from pathlib import Path

import numpy as np

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scattering.scat_analysis.burstfit_pipeline import BurstPipeline

def main():
    # Data path
    data_path = Path(__file__).parent.parent / "data" / "chime" / "casey_chime_I_491_2085_32000b_cntr_bpc.npy"
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Using data: {data_path}")
    print("Starting ULTRA_FAST test...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Ultra-fast configuration
        pipeline = BurstPipeline(
            inpath=data_path,
            outpath=output_dir,
            name="ultra_fast_test",
            dm_init=0.0,
            telescope="chime",
            t_factor=8,
            f_factor=64,
            steps=200,
            fitting_method="nested",
            likelihood="studentt",
            alpha_fixed=4.0,
            yes=True,  # Skip interactive prompts
        )
        
        print("Pipeline created, running fit...")
        
        results = pipeline.run_full(
            model_scan=True,
            diagnostics=False,
            plot=False,
            save=False,
            show=False,
            model_keys=["M3"],
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✓ COMPLETED in {elapsed:.1f} seconds")
        print(f"{'='*60}")
        
        best_params = results.get("best_params")
        if best_params:
            print(f"\nResults:")
            print(f"  τ@1GHz: {best_params.tau_1ghz:.4f} ms")
            print(f"  α: {best_params.alpha:.3f}")
            print(f"  Data shape: {pipeline.dataset.data.shape}")
            
            # Save results to JSON
            import json
            results_file = output_dir / "ultra_fast_test_fit_results.json"
            
            # Convert results to JSON-serializable format
            results_dict = {
                "best_params": {
                    "c0": best_params.c0,
                    "t0": best_params.t0,
                    "gamma": best_params.gamma,
                    "zeta": best_params.zeta,
                    "tau_1ghz": best_params.tau_1ghz,
                    "alpha": best_params.alpha,
                    "delta_dm": best_params.delta_dm,
                },
                "best_key": results.get("best_key", "M3"),
                "runtime_seconds": elapsed,
                "data_shape": list(pipeline.dataset.data.shape),
            }
            
            # Add goodness of fit if available
            if "goodness_of_fit" in results:
                gof = results["goodness_of_fit"]
                results_dict["goodness_of_fit"] = {
                    k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                    for k, v in gof.items()
                    if k != "residual_autocorr"  # Skip arrays
                }
            
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"\n✓ Results saved to: {results_file}")

        
        if elapsed > 60:
            print(f"\n⚠️  WARNING: Exceeded 60s timeout ({elapsed:.1f}s)")
        else:
            print(f"\n✓ PASSED: Under 60s limit")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ FAILED after {elapsed:.1f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
