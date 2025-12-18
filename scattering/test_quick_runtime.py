#!/usr/bin/env python3
"""
test_quick_runtime.py
=====================

Test scattering analysis with aggressive optimizations and 60-second timeout.

This script benchmarks the pipeline with various optimization levels to find
the fastest configuration that still produces scientifically valid results.
"""

import sys
import time
import signal
import logging
from pathlib import Path
from contextlib import contextmanager

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scattering.scat_analysis.burstfit_pipeline import BurstPipeline
from scattering.scat_analysis.config_utils import load_telescope_block

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when analysis exceeds time limit."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to enforce time limit on code execution."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Analysis exceeded {seconds} second time limit!")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def test_optimization_levels(data_path: Path, output_dir: Path):
    """Test different optimization levels and measure runtime.
    
    Returns
    -------
    results : dict
        Runtime and quality metrics for each optimization level
    """
    
    # Determine telescope from filename
    telescope_name = "CHIME" if "chime" in str(data_path).lower() else "DSA110"
    
    optimization_configs = {
        "ULTRA_FAST": {
            "t_factor": 8,
            "f_factor": 64,
            "steps": 200,
            "fitting_method": "nested",
            "likelihood": "studentt",
            "alpha_fixed": 4.0,
            "model_keys": ["M3"],  # Only fit the full model
        },
        "FAST": {
            "t_factor": 4,
            "f_factor": 32,
            "steps": 300,
            "fitting_method": "nested",
            "likelihood": "studentt",
            "alpha_fixed": 4.0,
            "model_keys": ["M2", "M3"],
        },
        "STANDARD": {
            "t_factor": 4,
            "f_factor": 32,
            "steps": 500,
            "fitting_method": "nested",
            "likelihood": "studentt",
            "alpha_fixed": 4.0,
            "model_keys": ["M0", "M1", "M2", "M3"],
        },
    }
    
    results = {}
    
    for level_name, config in optimization_configs.items():
        log.info(f"\n{'='*60}")
        log.info(f"Testing {level_name} optimization level")
        log.info(f"{'='*60}")
        log.info(f"Config: {config}")
        
        start_time = time.time()
        
        try:
            with time_limit(60):
                # Create pipeline with configuration
                pipeline = BurstPipeline(
                    inpath=data_path,
                    outpath=output_dir,
                    name=f"test_{level_name.lower()}",
                    dm_init=0.0,
                    telescope=telescope_name,
                    **config,
                )
                
                # Run pipeline without plotting or extensive diagnostics
                pipeline_results = pipeline.run_full(
                    model_scan=True,
                    diagnostics=False,  # Skip time-consuming diagnostics
                    plot=False,         # Skip plotting
                    save=False,
                    show=False,
                    model_keys=config.get("model_keys", ["M3"]),
                )
                
                elapsed = time.time() - start_time
                
                # Extract key results
                best_params = pipeline_results.get("best_params")
                gof = pipeline_results.get("goodness_of_fit", {})
                
                # Compute goodness of fit if not already done
                if not gof and pipeline.dataset and pipeline.dataset.model:
                    from scattering.scat_analysis.burstfit import goodness_of_fit as compute_gof
                    model_dyn = pipeline.dataset.model(best_params, pipeline_results["best_key"])
                    gof = compute_gof(
                        pipeline.dataset.data,
                        model_dyn,
                        pipeline.dataset.model.noise_std,
                        n_params=len(pipeline_results.get("param_names", [])),
                    )
                
                results[level_name] = {
                    "success": True,
                    "runtime_sec": elapsed,
                    "chi2_reduced": gof.get("chi2_reduced", np.nan),
                    "r_squared": gof.get("r_squared", np.nan),
                    "quality_flag": gof.get("quality_flag", "UNKNOWN"),
                    "tau_1ghz": best_params.tau_1ghz if best_params else np.nan,
                    "alpha": best_params.alpha if best_params else np.nan,
                    "data_shape": pipeline.dataset.data.shape if pipeline.dataset else None,
                    "config": config,
                }
                
                log.info(f"✓ {level_name} completed in {elapsed:.1f}s")
                log.info(f"  Data shape: {results[level_name]['data_shape']}")
                log.info(f"  χ²/dof: {results[level_name]['chi2_reduced']:.3f}")
                log.info(f"  R²: {results[level_name]['r_squared']:.3f}")
                log.info(f"  τ@1GHz: {results[level_name]['tau_1ghz']:.4f} ms")
                
        except TimeoutError as e:
            elapsed = time.time() - start_time
            log.error(f"✗ {level_name} TIMEOUT after {elapsed:.1f}s")
            results[level_name] = {
                "success": False,
                "runtime_sec": elapsed,
                "error": str(e),
                "config": config,
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            log.error(f"✗ {level_name} FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            results[level_name] = {
                "success": False,
                "runtime_sec": elapsed,
                "error": str(e),
                "config": config,
            }
    
    return results


def print_summary(results: dict):
    """Print formatted summary of benchmark results."""
    
    print("\n" + "="*80)
    print("RUNTIME OPTIMIZATION BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\n{'Level':<15} {'Status':<10} {'Runtime':<12} {'χ²/dof':<10} {'R²':<10} {'τ@1GHz':<10}")
    print("-"*80)
    
    for level, res in results.items():
        status = "✓ PASS" if res["success"] else "✗ FAIL"
        runtime = f"{res['runtime_sec']:.1f}s"
        chi2 = f"{res.get('chi2_reduced', np.nan):.3f}" if res["success"] else "N/A"
        r2 = f"{res.get('r_squared', np.nan):.3f}" if res["success"] else "N/A"
        tau = f"{res.get('tau_1ghz', np.nan):.4f}" if res["success"] else "N/A"
        
        print(f"{level:<15} {status:<10} {runtime:<12} {chi2:<10} {r2:<10} {tau:<10}")
    
    print("\n" + "="*80)
    
    # Find fastest successful configuration
    successful = [(k, v) for k, v in results.items() if v["success"]]
    if successful:
        fastest = min(successful, key=lambda x: x[1]["runtime_sec"])
        print(f"\n🏆 FASTEST SUCCESSFUL: {fastest[0]} ({fastest[1]['runtime_sec']:.1f}s)")
        print(f"   Data shape: {fastest[1].get('data_shape', 'N/A')}")
        print(f"   Quality: {fastest[1].get('quality_flag', 'N/A')}")
    else:
        print("\n⚠️  NO SUCCESSFUL RUNS - All configurations failed or timed out")
    
    print("="*80 + "\n")


def main():
    """Main entry point."""
    
    # Find test data
    base_dir = Path(__file__).parent.parent
    
    # Look for Casey CHIME data (from conversation history)
    data_options = [
        base_dir / "data" / "chime" / "casey_chime_I_491_2085_32000b_cntr_bpc.npy",
        base_dir / "scattering" / "data" / "chime" / "casey_chime_I_491_2085_32000b_cntr_bpc.npy",
    ]
    
    data_path = None
    for path in data_options:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        log.error("Could not find test data file. Checked:")
        for p in data_options:
            log.error(f"  - {p}")
        sys.exit(1)
    
    log.info(f"Using data file: {data_path}")
    
    # Output directory
    output_dir = base_dir / "scattering" / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Run benchmark
    log.info("\n" + "="*80)
    log.info("Starting runtime optimization benchmark with 60s timeout limit")
    log.info("="*80 + "\n")
    
    results = test_optimization_levels(data_path, output_dir)
    
    # Print summary
    print_summary(results)
    
    # Save results
    import json
    results_file = output_dir / "runtime_benchmark_results.json"
    
    # Convert to JSON-serializable format
    json_results = {}
    for level, res in results.items():
        json_results[level] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in res.items()
            if k != "config"  # Skip config for now
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    log.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
