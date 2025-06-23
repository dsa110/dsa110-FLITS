"""
run_analysis.py
===============

Command-line driver for the full BurstFit pipeline.

This script uses a YAML configuration file to manage run parameters,
while allowing command-line arguments to override any setting for a
specific run.

Primary Usage:
    python run_analysis.py /path/to/your/run_config.yaml

Overriding a setting from the command line:
    python run_analysis.py configs/dsa/casey_dsa.yaml --steps 500 --no-extend-chain
"""
import sys
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# --- Ensure the project's root directory is in the Python path ---
# This allows the script to be run from anywhere and still find the scat_analysis module.
# It assumes this script is in the project root, and the package is in 'scat_analysis/'.
try:
    from scat_analysis.burstfit_pipeline import BurstPipeline
    from scat_analysis.burstfit_corner import (
        quick_chain_check,
        get_clean_samples,
        make_beautiful_corner
    )
except ImportError:
    print("Error: Could not import the 'scat_analysis' package.")
    print("Please ensure this script is in the project's root directory,")
    print("or add the project root to your PYTHONPATH.")
    sys.exit(1)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run the full BurstFit pipeline using a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_path", 
        type=Path, 
        help="Path to the YAML run configuration file (e.g., 'configs/dsa/casey_dsa.yaml')."
    )
    # --- FIX: Add arguments for general config files ---
    parser.add_argument("--telcfg", type=Path, help="Override path to telescopes.yaml.")
    parser.add_argument("--sampcfg", type=Path, help="Override path to sampler.yaml.")
    
    # Optional overrides
    parser.add_argument("--path", type=Path, help="Override the data file path.")
    parser.add_argument("--dm_init", type=float, help="Override the initial Dispersion Measure.")
    parser.add_argument("--steps", type=int, help="Override the number of MCMC steps.")
    parser.add_argument("--nproc", type=int, help="Override the number of cores.")
    parser.add_argument("--extend-chain", action='store_true', dest='extend_chain', default=None)
    parser.add_argument("--no-extend-chain", action='store_false', dest='extend_chain')
    
    args = parser.parse_args()

    print(f"--- Loading configuration from: {args.config_path} ---")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- FIX: Smartly determine paths for general configs ---
    # The base directory is the directory of the run-specific config file.
    config_base_dir = args.config_path.parent

    # If --telcfg was NOT provided, assume telescopes.yaml is in the same dir
    # as the run config. Otherwise, use the provided path.
    telcfg_path = args.telcfg if args.telcfg else config_base_dir / "telescopes.yaml"
    sampcfg_path = args.sampcfg if args.sampcfg else config_base_dir / "sampler.yaml"

    # Add these paths to the config dictionary
    config['telcfg_path'] = telcfg_path
    config['sampcfg_path'] = sampcfg_path
    
    # Override other config settings with command-line arguments
    for key, value in vars(args).items():
        if key not in ['config_path', 'telcfg', 'sampcfg'] and value is not None:
            config[key] = value
            print(f"  -> Overriding '{key}' with command-line value: {value}")
            
    if 'path' not in config:
        raise ValueError("Data file 'path' must be specified in the YAML config or via --path.")

    # --- Run the Pipeline ---
    data_path = Path(config.pop('path'))
    dm_init = config.pop('dm_init', 0.0)
    
    print(f"\n--- Starting analysis for: {data_path.name} ---")

    pipe = BurstPipeline(path=data_path, dm_init=dm_init, **config)
    results = pipe.run_full(
        model_scan=config.get('model_scan', True),
        diagnostics=config.get('diagnostics', True),
        plot=config.get('plot', True),
        show=False
    )
    
    print("\n--- Initial Pipeline Run Summary ---")
    print(f"Best model found: {results['best_key']}")
    if results.get('goodness_of_fit'):
        print(f"Reduced Chi-squared: {results['goodness_of_fit']['chi2_reduced']:.2f}")
    print("Best-fit parameters (from highest-likelihood sample):")
    print(results['best_params'])

    if config.get('extend_chain', False):
        sampler = results["sampler"]
        sampler.pool = None

        print("\n--- Starting Interactive Chain Convergence Check ---")
        chunks_added = 0
        max_chunks = config.get('max_chunks', 5)
        chunk_size = config.get('chunk_size', 2000)
        
        while not quick_chain_check(sampler):
            if chunks_added >= max_chunks:
                print(f"Reached max extra steps ({max_chunks * chunk_size}); proceeding.")
                break
            print(f"\nChain not fully converged. Running for {chunk_size} more steps...")
            sampler.run_mcmc(None, chunk_size, progress=True)
            chunks_added += 1

        # --- 5. Generate and Save Final Corner Plot ---
        print("\n--- Generating Final Corner Plot ---")
        param_names = results["param_names"]
        best_p = results["best_params"] # Use original best-fit as truth value for the plot
        
        final_clean_samples = get_clean_samples(sampler, param_names, verbose=True)

        fig_corner = make_beautiful_corner(
            final_clean_samples, param_names, best_params=best_p,
            title=f"Posterior for {results['best_key']} ({final_clean_samples.shape[0]} samples)"
        )

        corner_path = data_path.with_name(f"{data_path.stem}_corner.png")
        fig_corner.savefig(corner_path, dpi=200, bbox_inches="tight")
        print(f"Saved final corner plot to: {corner_path}")
        plt.show()

    print("\n--- Analysis complete. ---")


if __name__ == "__main__":
    main()
