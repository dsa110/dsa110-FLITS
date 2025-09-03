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
from pathlib import Path
import matplotlib.pyplot as plt

from scat_analysis.config_utils import (
    Config,
    PipelineOptions,
    load_config,
    load_sampler_block,
    load_telescope_block,
)

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
    config = load_config(args.config_path)

    # Optional overrides from CLI
    if args.path:
        config.path = Path(args.path)
    if args.dm_init is not None:
        config.dm_init = args.dm_init
    if args.steps is not None:
        config.pipeline.steps = args.steps
    if args.nproc is not None:
        config.pipeline.nproc = args.nproc
    if args.extend_chain is not None:
        config.pipeline.extend_chain = args.extend_chain

    # Overrides for telescope/sampler YAML paths
    if args.telcfg:
        config.telescope = load_telescope_block(args.telcfg, config.telescope.name)
    if args.sampcfg:
        config.sampler = load_sampler_block(args.sampcfg, config.sampler.name)

    print(f"\n--- Starting analysis for: {config.path.name} ---")

    pipe = BurstPipeline(
        inpath=config.path,
        outpath=config.path.parent,
        name=config.path.stem,
        dm_init=config.dm_init,
        telescope=config.telescope,
        sampler=config.sampler,
        steps=config.pipeline.steps,
        f_factor=config.pipeline.f_factor,
        t_factor=config.pipeline.t_factor,
        nproc=config.pipeline.nproc,
        extend_chain=config.pipeline.extend_chain,
        chunk_size=config.pipeline.chunk_size,
        max_chunks=config.pipeline.max_chunks,
    )
    results = pipe.run_full(
        model_scan=config.pipeline.model_scan,
        diagnostics=config.pipeline.diagnostics,
        plot=config.pipeline.plot,
        show=False,
    )
    
    print("\n--- Initial Pipeline Run Summary ---")
    print(f"Best model found: {results['best_key']}")
    if results.get('goodness_of_fit'):
        print(f"Reduced Chi-squared: {results['goodness_of_fit']['chi2_reduced']:.2f}")
    print("Best-fit parameters (from highest-likelihood sample):")
    print(results['best_params'])

    if config.pipeline.extend_chain:
        sampler = results["sampler"]
        sampler.pool = None

        print("\n--- Starting Interactive Chain Convergence Check ---")
        chunks_added = 0
        max_chunks = config.pipeline.max_chunks or 5
        chunk_size = config.pipeline.chunk_size or 2000
        
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

        corner_path = config.path.with_name(f"{config.path.stem}_corner.png")
        fig_corner.savefig(corner_path, dpi=200, bbox_inches="tight")
        print(f"Saved final corner plot to: {corner_path}")
        plt.show()

    print("\n--- Analysis complete. ---")


if __name__ == "__main__":
    main()
