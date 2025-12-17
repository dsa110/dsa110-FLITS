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
import csv

# --- Package-relative imports (work via console-script entry point) ---
try:
    from .scat_analysis.burstfit_pipeline import BurstPipeline
    from .scat_analysis.burstfit_corner import (
        quick_chain_check,
        get_clean_samples,
        make_beautiful_corner,
    )
    from .scat_analysis.config_utils import load_config
except Exception:
    # Fallback for direct execution without installation:
    # add this directory to sys.path and import again
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from scat_analysis.burstfit_pipeline import BurstPipeline
        from scat_analysis.burstfit_corner import (
            quick_chain_check,
            get_clean_samples,
            make_beautiful_corner,
        )
        from scat_analysis.config_utils import load_config
    except Exception as e:
        print("Error: Could not import 'scat_analysis'.")
        print("Try installing the package (pip install -e .) and using 'flits-scat',")
        print("or run with: python -m FLITS.scattering.run_scat_analysis <args>.")
        print(f"Details: {e}")
        sys.exit(1)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run the full BurstFit pipeline using a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the YAML run configuration file (e.g., 'configs/dsa/casey_dsa.yaml').",
    )
    # --- FIX: Add arguments for general config files ---
    parser.add_argument("--telcfg", type=Path, help="Override path to telescopes.yaml.")
    parser.add_argument("--sampcfg", type=Path, help="Override path to sampler.yaml.")

    # Optional overrides
    parser.add_argument("--path", type=Path, help="Override the data file path.")
    parser.add_argument(
        "--dm_init", type=float, help="Override the initial Dispersion Measure."
    )
    parser.add_argument("--steps", type=int, help="Override the number of MCMC steps.")
    parser.add_argument("--nproc", type=int, help="Override the number of cores.")
    parser.add_argument(
        "--extend-chain", action="store_true", dest="extend_chain", default=None
    )
    parser.add_argument("--no-extend-chain", action="store_false", dest="extend_chain")
    # New modeling controls
    parser.add_argument("--alpha-fixed", type=float, default=None)
    parser.add_argument("--alpha-mu", type=float, default=4.4)
    parser.add_argument("--alpha-sigma", type=float, default=0.6)
    parser.add_argument("--delta-dm-sigma", type=float, default=0.1)
    parser.add_argument(
        "--likelihood", choices=["gaussian", "studentt"], default="gaussian"
    )
    parser.add_argument("--studentt-nu", type=float, default=5.0)
    parser.add_argument("--no-logspace", dest="sample_log_params", action="store_false")
    parser.add_argument(
        "--ncomp",
        type=int,
        default=1,
        help="Number of Gaussian components (shared PBF)",
    )
    parser.add_argument(
        "--auto-components",
        action="store_true",
        help="Greedy BIC-based component selection",
    )
    # DM Refinement (NEW)
    parser.add_argument(
        "--refine-dm",
        action="store_true",
        help="Run phase-coherence DM estimation before scattering analysis",
    )
    parser.add_argument(
        "--dm-search-window",
        type=float,
        default=5.0,
        help="Half-width of DM search range (pc/cm³) for refinement",
    )
    parser.add_argument(
        "--dm-grid-resolution",
        type=float,
        default=0.01,
        help="DM grid spacing (pc/cm³) for refinement",
    )
    parser.add_argument(
        "--dm-n-bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap samples for DM uncertainty estimation",
    )
    # Pipeline control flags
    parser.add_argument(
        "--no-scan",
        dest="model_scan",
        action="store_false",
        default=None,
        help="Skip model selection scan",
    )
    parser.add_argument(
        "--no-diag",
        dest="diagnostics",
        action="store_false",
        default=None,
        help="Skip post-fit diagnostics",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        default=None,
        help="Skip plotting",
    )
    # Earmarks / placeholders
    parser.add_argument("--anisotropy-enabled", action="store_true")
    parser.add_argument("--anisotropy-axial-ratio", type=float, default=1.0)
    parser.add_argument("--baseline-order", type=int, default=0)
    parser.add_argument("--correlated-resid", action="store_true")
    parser.add_argument("--sampler", choices=["emcee", "nested"], default="emcee")
    parser.add_argument(
        "--init-guess",
        type=str,
        default=None,
        help="Path to JSON seed for initial guess",
    )
    parser.add_argument(
        "--walker-width-frac",
        type=float,
        default=0.01,
        help="Fraction of prior span for walker init width",
    )

    args = parser.parse_args()

    print(f"--- Loading configuration from: {args.config_path} ---")
    config = load_config(args.config_path)

    # Extract values from Config dataclass
    data_path = config.path
    dm_init = config.dm_init
    outpath = args.path.parent if args.path else data_path.parent
    frb_name = data_path.stem

    # Override with CLI arguments if provided
    if args.path:
        data_path = args.path
    if args.dm_init is not None:
        dm_init = args.dm_init

    # Build pipeline kwargs from config and CLI overrides
    pipeline_kwargs = {
        "telescope": config.telescope,
        "sampler": config.sampler,
        "f_factor": config.pipeline.f_factor,
        "t_factor": config.pipeline.t_factor,
        "steps": args.steps if args.steps else config.pipeline.steps,
        "nproc": args.nproc if args.nproc else config.pipeline.nproc,
        "extend_chain": (
            args.extend_chain
            if args.extend_chain is not None
            else config.pipeline.extend_chain
        ),
        # Pass through additional CLI modeling controls
        "alpha_fixed": args.alpha_fixed,
        "alpha_mu": args.alpha_mu,
        "alpha_sigma": args.alpha_sigma,
        "delta_dm_sigma": args.delta_dm_sigma,
        "likelihood": args.likelihood,
        "studentt_nu": args.studentt_nu,
        "sample_log_params": args.sample_log_params,
        "ncomp": args.ncomp,
        "auto_components": args.auto_components,
        "init_guess": args.init_guess,
        "walker_width_frac": args.walker_width_frac,
        # DM refinement controls (NEW)
        "refine_dm": args.refine_dm,
        "dm_search_window": args.dm_search_window,
        "dm_grid_resolution": args.dm_grid_resolution,
        "dm_n_bootstrap": args.dm_n_bootstrap,
    }

    print(f"\n--- Starting analysis for: {data_path.name} ---")

    # Use DM from config or CLI
    if dm_init is None:
        dm_init = 0.0

    pipe = BurstPipeline(
        inpath=data_path,
        outpath=outpath,
        name=frb_name,
        dm_init=dm_init,
        **pipeline_kwargs,
    )

    # Determine run options (CLI overrides config)
    run_model_scan = (
        args.model_scan if args.model_scan is not None else config.pipeline.model_scan
    )
    run_diagnostics = (
        args.diagnostics
        if args.diagnostics is not None
        else config.pipeline.diagnostics
    )
    run_plot = args.plot if args.plot is not None else config.pipeline.plot

    results = pipe.run_full(
        model_scan=run_model_scan,
        diagnostics=run_diagnostics,
        plot=run_plot,
        show=False,
    )

    print("\n--- Initial Pipeline Run Summary ---")
    print(f"Best model found: {results['best_key']}")
    if results.get("goodness_of_fit"):
        print(f"Reduced Chi-squared: {results['goodness_of_fit']['chi2_reduced']:.2f}")
    print("Best-fit parameters (from highest-likelihood sample):")
    # print(results["best_params"]) # Replaced by consolidated summary
    print_fit_summary(results)

    extend_chain = pipeline_kwargs.get("extend_chain", config.pipeline.extend_chain)
    if extend_chain:
        sampler = results["sampler"]
        sampler.pool = None

        print("\n--- Starting Interactive Chain Convergence Check ---")
        chunks_added = 0
        max_chunks = config.pipeline.max_chunks or 5
        chunk_size = config.pipeline.chunk_size or 2000

        while not quick_chain_check(sampler):
            if chunks_added >= max_chunks:
                print(
                    f"Reached max extra steps ({max_chunks * chunk_size}); proceeding."
                )
                break
            print(
                f"\nChain not fully converged. Running for {chunk_size} more steps..."
            )
            sampler.run_mcmc(None, chunk_size, progress=True)
            chunks_added += 1

        # --- 5. Generate and Save Final Corner Plot ---
        print("\n--- Generating Final Corner Plot ---")
        param_names = results["param_names"]
        best_p = results[
            "best_params"
        ]  # Use original best-fit as truth value for the plot

        final_clean_samples = get_clean_samples(sampler, param_names, verbose=True)

        fig_corner = make_beautiful_corner(
            final_clean_samples,
            param_names,
            best_params=best_p,
            title=f"Posterior for {results['best_key']} ({final_clean_samples.shape[0]} samples)",
        )

        corner_path = data_path.with_name(f"{data_path.stem}_corner.png")
        fig_corner.savefig(corner_path, dpi=200, bbox_inches="tight")
        print(f"Saved final corner plot to: {corner_path}")
        plt.show()

    print("\n--- Analysis complete. ---")


if __name__ == "__main__":
    main()
