#!/usr/bin/env python3
"""
Scattering Analysis Pipeline Script
====================================

This script performs FRB scattering analysis using the FLITS pipeline.

Workflow:
1. Load configuration from YAML file
2. Create BurstPipeline instance
3. (Optional) Interactively refine initial guess
4. Run MCMC fitting
5. Generate diagnostic plots and corner plots

Usage:
    python scattering_analysis.py --config ../configs/bursts/dsa/casey_dsa.yaml
    
For interactive initial guess refinement, run in Jupyter/IPython.

Author: FLITS Team
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PATH SETUP
# =============================================================================

# Add parent directory and FLITS root to path for imports
script_dir = Path(__file__).parent.resolve()
scattering_dir = script_dir.parent
flits_root = scattering_dir.parent
sys.path.insert(0, str(scattering_dir))
sys.path.insert(0, str(flits_root))

# =============================================================================
# IMPORTS
# =============================================================================

# Install required packages if missing
try:
    import chainconsumer, seaborn, emcee, arviz
except ImportError:
    print("Installing required packages...")
    os.system("pip install seaborn emcee chainconsumer arviz")

# Import pipeline components
from scat_analysis.burstfit_pipeline import BurstPipeline
from scat_analysis.burstfit_corner import (
    quick_chain_check,
    get_clean_samples,
    make_beautiful_corner,
    make_beautiful_corner_wide,
)
from scat_analysis.burstfit import FRBParams


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_burst_config(config_path: Path) -> dict:
    """
    Load burst configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the burst configuration YAML file.

    Returns
    -------
    dict
        Configuration dictionary with extracted parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract key parameters
    data_path = Path(config["path"])
    burst_name = data_path.stem.split("_")[0]
    telescope = config["telescope"]
    dm_initial = config.get("dm_init", 0.0)

    # Set output directory
    plot_dir = Path(config.get("outpath", f"../plots/{telescope}"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Burst: {burst_name}")
    print(f"Telescope: {telescope}")
    print(f"Data: {data_path}")
    print(f"Output: {plot_dir}")

    return {
        "config": config,
        "data_path": data_path,
        "burst_name": burst_name,
        "telescope": telescope,
        "dm_initial": dm_initial,
        "plot_dir": plot_dir,
    }


# =============================================================================
# PIPELINE CREATION
# =============================================================================


def create_pipeline(burst_config: dict) -> BurstPipeline:
    """
    Create a BurstPipeline instance from configuration.

    Parameters
    ----------
    burst_config : dict
        Configuration dictionary from load_burst_config().

    Returns
    -------
    BurstPipeline
        Configured pipeline instance ready for fitting.
    """
    config = burst_config["config"]

    pipeline_params = {
        "telescope": config["telescope"],
        "telcfg_path": "../configs/telescopes.yaml",
        "sampcfg_path": "../configs/sampler.yaml",
        "steps": config.get("steps", 1000),
        "f_factor": config.get("f_factor", 384),
        "t_factor": config.get("t_factor", 2),
        "center_burst": config.get("center_burst", True),
        "outer_trim": config.get("outer_trim", 0.49),
        "smooth_ms": config.get("smooth_ms", 0.1),
        "nproc": config.get("nproc", 16),
        "yes": True,  # Auto-confirm pool creation
    }

    pipe = BurstPipeline(
        name=burst_config["burst_name"],
        inpath=burst_config["data_path"],
        outpath=burst_config["plot_dir"],
        dm_init=burst_config["dm_initial"],
        **pipeline_params,
    )

    print(f"Pipeline created for {burst_config['burst_name']} ({config['telescope'].upper()})")
    return pipe


# =============================================================================
# INTERACTIVE INITIAL GUESS (for Jupyter/IPython)
# =============================================================================


def interactive_initial_guess(pipe: BurstPipeline, model_key: str = "M3"):
    """
    Launch interactive widget for initial guess refinement.

    This function is intended for use in Jupyter/IPython environments.
    After adjusting parameters in the widget, call apply_initial_guess()
    to inject the refined parameters into the pipeline.

    Parameters
    ----------
    pipe : BurstPipeline
        The pipeline instance to refine initial guess for.
    model_key : str, optional
        Model to use for fitting. Default is "M3" (full scattering model).

    Returns
    -------
    InitialGuessWidget
        The widget instance for further interaction.
    """
    from scat_analysis.burstfit_interactive import InitialGuessWidget

    guess_widget = InitialGuessWidget(dataset=pipe.dataset, model_key=model_key)

    # Display interactive interface
    from IPython.display import display

    display(guess_widget.create_widget())

    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("=" * 60)
    print("1. Adjust sliders to roughly match the data")
    print("2. Click 'Auto-Optimize' to refine with L-BFGS-B")
    print("3. Click 'Accept & Continue' when satisfied")
    print("4. Call apply_initial_guess(pipe, guess_widget) to use in pipeline")
    print("=" * 60)

    return guess_widget


def apply_initial_guess(pipe: BurstPipeline, guess_widget) -> FRBParams:
    """
    Apply the refined initial guess from the widget to the pipeline.

    Parameters
    ----------
    pipe : BurstPipeline
        The pipeline instance.
    guess_widget : InitialGuessWidget
        The widget with refined parameters.

    Returns
    -------
    FRBParams
        The optimized parameters that were injected.
    """
    optimized_guess = guess_widget.get_params()

    # Inject the custom initial guess directly into the pipeline
    pipe.seed_single = optimized_guess

    print("[OK] Custom initial guess injected into pipeline")
    print(f"\nParameters that will be used for MCMC:")
    for key, val in optimized_guess.__dict__.items():
        if val is not None:
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

    return optimized_guess


# =============================================================================
# RUN PIPELINE
# =============================================================================


def run_pipeline(
    pipe: BurstPipeline,
    config: dict,
    model_keys: list = None,
) -> dict:
    """
    Run the full MCMC fitting pipeline.

    Parameters
    ----------
    pipe : BurstPipeline
        Configured pipeline instance.
    config : dict
        Configuration dictionary.
    model_keys : list, optional
        List of model keys to fit. Default is ["M3"].

    Returns
    -------
    dict
        Results dictionary containing sampler, best parameters, etc.
    """
    if model_keys is None:
        model_keys = ["M3"]

    results = pipe.run_full(
        model_scan=config.get("model_scan", True),
        model_keys=model_keys,
        diagnostics=config.get("diagnostics", True),
        plot=config.get("plot", True),
        save=True,
        show=False,
    )

    print("\n" + "=" * 60)
    print("PIPELINE RUN SUMMARY")
    print("=" * 60)
    print(f"Best model: {results['best_key']}")
    if "goodness_of_fit" in results:
        print(f"Reduced chi2: {results['goodness_of_fit']['chi2_reduced']:.2f}")
    print("\nBest-fit parameters:")
    for param, value in results["best_params"].items():
        print(f"  {param}: {value:.4g}")

    return results


# =============================================================================
# POST-ANALYSIS: CONVERGENCE AND CORNER PLOTS
# =============================================================================


def check_convergence(
    results: dict,
    config: dict,
    extend_if_needed: bool = False,
) -> bool:
    """
    Check MCMC chain convergence and optionally extend.

    Parameters
    ----------
    results : dict
        Results dictionary from run_pipeline().
    config : dict
        Configuration dictionary.
    extend_if_needed : bool, optional
        If True, run additional MCMC steps if not converged.

    Returns
    -------
    bool
        True if chain is converged, False otherwise.
    """
    sampler = results["sampler"]

    # Detach sampler from pool for serial operation
    sampler.pool = None

    print("Checking MCMC chain convergence...")

    if extend_if_needed:
        max_extra_chunks = config.get("max_chunks", 2)
        chunk_size = config.get("chunk_size", 100)
        chunks_added = 0

        while not quick_chain_check(sampler):
            if chunks_added >= max_extra_chunks:
                print(f"Reached max extra steps ({max_extra_chunks * chunk_size}); proceeding.")
                break
            print(f"Running {chunk_size} more steps for convergence...")
            sampler.run_mcmc(None, chunk_size, progress=True)
            chunks_added += 1

        converged = quick_chain_check(sampler)
        print(f"Chain converged after {chunks_added} additional chunks.")
    else:
        converged = quick_chain_check(sampler)
        print(f"Chain convergence: {'PASSED' if converged else 'NOT CONVERGED'}")

    return converged


def generate_corner_plot(
    results: dict,
    plot_dir: Path,
    burst_name: str,
    show: bool = True,
) -> None:
    """
    Generate and save corner plot from MCMC samples.

    Parameters
    ----------
    results : dict
        Results dictionary from run_pipeline().
    plot_dir : Path
        Directory to save the corner plot.
    burst_name : str
        Name of the burst for filename.
    show : bool, optional
        If True, display the plot. Default is True.
    """
    sampler = results["sampler"]
    best_p = results["best_params"]
    param_names = results["param_names"]

    print("Generating corner plot...")
    final_clean_samples = get_clean_samples(sampler, param_names, verbose=True)

    fig_corner = make_beautiful_corner_wide(
        final_clean_samples,
        param_names,
        best_params=best_p,
        title=f"Posterior for {results['best_key']} ({final_clean_samples.shape[0]} samples)",
    )

    corner_path = plot_dir / f"{burst_name}_scat_corner.pdf"
    fig_corner.savefig(corner_path, dpi=300, bbox_inches="tight")
    print(f"Saved corner plot to: {corner_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_corner)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main(config_path: str = None):
    """
    Main execution function for command-line usage.

    Parameters
    ----------
    config_path : str, optional
        Path to burst configuration YAML file.
    """
    if config_path is None:
        # Default configuration
        config_path = "../configs/bursts/dsa/casey_dsa.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Load configuration
    burst_config = load_burst_config(config_path)

    # Create pipeline
    pipe = create_pipeline(burst_config)

    # Run pipeline
    results = run_pipeline(
        pipe,
        burst_config["config"],
        model_keys=["M3"],
    )

    # Check convergence
    check_convergence(
        results,
        burst_config["config"],
        extend_if_needed=burst_config["config"].get("extend_chain", False),
    )

    # Generate corner plot
    generate_corner_plot(
        results,
        burst_config["plot_dir"],
        burst_config["burst_name"],
        show=False,
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {burst_config['plot_dir']}")
    print(f"  - Diagnostic plot: {burst_config['burst_name']}_scat_fit.pdf")
    print(f"  - Corner plot: {burst_config['burst_name']}_scat_corner.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FRB Scattering Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scattering_analysis.py --config ../configs/bursts/dsa/casey_dsa.yaml
  python scattering_analysis.py --config ../configs/bursts/chime/freya_chime.yaml
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="../configs/bursts/dsa/casey_dsa.yaml",
        help="Path to burst configuration YAML file",
    )

    args = parser.parse_args()
    main(args.config)
