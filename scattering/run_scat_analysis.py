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
    # New modeling controls
    parser.add_argument("--alpha-fixed", type=float, default=None)
    parser.add_argument("--alpha-mu", type=float, default=4.4)
    parser.add_argument("--alpha-sigma", type=float, default=0.6)
    parser.add_argument("--delta-dm-sigma", type=float, default=0.1)
    parser.add_argument("--likelihood", choices=["gaussian","studentt"], default="gaussian")
    parser.add_argument("--studentt-nu", type=float, default=5.0)
    parser.add_argument("--no-logspace", dest='sample_log_params', action='store_false')
    parser.add_argument("--ncomp", type=int, default=1, help="Number of Gaussian components (shared PBF)")
    parser.add_argument("--auto-components", action='store_true', help="Greedy BIC-based component selection")
    # Earmarks / placeholders
    parser.add_argument("--anisotropy-enabled", action='store_true')
    parser.add_argument("--anisotropy-axial-ratio", type=float, default=1.0)
    parser.add_argument("--baseline-order", type=int, default=0)
    parser.add_argument("--correlated-resid", action='store_true')
    parser.add_argument("--sampler", choices=["emcee","nested"], default="emcee")
    parser.add_argument("--init-guess", type=str, default=None, help="Path to JSON seed for initial guess")
    parser.add_argument("--walker-width-frac", type=float, default=0.01, help="Fraction of prior span for walker init width")
    
    args = parser.parse_args()

    print(f"--- Loading configuration from: {args.config_path} ---")
    config = load_config(args.config_path)

    # --- Resolve config helper files (telescopes.yaml, sampler.yaml) ---
    # Prefer CLI overrides; otherwise, search sensible locations relative to the
    # run config path and this script's directory.
    def _resolve_cfg(base_dir: Path, filename: str) -> Path:
        candidates = [
            base_dir / filename,
            base_dir.parent / filename,
            base_dir.parent.parent / filename,
            Path(__file__).parent / "configs" / filename,
        ]
        for c in candidates:
            if c.exists():
                return c
        # Fallback: return the first candidate even if missing; downstream code may handle
        return candidates[-1]

    config_base_dir = args.config_path.parent
    telcfg_path = Path(args.telcfg) if args.telcfg else _resolve_cfg(config_base_dir, "telescopes.yaml")
    sampcfg_path = Path(args.sampcfg) if args.sampcfg else _resolve_cfg(config_base_dir, "sampler.yaml")

    # Add these paths to the config dictionary with names expected by BurstDataset
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
    dm_in_cfg = 'dm_init' in config
    dm_init = config.pop('dm_init', None)
    outpath = Path(config.pop('outpath', data_path.parent))
    frb_name = config.pop('frb', data_path.stem)
    
    print(f"\n--- Starting analysis for: {data_path.name} ---")

    # Optional: auto-populate dm_init from burst_props.csv if not provided
    try:
        # Resolve CSV path within the repo
        flits_root = Path(__file__).resolve().parents[1]
        csv_candidates = [
            flits_root / 'scintillation' / 'burst_props.csv',
            flits_root.parent / 'scintillation' / 'burst_props.csv',
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path is not None:
            with csv_path.open('r', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                def _norm_key(k):
                    return ''.join(ch for ch in k.lower() if ch.isalnum())
                rows = [{_norm_key(k): v for k, v in r.items()} for r in reader]
                # Find row by name
                key_name = frb_name.strip().lower()
                hit = next((r for r in rows if r.get('names', '').strip().lower() == key_name), None)
                if hit is not None and not dm_in_cfg and dm_init is None:
                    val = hit.get('dmopt') or hit.get('dmheimdall')
                    if val:
                        # Strip non-numeric
                        v = ''.join(ch for ch in val if (ch.isdigit() or ch in '.-'))
                        try:
                            dm_init = float(v)
                            print(f"  -> Using DM from burst_props.csv for '{frb_name}': {dm_init}")
                        except ValueError:
                            pass
    except Exception as e:
        # Non-fatal; continue without CSV metadata
        print(f"(warn) Unable to read burst_props.csv: {e}")

    if dm_init is None:
        dm_init = 0.0

    pipe = BurstPipeline(inpath=data_path, outpath=outpath, name=frb_name, dm_init=dm_init, **config)
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
