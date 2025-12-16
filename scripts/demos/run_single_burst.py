#!/usr/bin/env python
"""
Run scattering analysis for a single burst and store results in the shared database.

Usage:
    python run_single_burst.py casey dsa
    python run_single_burst.py casey chime
    python run_single_burst.py hamilton dsa --steps 15000
    python run_single_burst.py --list  # Show all available bursts

All results are stored in flits_results.db for comparison across bursts.
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add scattering to path
sys.path.insert(0, str(Path(__file__).parent / "scattering"))

from scat_analysis.burstfit_pipeline import BurstPipeline
from scat_analysis.config_utils import load_config
from flits.batch import ResultsDatabase, ScatteringResult
from flits.batch.config_generator import ConfigGenerator


def list_available_bursts():
    """List all available bursts in the data directory."""
    gen = ConfigGenerator(Path("data"))
    bursts = gen.discover_bursts()
    
    print("\n" + "="*60)
    print("AVAILABLE BURSTS")
    print("="*60)
    print(f"\n{'Burst':<15} {'Telescopes':<15} {'DM (CHIME)':<12} {'DM (DSA)':<12}")
    print("-"*55)
    
    for name in sorted(bursts.keys()):
        infos = bursts[name]
        telescopes = ", ".join(i.telescope.upper() for i in infos)
        dm_chime = next((f"{i.dm:.4f}" for i in infos if i.telescope == "chime"), "—")
        dm_dsa = next((f"{i.dm:.4f}" for i in infos if i.telescope == "dsa"), "—")
        print(f"{name:<15} {telescopes:<15} {dm_chime:<12} {dm_dsa:<12}")
    
    print(f"\nTotal: {len(bursts)} bursts")
    print("="*60 + "\n")


def run_single_burst(
    burst_name: str,
    telescope: str,
    steps: int = 10000,
    nproc: int = 8,
    db_path: str = "flits_results.db",
    output_base: str = "output",
    model_scan: bool = True,
    diagnostics: bool = True,
    show_plots: bool = False,
):
    """
    Run scattering analysis for a single burst and store in database.
    
    Args:
        burst_name: Name of the burst (e.g., "casey", "hamilton")
        telescope: "dsa" or "chime"
        steps: Number of MCMC steps
        nproc: Number of parallel processes
        db_path: Path to shared results database
        output_base: Base directory for output files
        model_scan: Whether to compare M0-M3 models
        diagnostics: Whether to run sub-band diagnostics
        show_plots: Whether to display plots interactively
    """
    print("="*60)
    print(f"SCATTERING ANALYSIS: {burst_name}/{telescope.upper()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Load configuration
    config_path = f"scattering/configs/bursts/{telescope}/{burst_name}_{telescope}.yaml"
    
    if not Path(config_path).exists():
        print(f"\n❌ Error: Config not found: {config_path}")
        print(f"   Run 'python run_single_burst.py --list' to see available bursts")
        return None
    
    print(f"\n📄 Config: {config_path}")
    config = load_config(config_path)
    
    print(f"   Data: {config.path}")
    print(f"   Telescope: {config.telescope.name}")
    print(f"   Downsampling: {config.pipeline.f_factor}x freq, {config.pipeline.t_factor}x time")
    
    # Create output directory
    output_dir = Path(output_base) / f"{burst_name}_{telescope}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Output: {output_dir}")
    
    # Create pipeline
    print(f"\n🚀 Creating pipeline (nproc={nproc}, steps={steps})...")
    pipe = BurstPipeline(
        inpath=config.path,
        outpath=output_dir,
        name=burst_name,
        dm_init=config.dm_init,
        telescope=config.telescope,
        sampler=config.sampler,
        f_factor=config.pipeline.f_factor,
        t_factor=config.pipeline.t_factor,
        steps=steps,
        nproc=nproc,
    )
    
    # Run analysis
    print(f"\n⏳ Running MCMC ({steps} steps)...")
    print(f"   Model scan: {model_scan}")
    print(f"   Diagnostics: {diagnostics}")
    print("   This may take 5-30 minutes depending on data size...\n")
    
    results = pipe.run_full(
        model_scan=model_scan,
        diagnostics=diagnostics,
        plot=True,
        show=show_plots,
        save=True,
    )
    
    # Extract key results
    best_p = results['best_params']
    gof = results['goodness_of_fit']
    
    print(f"\n✅ Analysis complete!")
    print(f"   Best model: {results['best_key']}")
    print(f"   χ²/dof: {gof['chi2_reduced']:.2f}")
    
    print(f"\n📊 Best-fit parameters:")
    print(f"   τ_1GHz = {best_p.tau_1ghz:.4f} ms")
    print(f"   α = {best_p.alpha:.2f}")
    print(f"   t0 = {best_p.t0:.2f} ms")
    print(f"   ζ (width) = {best_p.zeta:.4f} ms")
    print(f"   γ (spectral) = {best_p.gamma:.2f}")
    
    # Check convergence
    sampler = results.get('sampler')
    rhat_max = None
    if sampler is not None:
        try:
            from scat_analysis.burstfit import gelman_rubin
            chain = sampler.get_chain(flat=False)
            rhat = gelman_rubin(chain)
            rhat_max = float(max(rhat))
            converged = rhat_max < 1.1
            status = "✓ CONVERGED" if converged else "⚠ NOT CONVERGED"
            print(f"\n🔍 Convergence: R̂_max = {rhat_max:.3f} {status}")
        except Exception as e:
            print(f"\n⚠ Could not compute R̂: {e}")
    
    # Determine quality flag
    chi2 = gof['chi2_reduced']
    if chi2 < 0.5:
        quality = "suspicious"  # Overfitting
    elif chi2 > 5.0:
        quality = "poor"
    elif rhat_max and rhat_max > 1.1:
        quality = "marginal"  # Not converged
    else:
        quality = "good"
    
    # Store in database
    print(f"\n💾 Saving to database: {db_path}")
    db = ResultsDatabase(db_path)
    
    result = ScatteringResult.from_pipeline_results(
        burst_name=burst_name,
        telescope=telescope,
        results=results,
        config_path=config_path,
        data_path=str(config.path),
    )
    result.quality_flag = quality
    result.gelman_rubin_max = rhat_max
    
    db.add_scattering_result(result)
    
    # Show database summary
    all_results = db.get_scattering_results()
    print(f"   Database now contains {len(all_results)} result(s)")
    
    db.close()
    
    # List output files
    print(f"\n📁 Output files:")
    for f in sorted(output_dir.glob("*.pdf")):
        print(f"   - {f.name}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60 + "\n")
    
    return results


def show_database_summary(db_path: str = "flits_results.db"):
    """Show summary of all results in the database."""
    db = ResultsDatabase(db_path)
    
    print("\n" + "="*70)
    print("DATABASE SUMMARY")
    print("="*70)
    
    results = db.get_scattering_results()
    
    if not results:
        print("\n   (No results yet)")
    else:
        print(f"\n{'Burst':<12} {'Tel':<6} {'Model':<4} {'τ_1GHz (ms)':<14} {'α':<10} {'χ²/dof':<8} {'Quality':<10}")
        print("-"*70)
        
        for r in sorted(results, key=lambda x: (x.burst_name, x.telescope)):
            tau = f"{r.tau_1ghz:.4f}" if r.tau_1ghz else "—"
            tau_err = f"±{r.tau_1ghz_err:.4f}" if r.tau_1ghz_err else ""
            alpha = f"{r.alpha:.2f}" if r.alpha else "—"
            chi2 = f"{r.chi2_reduced:.2f}" if r.chi2_reduced else "—"
            
            print(f"{r.burst_name:<12} {r.telescope:<6} {r.best_model:<4} {tau}{tau_err:<14} {alpha:<10} {chi2:<8} {r.quality_flag:<10}")
    
    print("="*70 + "\n")
    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run scattering analysis for a single burst",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("burst_name", nargs="?", help="Burst name (e.g., casey, hamilton)")
    parser.add_argument("telescope", nargs="?", choices=["dsa", "chime"], 
                       help="Telescope: dsa or chime")
    
    parser.add_argument("--steps", type=int, default=10000, 
                       help="MCMC steps (default: 10000)")
    parser.add_argument("--nproc", type=int, default=8,
                       help="Parallel processes (default: 8)")
    parser.add_argument("--db", type=str, default="flits_results.db",
                       help="Database path (default: flits_results.db)")
    parser.add_argument("--output", type=str, default="output",
                       help="Output base directory (default: output)")
    
    parser.add_argument("--no-model-scan", action="store_true",
                       help="Skip model comparison, fit M3 directly")
    parser.add_argument("--no-diagnostics", action="store_true",
                       help="Skip sub-band diagnostics")
    parser.add_argument("--show", action="store_true",
                       help="Display plots interactively")
    
    parser.add_argument("--list", action="store_true",
                       help="List available bursts")
    parser.add_argument("--summary", action="store_true",
                       help="Show database summary")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list:
        list_available_bursts()
        return
    
    if args.summary:
        show_database_summary(args.db)
        return
    
    # Require burst_name and telescope for analysis
    if not args.burst_name or not args.telescope:
        parser.print_help()
        print("\n❌ Error: burst_name and telescope are required")
        print("   Example: python run_single_burst.py casey dsa")
        print("   Run with --list to see available bursts")
        return
    
    # Run analysis
    run_single_burst(
        burst_name=args.burst_name.lower(),
        telescope=args.telescope.lower(),
        steps=args.steps,
        nproc=args.nproc,
        db_path=args.db,
        output_base=args.output,
        model_scan=not args.no_model_scan,
        diagnostics=not args.no_diagnostics,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()

