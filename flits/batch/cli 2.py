#!/usr/bin/env python
"""
FLITS Batch Analysis CLI
========================

Command-line interface for batch processing of FRB scattering and scintillation analysis.

Usage:
    flits-batch run /path/to/data --output ./results
    flits-batch generate-configs /path/to/data
    flits-batch joint-analysis ./flits_results.db
    flits-batch summary ./flits_results.db --output ./plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def cmd_run(args):
    """Run batch analysis."""
    from .batch_runner import BatchRunner, BatchConfig
    
    config = BatchConfig(
        data_root=args.data_root,
        output_root=args.output,
        db_path=args.db,
        max_workers=args.workers,
        mcmc_steps=args.steps,
        nproc_per_burst=args.nproc,
        run_scattering=not args.scintillation_only,
        run_scintillation=not args.scattering_only,
        model_scan=not args.no_model_scan,
        diagnostics=args.diagnostics,
        plot=not args.no_plots,
    )
    
    runner = BatchRunner(config)
    
    burst_filter = args.bursts.split(",") if args.bursts else None
    results = runner.run_all(burst_names=burst_filter)
    
    # Export results
    runner.export_results("csv")
    if args.latex:
        runner.export_results("latex")
    
    n_success = sum(1 for r in results if r.success)
    print(f"\n✅ Completed: {n_success}/{len(results)} bursts successful")
    print(f"📊 Results database: {args.db}")
    print(f"📁 Output directory: {args.output}")


def cmd_generate_configs(args):
    """Generate config files from data directory."""
    from .config_generator import ConfigGenerator
    
    generator = ConfigGenerator(args.data_root, args.output)
    manifest = generator.generate_batch_manifest()
    
    print(f"\n✅ Generated configs and manifest: {manifest}")


def cmd_joint_analysis(args):
    """Run joint analysis on existing results."""
    from .results_db import ResultsDatabase
    from .joint_analysis import JointAnalysis
    
    db = ResultsDatabase(args.db_path)
    joint = JointAnalysis(db)
    
    # The new `run_analysis` method orchestrates all steps
    joint.run_analysis(
        output_dir=args.output,
        show_plots=not args.no_show,
    )
    
    print(f"\n✅ Joint analysis complete.")
    if args.output:
        print(f"📊 Results saved to {args.output}")


def cmd_summary(args):
    """Generate summary plots from results database."""
    from .results_db import ResultsDatabase
    from .summary_plots import create_all_summary_plots
    
    db = ResultsDatabase(args.db_path)
    plots = create_all_summary_plots(db, args.output, show=not args.no_show)
    
    print(f"\n✅ Generated {len(plots)} summary plots in {args.output}")
    for p in plots:
        print(f"   - {p.name}")


def cmd_export(args):
    """Export results to various formats."""
    from .results_db import ResultsDatabase
    
    db = ResultsDatabase(args.db_path)
    
    for fmt in args.formats.split(","):
        fmt = fmt.strip()
        output = args.output / f"results.{fmt}" if args.output else None
        
        if fmt == "csv":
            df = db.get_comparison_table()
            if output:
                df.to_csv(output, index=False)
                print(f"✅ Exported CSV: {output}")
            else:
                print(df.to_string())
                
        elif fmt == "latex":
            latex = db.export_latex_table(output)
            if not output:
                print(latex)
                
        elif fmt == "json":
            df = db.get_comparison_table()
            if output:
                df.to_json(output, orient="records", indent=2)
                print(f"✅ Exported JSON: {output}")
            else:
                print(df.to_json(orient="records", indent=2))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FLITS Batch Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # -------------------------------------------------------------------------
    # run: Execute batch analysis
    # -------------------------------------------------------------------------
    run_parser = subparsers.add_parser("run", help="Run batch analysis on all bursts")
    run_parser.add_argument("data_root", type=Path, help="Root data directory (containing chime/, dsa/)")
    run_parser.add_argument("-o", "--output", type=Path, default=Path("batch_output"),
                           help="Output directory for results")
    run_parser.add_argument("--db", type=Path, default=Path("flits_results.db"),
                           help="Results database path")
    run_parser.add_argument("--workers", type=int, default=1,
                           help="Number of parallel workers (not recommended for MCMC)")
    run_parser.add_argument("--steps", type=int, default=10000,
                           help="MCMC steps per burst")
    run_parser.add_argument("--nproc", type=int, default=4,
                           help="Processes per individual burst MCMC")
    run_parser.add_argument("--bursts", type=str, default=None,
                           help="Comma-separated list of burst names to process")
    run_parser.add_argument("--scattering-only", action="store_true",
                           help="Run only scattering analysis")
    run_parser.add_argument("--scintillation-only", action="store_true",
                           help="Run only scintillation analysis")
    run_parser.add_argument("--no-model-scan", action="store_true",
                           help="Skip model comparison, fit M3 directly")
    run_parser.add_argument("--diagnostics", action="store_true",
                           help="Run detailed diagnostics (slower)")
    run_parser.add_argument("--no-plots", action="store_true",
                           help="Skip plot generation")
    run_parser.add_argument("--latex", action="store_true",
                           help="Also export LaTeX table")
    run_parser.set_defaults(func=cmd_run)
    
    # -------------------------------------------------------------------------
    # generate-configs: Create config files from data
    # -------------------------------------------------------------------------
    config_parser = subparsers.add_parser("generate-configs", 
                                          help="Generate config files from data directory")
    config_parser.add_argument("data_root", type=Path, help="Root data directory")
    config_parser.add_argument("-o", "--output", type=Path, default=None,
                              help="Output directory for configs")
    config_parser.set_defaults(func=cmd_generate_configs)
    
    # -------------------------------------------------------------------------
    # joint-analysis: Run joint τ-Δν analysis
    # -------------------------------------------------------------------------
    joint_parser = subparsers.add_parser("joint-analysis",
                                         help="Run joint scattering-scintillation analysis")
    joint_parser.add_argument("db_path", type=Path, help="Results database path")
    joint_parser.add_argument("-o", "--output", type=Path, default=None,
                             help="Output directory for plots and report")
    joint_parser.add_argument("--no-show", action="store_true",
                             help="Don't display plots interactively")
    joint_parser.set_defaults(func=cmd_joint_analysis)
    
    # -------------------------------------------------------------------------
    # summary: Generate summary plots
    # -------------------------------------------------------------------------
    summary_parser = subparsers.add_parser("summary",
                                           help="Generate summary plots from results")
    summary_parser.add_argument("db_path", type=Path, help="Results database path")
    summary_parser.add_argument("-o", "--output", type=Path, default=Path("summary_plots"),
                               help="Output directory for plots")
    summary_parser.add_argument("--no-show", action="store_true",
                               help="Don't display plots interactively")
    summary_parser.set_defaults(func=cmd_summary)
    
    # -------------------------------------------------------------------------
    # export: Export results
    # -------------------------------------------------------------------------
    export_parser = subparsers.add_parser("export",
                                          help="Export results to various formats")
    export_parser.add_argument("db_path", type=Path, help="Results database path")
    export_parser.add_argument("-f", "--formats", type=str, default="csv",
                              help="Comma-separated formats: csv, latex, json")
    export_parser.add_argument("-o", "--output", type=Path, default=None,
                              help="Output directory (default: print to stdout)")
    export_parser.set_defaults(func=cmd_export)
    
    # -------------------------------------------------------------------------
    # Parse and execute
    # -------------------------------------------------------------------------
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    try:
        args.func(args)
    except Exception as e:
        log.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

