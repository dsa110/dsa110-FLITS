"""
batch_runner.py
===============

Orchestrate multi-burst analysis with parallel execution and progress tracking.
"""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config_generator import BurstInfo, ConfigGenerator
from .results_db import ResultsDatabase, ScatteringResult, ScintillationResult

log = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    data_root: Path
    output_root: Path
    db_path: Path
    max_workers: int = 4
    telescopes: list[str] = field(default_factory=lambda: ["chime", "dsa"])
    run_scattering: bool = True
    run_scintillation: bool = True
    # Pipeline options
    mcmc_steps: int = 10000
    nproc_per_burst: int = 4  # Processes per individual burst
    model_scan: bool = True
    diagnostics: bool = False  # Disable for speed in batch mode
    plot: bool = True
    save_samplers: bool = True
    scint_config_dir: Path | None = None  # default: <repo>/configs/batch


@dataclass
class BatchResult:
    """Result from processing a single burst."""

    burst_name: str
    telescope: str
    success: bool
    error_message: str = ""
    scattering_result: ScatteringResult | None = None
    scintillation_result: ScintillationResult | None = None
    duration_seconds: float = 0.0


def discover_scint_configs(base_dir, telescopes) -> dict[str, dict[str, Path]]:
    """Map {burst: {telescope: existing scint config path}}.

    Scint configs are hand-tuned (RFI / manual burst windows can't be derived
    without inspecting the data), so they are resolved from
    <base_dir>/<telescope>/<burst>_<telescope>.yaml rather than generated.
    """
    base = Path(base_dir)
    found: dict[str, dict[str, Path]] = {}
    for telescope in telescopes:
        for cfg in sorted((base / telescope).glob(f"*_{telescope}.yaml")):
            burst = cfg.stem[: -len(f"_{telescope}")]
            found.setdefault(burst, {})[telescope] = cfg
    return found


def _run_scattering_analysis(
    config_path: Path,
    output_dir: Path,
    burst_name: str,
    telescope: str,
) -> ScatteringResult | None:
    """Run scattering pipeline for a single burst (isolated function for multiprocessing)."""

    # Ensure imports work
    flits_root = config_path.parent.parent.parent.parent
    if str(flits_root) not in sys.path:
        sys.path.insert(0, str(flits_root))

    from flits.scattering.scat_analysis.config_utils import load_config
    from flits.scattering.scat_analysis.pipeline import BurstPipeline

    try:
        config = load_config(str(config_path))

        pipe = BurstPipeline(
            inpath=config.path,
            outpath=output_dir,
            name=burst_name,
            dm_init=config.dm_init,
            telescope=config.telescope,
            sampler=config.sampler,
            f_factor=config.pipeline.f_factor,
            t_factor=config.pipeline.t_factor,
            steps=config.pipeline.steps,
            nproc=4,
        )

        results = pipe.run_full(
            model_scan=True,
            diagnostics=False,
            plot=True,
            show=False,
            save=True,
        )

        return ScatteringResult.from_pipeline_results(
            burst_name=burst_name,
            telescope=telescope,
            results=results,
            config_path=str(config_path),
            data_path=str(config.path),
        )

    except Exception as e:
        log.error(f"Scattering analysis failed for {burst_name}/{telescope}: {e}")
        traceback.print_exc()
        return None


def _run_scintillation_analysis(
    config_path: Path,
    burst_name: str,
    telescope: str,
) -> ScintillationResult | None:
    """Run scintillation pipeline for a single burst."""

    flits_root = config_path.parent.parent.parent.parent
    if str(flits_root) not in sys.path:
        sys.path.insert(0, str(flits_root))

    from scintillation.scint_analysis import config as scint_config
    from scintillation.scint_analysis import pipeline

    try:
        loaded_config = scint_config.load_config(str(config_path))
        scint_pipeline = pipeline.ScintillationAnalysis(loaded_config)
        scint_pipeline.run()

        if scint_pipeline.final_results:
            return ScintillationResult.from_pipeline_results(
                burst_name=burst_name,
                telescope=telescope,
                final_results=scint_pipeline.final_results,
                acf_results=scint_pipeline.acf_results or {},
                config_path=str(config_path),
                data_path=loaded_config.get("input_data_path", ""),
            )
        return None

    except Exception as e:
        log.error(f"Scintillation analysis failed for {burst_name}/{telescope}: {e}")
        traceback.print_exc()
        return None


class BatchRunner:
    """Orchestrate batch analysis across multiple FRBs."""

    def __init__(self, config: BatchConfig):
        """
        Initialize batch runner.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.db = ResultsDatabase(config.db_path)
        self.config_generator = ConfigGenerator(config.data_root)

        # Create output directory
        self.output_root = config.output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Track results
        self.results: list[BatchResult] = []

    def discover_bursts(self) -> dict[str, list[BurstInfo]]:
        """Discover all available bursts."""
        return self.config_generator.discover_bursts(self.config.telescopes)

    def _process_single_burst(
        self,
        burst_name: str,
        burst_info: BurstInfo,
        scat_config_path: Path | None,
        scint_config_path: Path | None,
    ) -> BatchResult:
        """Process a single burst (both pipelines)."""
        import time

        start_time = time.time()

        telescope = burst_info.telescope
        burst_output_dir = self.output_root / burst_name / telescope
        burst_output_dir.mkdir(parents=True, exist_ok=True)

        result = BatchResult(
            burst_name=burst_name,
            telescope=telescope,
            success=True,
        )

        # Run scattering analysis
        if self.config.run_scattering and scat_config_path:
            log.info(f"Running scattering analysis: {burst_name}/{telescope}")
            try:
                scat_result = _run_scattering_analysis(
                    scat_config_path, burst_output_dir, burst_name, telescope
                )
                if scat_result:
                    result.scattering_result = scat_result
                    self.db.add_scattering_result(scat_result)
                else:
                    result.success = False
                    result.error_message += "Scattering analysis returned no results. "
            except Exception as e:
                result.success = False
                result.error_message += f"Scattering error: {e}. "

        # Run scintillation analysis
        if self.config.run_scintillation and scint_config_path:
            log.info(f"Running scintillation analysis: {burst_name}/{telescope}")
            try:
                scint_result = _run_scintillation_analysis(scint_config_path, burst_name, telescope)
                if scint_result:
                    result.scintillation_result = scint_result
                    self.db.add_scintillation_result(scint_result)
                else:
                    result.success = False
                    result.error_message += "Scintillation analysis returned no results. "
            except Exception as e:
                result.success = False
                result.error_message += f"Scintillation error: {e}. "

        result.duration_seconds = time.time() - start_time
        return result

    def _discover_scint_configs(self) -> dict[str, dict[str, Path]]:
        base = self.config.scint_config_dir or (
            Path(__file__).resolve().parents[2] / "configs" / "batch"
        )
        return discover_scint_configs(base, self.config.telescopes)

    def run_all(
        self,
        burst_names: list[str] | None = None,
        parallel: bool = False,
        progress_callback: Callable[[BatchResult], None] | None = None,
    ) -> list[BatchResult]:
        """
        Run analysis on all (or specified) bursts.

        Args:
            burst_names: Optional list of burst names to process (default: all)
            parallel: Whether to run bursts in parallel (not recommended for MCMC)
            progress_callback: Optional callback called after each burst completes

        Returns:
            List of BatchResult objects
        """
        bursts = self.discover_bursts()

        if burst_names:
            bursts = {k: v for k, v in bursts.items() if k in burst_names}

        log.info(f"Starting batch analysis of {len(bursts)} bursts")

        # Generate configs for all bursts
        scat_configs = self.config_generator.generate_all_configs(
            self.config.telescopes,
            steps=self.config.mcmc_steps,
            nproc=self.config.nproc_per_burst,
            model_scan=self.config.model_scan,
            diagnostics=self.config.diagnostics,
            plot=self.config.plot,
        )

        # Scint configs are hand-tuned (RFI / burst windows need the data), so
        # resolve existing ones rather than generate.
        scint_configs = self._discover_scint_configs() if self.config.run_scintillation else {}

        self.results = []
        total = sum(len(infos) for infos in bursts.values())
        completed = 0

        for burst_name, burst_infos in bursts.items():
            for burst_info in burst_infos:
                telescope = burst_info.telescope

                # Get config paths
                scat_config = scat_configs.get(burst_name, {}).get(telescope)
                scint_config = scint_configs.get(burst_name, {}).get(telescope)

                log.info(f"[{completed + 1}/{total}] Processing {burst_name}/{telescope}")

                result = self._process_single_burst(
                    burst_name, burst_info, scat_config, scint_config
                )
                self.results.append(result)

                if progress_callback:
                    progress_callback(result)

                completed += 1

                status = "✓" if result.success else "✗"
                log.info(f"  {status} Completed in {result.duration_seconds:.1f}s")
                if not result.success:
                    log.warning(f"  Error: {result.error_message}")

        # Generate summary
        self._generate_summary()

        return self.results

    def _generate_summary(self):
        """Generate batch processing summary."""
        summary_path = self.output_root / "batch_summary.txt"

        n_success = sum(1 for r in self.results if r.success)
        n_failed = len(self.results) - n_success
        total_time = sum(r.duration_seconds for r in self.results)

        lines = [
            "=" * 60,
            "FLITS BATCH ANALYSIS SUMMARY",
            f"Timestamp: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            f"Total bursts processed: {len(self.results)}",
            f"  Successful: {n_success}",
            f"  Failed: {n_failed}",
            f"Total processing time: {total_time / 60:.1f} minutes",
            "",
            "Results by burst:",
            "-" * 40,
        ]

        for result in self.results:
            status = "✓" if result.success else "✗"
            lines.append(
                f"  {status} {result.burst_name}/{result.telescope} ({result.duration_seconds:.1f}s)"
            )
            if not result.success:
                lines.append(f"      Error: {result.error_message}")

        lines.extend(
            [
                "",
                "=" * 60,
                f"Results database: {self.config.db_path}",
                f"Output directory: {self.output_root}",
                "=" * 60,
            ]
        )

        summary_text = "\n".join(lines)

        with open(summary_path, "w") as f:
            f.write(summary_text)

        print(summary_text)
        log.info(f"Summary written to {summary_path}")

    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table from results database."""
        return self.db.get_comparison_table()

    def export_results(
        self,
        format: str = "csv",
        output_path: Path | None = None,
    ) -> Path:
        """
        Export results to file.

        Args:
            format: Output format ("csv", "latex", "json")
            output_path: Output file path (default: output_root/results.{format})
        """
        if output_path is None:
            output_path = self.output_root / f"results.{format}"

        df = self.get_comparison_table()

        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "latex":
            self.db.export_latex_table(output_path)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        log.info(f"Exported results to {output_path}")
        return output_path


def main():
    """CLI entry point for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(description="FLITS batch analysis")
    parser.add_argument("data_root", type=Path, help="Root data directory")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("batch_output"), help="Output directory"
    )
    parser.add_argument(
        "--db", type=Path, default=Path("flits_results.db"), help="Results database path"
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--steps", type=int, default=10000, help="MCMC steps per burst")
    parser.add_argument("--bursts", nargs="+", help="Specific burst names to process")
    parser.add_argument(
        "--scattering-only", action="store_true", help="Run only scattering analysis"
    )
    parser.add_argument(
        "--scintillation-only", action="store_true", help="Run only scintillation analysis"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = BatchConfig(
        data_root=args.data_root,
        output_root=args.output,
        db_path=args.db,
        max_workers=args.workers,
        mcmc_steps=args.steps,
        run_scattering=not args.scintillation_only,
        run_scintillation=not args.scattering_only,
    )

    runner = BatchRunner(config)
    results = runner.run_all(burst_names=args.bursts)

    # Export results
    runner.export_results("csv")
    runner.export_results("latex")

    # Print final summary
    n_success = sum(1 for r in results if r.success)
    print(f"\n✅ Completed: {n_success}/{len(results)} bursts successful")


if __name__ == "__main__":
    main()
