"""CLI: build foreground data artifacts (registry, tau catalog, attribution matrix)."""

from __future__ import annotations

import argparse
from pathlib import Path

from galaxies.foreground.attribution_matrix import write_attribution_matrix
from galaxies.foreground.census_registry import write_intervening_census_registry
from galaxies.foreground.tau_consistency import write_tau_consistency_catalog


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build galaxies/foreground/data artifacts.")
    ap.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Path to scratch/codetection (default: pipeline/scratch/codetection)",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Output directory (default: galaxies/foreground/data)",
    )
    args = ap.parse_args(argv)

    data_dir = args.data_dir
    registry_path = write_intervening_census_registry(
        path=(data_dir / "intervening_census_registry.csv") if data_dir else None,
        scratch_dir=args.scratch_dir,
    )
    tau_path = write_tau_consistency_catalog(
        path=(data_dir / "tau_consistency_catalog.csv") if data_dir else None,
    )
    matrix_path = write_attribution_matrix(
        path=(data_dir / "sightline_attribution_matrix.csv") if data_dir else None,
    )
    print(f"Wrote {registry_path}")
    print(f"Wrote {tau_path}")
    print(f"Wrote {matrix_path}")


if __name__ == "__main__":
    main()
