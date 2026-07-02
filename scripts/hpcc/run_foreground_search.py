#!/usr/bin/env python3
"""Resume-safe foreground galaxy search for all configured sightlines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from galaxies.foreground import search


def main() -> None:
    ap = argparse.ArgumentParser(description="Run galaxies.foreground.search with per-sightline resume.")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/repro-foreground-search-hpcc"),
        help="Directory for per-sightline CSV outputs",
    )
    ap.add_argument("--impact-kpc", type=float, default=100.0)
    ap.add_argument("--no-build-unified", action="store_true")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional sightline nicknames to run (default: all TARGETS)",
    )
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "done.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
    summary_parts: list[pd.DataFrame] = []

    targets = list(search.TARGETS)
    if args.only:
        wanted = {n.lower() for n in args.only}
        targets = [t for t in targets if t[0].lower() in wanted]

    for target in targets:
        name = target[0]
        if name in done:
            print(f"skip {name} (already in {done_path})")
            continue
        print(f"run {name}", flush=True)
        search.TARGETS = [target]
        search.run_search(
            output_dir=str(out),
            impact_kpc=args.impact_kpc,
            build_unified=not args.no_build_unified,
        )
        summary_path = out / "search_summary.csv"
        if summary_path.exists():
            summary_parts.append(pd.read_csv(summary_path))
        with done_path.open("a") as fh:
            fh.write(name + "\n")

    if summary_parts:
        pd.concat(summary_parts, ignore_index=True).to_csv(out / "_latest_summary_chunk.csv", index=False)
    print(f"finished; outputs under {out}", flush=True)


if __name__ == "__main__":
    main()
