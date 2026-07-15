#!/usr/bin/env python3
"""Drive the blinded C1 calibration matrix with bounded local parallelism.

Runs every (modulation, width) cell of the frozen grid through
``validate_freya_c1.py calibrate`` (checkpoint-per-cell, idempotent), then
the null campaign and the aggregate verdict. Never calls ``unblind``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
VALIDATOR = HERE / "validate_freya_c1.py"
MODULATION_INDICES = (0.10, 0.15, 0.17, 0.20, 0.30, 1.00)
WIDTH_CHANNELS = (3.0, 6.0, 10.0, 16.0)


def _run(arguments: list[str]) -> tuple[list[str], int, str]:
    process = subprocess.run(
        [sys.executable, str(VALIDATOR), *arguments],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    tail = (process.stdout + process.stderr).strip().splitlines()
    return arguments, process.returncode, tail[-1] if tail else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()

    start = time.time()
    cells = [
        ["calibrate", "--modulation", f"{m}", "--width", f"{w}", "--trials", str(args.trials)]
        for m in MODULATION_INDICES
        for w in WIDTH_CHANNELS
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, cell): cell for cell in cells}
        for future in as_completed(futures):
            arguments, code, line = future.result()
            print(f"[{time.time() - start:7.1f}s] rc={code} {' '.join(arguments[1:])}: {line}")
            if code != 0:
                failures.append(arguments)
    if failures:
        print(json.dumps({"failed_cells": [" ".join(item) for item in failures]}))
        return 1

    for stage in ("nulls", "aggregate"):
        arguments, code, line = _run([stage])
        print(f"[{time.time() - start:7.1f}s] rc={code} {stage}: {line}")
        if stage == "aggregate":
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
