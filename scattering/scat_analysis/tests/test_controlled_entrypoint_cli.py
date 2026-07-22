from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DRIVER_DIR = REPO / "analysis" / "scattering-refit-2026-06"


def _run(script: str, runs: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update({"FLITS_REPO": str(REPO), "FLITS_RUNS": str(runs)})
    return subprocess.run(
        [sys.executable, str(DRIVER_DIR / script), "test", "20", "1", *extra],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_legacy_entrypoint_does_not_require_controlled_flags(tmp_path: Path) -> None:
    result = _run("run_joint_fit.py", tmp_path)

    assert result.returncode != 0
    assert "missing config" in result.stderr
    assert "the following arguments are required" not in result.stderr


def test_controlled_entrypoint_requires_seed_contract_and_receipt(
    tmp_path: Path,
) -> None:
    result = _run("run_controlled_joint_fit.py", tmp_path)

    assert result.returncode == 2
    assert "--seed" in result.stderr
    assert "--contract" in result.stderr
    assert "--receipt" in result.stderr


def test_controlled_entrypoint_requires_fixed_gain_prior(tmp_path: Path) -> None:
    result = _run(
        "run_controlled_joint_fit.py",
        tmp_path,
        "--seed",
        "20260722",
        "--contract",
        str(tmp_path / "contract.json"),
        "--receipt",
        str(tmp_path / "receipt.json"),
    )

    assert result.returncode == 2
    assert "require a fixed --gain-s2" in result.stderr


def test_controlled_entrypoint_rejects_fixed_dispersion_measure(tmp_path: Path) -> None:
    result = _run(
        "run_controlled_joint_fit.py",
        tmp_path,
        "--seed",
        "20260722",
        "--contract",
        str(tmp_path / "contract.json"),
        "--receipt",
        str(tmp_path / "receipt.json"),
        "--gain-s2",
        "100",
        "--fixed-delta-dm-C",
        "0",
    )

    assert result.returncode == 2
    assert "do not accept fixed residual dispersion measure" in result.stderr
