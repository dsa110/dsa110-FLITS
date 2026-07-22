"""Tests for tau_consistency JSON loading and refit runner errors."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from galaxies.foreground.run_tau_consistency_refits import run_burst
from galaxies.foreground.tau_consistency import (
    _joint_fit_scalar,
    _posterior_median,
    find_allexp_joint_json,
    load_allexp_joint_tau_for_budget,
    load_citable_budget_nicknames,
    load_joint_free_alpha,
    tau_consistency_from_refit,
)

# Contract: analysis/scattering-dm-locked-2026-07-14/results/fit_adjudication.csv,
# rows with adjudication == accepted_physical (also pinned by
# test_fit_adjudication.py::test_pbf_roster_is_exactly_the_physically_accepted_subset).
FIT_ADJUDICATION_CSV = (
    Path(__file__).resolve().parents[2]
    / "analysis"
    / "scattering-dm-locked-2026-07-14"
    / "results"
    / "fit_adjudication.csv"
)

JULY_ACCEPTED_VARIANTS = {
    "whitney": "C2D3",
    "oran": "C2D1",
    "isha": "C2D1",
    "phineas": "C3D3",
    "freya": "C1D1",
    "johndoeii": "C2D2",
    "mahi": "C1D2",
}

NON_ACCEPTED_BURSTS = ["casey", "zach", "chromatica"]


def _adjudication_rows() -> list[dict[str, str]]:
    with FIT_ADJUDICATION_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _fixed_delta_dm(burst: str) -> tuple[float, float]:
    for row in _adjudication_rows():
        if row["burst"].lower() == burst.lower():
            return float(row["fixed_delta_dm_C"]), float(row["fixed_delta_dm_D"])
    raise KeyError(burst)


def _cmd_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_posterior_median_dict_and_scalar():
    assert _posterior_median({"median": 0.5}) == 0.5
    assert _posterior_median(0.061) == 0.061
    assert np.isnan(_posterior_median(None))


def test_joint_fit_scalar_ppc_payload():
    payload = {"tau_1ghz": 0.06086799947757, "alpha": 2.396}
    assert _joint_fit_scalar(payload, "tau_1ghz") == 0.06086799947757
    assert _joint_fit_scalar(payload, "alpha") == 2.396


def test_tau_consistency_from_refit_scalar():
    row = tau_consistency_from_refit({"tau_1ghz": 0.1, "alpha": 4.0})
    assert row["tau_consistency_1ghz_ms"] == 0.1
    assert row["refit_status"] == "alpha4_joint_complete"


def test_load_allexp_joint_tau_for_budget_casey():
    row = load_allexp_joint_tau_for_budget("casey")
    assert row is not None
    assert row["source"] == "allexp_joint"
    assert row["tau"] == pytest.approx(0.018589939827674748, rel=1e-4)
    assert row["quality_flag"] in ("PASS", "MARGINAL")
    assert row["err_minus"] > 0


def test_load_allexp_joint_tau_excluded_burst():
    assert load_allexp_joint_tau_for_budget("chromatica") is None


def test_load_allexp_joint_tau_johndoeii_promoted_c2d2():
    row = load_allexp_joint_tau_for_budget("johndoeII")
    assert row is not None
    assert row["source"] == "allexp_joint"
    assert row["quality_flag"] in ("PASS", "MARGINAL")
    assert row["tau"] == pytest.approx(2.219292440306282, rel=1e-4)
    assert row["chi2_reduced"] == pytest.approx(1.2335789483802273, rel=1e-4)


def test_citable_budget_nicknames_includes_whitney():
    names = load_citable_budget_nicknames()
    assert "whitney" in names
    assert "johndoeii" in names
    assert "chromatica" not in names


def test_load_joint_free_alpha_scalar_json(tmp_path, monkeypatch):
    ppc = {
        "burst": "fake",
        "tau_1ghz": 0.42,
        "alpha": 3.1,
        "suffix": "_ppc",
    }
    fit = {
        "burst": "fake",
        "percentiles": {
            "tau_1ghz": {"median": 0.99},
            "alpha": {"median": 4.0},
        },
    }
    root = tmp_path / "fits"
    root.mkdir()
    (root / "fake_joint_ppc_multi_pbf-exp-exp.json").write_text(json.dumps(ppc))
    (root / "fake_joint_fit_sharedzeta_pbf-exp-exp.json").write_text(json.dumps(fit))
    monkeypatch.setattr(
        "galaxies.foreground.tau_consistency.JOINT_GATE_CSV",
        tmp_path / "missing_gate.csv",
    )
    monkeypatch.setattr(
        "galaxies.foreground.tau_consistency.ALLEXP_FITS_DIR",
        root,
    )
    assert find_allexp_joint_json("fake") == root / "fake_joint_fit_sharedzeta_pbf-exp-exp.json"
    loaded = load_joint_free_alpha("fake")
    assert loaded["tau_joint_1ghz_ms"] == 0.99
    assert loaded["alpha_joint_free"] == 4.0


def test_load_joint_free_alpha_johndoeii_uses_promoted_c2d2():
    loaded = load_joint_free_alpha("johndoeII")
    assert loaded["joint_gate_final"] == "MARGINAL"
    assert loaded["joint_gate_source"].endswith("johndoeII_joint_fit_C2D2.json")
    assert loaded["tau_joint_1ghz_ms"] == pytest.approx(2.219292440306282, rel=1e-4)
    assert loaded["alpha_joint_free"] == pytest.approx(4.066411381454531, rel=1e-4)


def test_run_burst_raises_when_joint_output_missing(tmp_path, monkeypatch):
    # Must be a July-accepted burst: run_burst rejects non-accepted bursts
    # with ValueError before it ever gets to the subprocess/output-missing path.
    out_dir = tmp_path / "tau_consistency"
    monkeypatch.setattr(
        "galaxies.foreground.run_tau_consistency_refits.TAU_CONSISTENCY_DIR",
        out_dir,
    )
    monkeypatch.setattr(
        "galaxies.foreground.tau_consistency.TAU_CONSISTENCY_DIR",
        out_dir,
    )
    monkeypatch.setenv("FLITS_RUNS", str(tmp_path / "runs"))
    with patch("galaxies.foreground.run_tau_consistency_refits.subprocess.run"):
        with pytest.raises(FileNotFoundError, match="expected output missing"):
            run_burst("oran")


def test_load_july_accepted_morphologies_matches_adjudication():
    from galaxies.foreground.tau_consistency import (
        load_july_accepted_morphologies,
        parse_cxdy_variant,
    )

    morphologies = load_july_accepted_morphologies()
    normalized = {str(k).lower(): morph.variant for k, morph in morphologies.items()}
    assert normalized == JULY_ACCEPTED_VARIANTS
    assert set(morphologies) == set(JULY_ACCEPTED_VARIANTS)
    for nick, morph in morphologies.items():
        n_c, n_d = parse_cxdy_variant(JULY_ACCEPTED_VARIANTS[nick])
        assert morph.components_C == n_c
        assert morph.components_D == n_d


def test_load_july_accepted_morphologies_rejects_roster_drift(tmp_path):
    from galaxies.foreground.tau_consistency import load_july_accepted_morphologies

    rows = _adjudication_rows()
    accepted = next(row for row in rows if row["adjudication"] == "accepted_physical")
    accepted["adjudication"] = "rejected_test"
    altered = tmp_path / "fit_adjudication.csv"
    with altered.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="roster mismatch"):
        load_july_accepted_morphologies(altered)


@pytest.mark.parametrize(
    "variant, expected",
    [("C2D3", (2, 3)), ("C2D1", (2, 1)), ("C3D3", (3, 3)), ("C1D1", (1, 1)), ("C1D2", (1, 2))],
)
def test_parse_cxdy_variant_valid(variant, expected):
    from galaxies.foreground.tau_consistency import parse_cxdy_variant

    assert parse_cxdy_variant(variant) == expected


@pytest.mark.parametrize("bad", ["", "C2", "D3", "CXDY", "C2D3extra"])
def test_parse_cxdy_variant_invalid_raises(bad):
    from galaxies.foreground.tau_consistency import parse_cxdy_variant

    with pytest.raises(ValueError):
        parse_cxdy_variant(bad)


@pytest.mark.parametrize("burst", sorted(JULY_ACCEPTED_VARIANTS))
def test_build_alpha4_joint_cmd_contract(burst):
    from galaxies.foreground.run_tau_consistency_refits import build_alpha4_joint_cmd
    from galaxies.foreground.tau_consistency import (
        load_july_accepted_morphologies,
        parse_cxdy_variant,
    )

    morph = load_july_accepted_morphologies()[burst]
    cmd = build_alpha4_joint_cmd(burst, morph, nlive=600, nproc=8)
    n_c, n_d = parse_cxdy_variant(JULY_ACCEPTED_VARIANTS[burst])
    fixed_c, fixed_d = _fixed_delta_dm(burst)

    assert _cmd_value(cmd, "--components-C") == str(n_c)
    assert _cmd_value(cmd, "--components-D") == str(n_d)
    assert _cmd_value(cmd, "--alpha-lo") == "4"
    assert _cmd_value(cmd, "--alpha-hi") == "4"
    assert "--pbf-C" not in cmd
    assert "--pbf-D" not in cmd
    assert float(_cmd_value(cmd, "--fixed-delta-dm-C")) == pytest.approx(fixed_c, abs=1e-9)
    assert float(_cmd_value(cmd, "--fixed-delta-dm-D")) == pytest.approx(fixed_d, abs=1e-9)


@pytest.mark.parametrize("burst", sorted(JULY_ACCEPTED_VARIANTS))
def test_run_burst_command_construction(burst, tmp_path, monkeypatch):
    from galaxies.foreground import run_tau_consistency_refits as runner
    from galaxies.foreground.tau_consistency import parse_cxdy_variant

    monkeypatch.setattr(runner, "TAU_CONSISTENCY_DIR", tmp_path / "tau_consistency")
    monkeypatch.setattr(
        "galaxies.foreground.tau_consistency.TAU_CONSISTENCY_DIR",
        tmp_path / "tau_consistency",
    )
    monkeypatch.setenv("FLITS_RUNS", str(tmp_path / "runs"))
    n_c, n_d = parse_cxdy_variant(JULY_ACCEPTED_VARIANTS[burst])
    suffix = "" if JULY_ACCEPTED_VARIANTS[burst] == "C1D1" else f"_C{n_c}D{n_d}"
    produced = tmp_path / "runs" / "data" / "joint" / f"{burst}_joint_fit{suffix}.json"
    produced.parent.mkdir(parents=True, exist_ok=True)
    produced.write_text(json.dumps({}))

    with patch.object(runner, "subprocess") as mock_subprocess:
        runner.run_burst(burst)
        cmd = mock_subprocess.run.call_args[0][0]

    fixed_c, fixed_d = _fixed_delta_dm(burst)

    assert _cmd_value(cmd, "--components-C") == str(n_c)
    assert _cmd_value(cmd, "--components-D") == str(n_d)
    assert _cmd_value(cmd, "--alpha-lo") == "4"
    assert _cmd_value(cmd, "--alpha-hi") == "4"
    assert "--pbf-C" not in cmd
    assert "--pbf-D" not in cmd
    assert float(_cmd_value(cmd, "--fixed-delta-dm-C")) == pytest.approx(fixed_c, abs=1e-9)
    assert float(_cmd_value(cmd, "--fixed-delta-dm-D")) == pytest.approx(fixed_d, abs=1e-9)


@pytest.mark.parametrize("burst", NON_ACCEPTED_BURSTS)
def test_run_burst_rejects_non_accepted_bursts(burst, tmp_path, monkeypatch):
    from galaxies.foreground import run_tau_consistency_refits as runner

    monkeypatch.setattr(runner, "TAU_CONSISTENCY_DIR", tmp_path / "tau_consistency")
    monkeypatch.setattr(
        "galaxies.foreground.tau_consistency.TAU_CONSISTENCY_DIR",
        tmp_path / "tau_consistency",
    )
    monkeypatch.setenv("FLITS_RUNS", str(tmp_path / "runs"))
    with patch.object(runner, "subprocess") as mock_subprocess:
        with pytest.raises(ValueError):
            runner.run_burst(burst)
    mock_subprocess.run.assert_not_called()


def test_dry_run_prints_seven_lines(capsys):
    from galaxies.foreground import run_tau_consistency_refits as runner

    runner.main(["--dry-run"])
    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert len(lines) == 7
