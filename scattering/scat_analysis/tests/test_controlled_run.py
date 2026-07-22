from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scattering.scat_analysis.burstfit import FRBModel
from scattering.scat_analysis.burstfit_joint import (
    _append_derived_alpha_percentiles,
    _weighted_percentiles,
)
from scattering.scat_analysis.controlled_run import (
    ControlledRunError,
    _hash_distribution_snapshot,
    canonical_npz_sha256,
    environment_identity,
    finalize_receipt,
    identity_sha256,
    preflight,
    reverify_preflight,
    sha256,
)
from scattering.scat_analysis.joint_fit_diagnostics import (
    build_diagnostics,
    render_fit_panel,
    write_diagnostics,
)
from scattering.scat_analysis.joint_model_grid import build_model_grid_arrays
from scattering.scat_analysis.turbulence import alpha_from_beta

_PROCESSING_ENVIRONMENT = {
    "FLITS_JOINT_AUTO_TF": "1",
    "FLITS_MAX_CHANNELS": "64",
}
_COMMAND_ARGV = ["python", "run_controlled_joint_fit.py", "test", "--seed", "20260722"]
_PARAMETER_NAMES = [
    "tau_1ghz",
    "beta",
    "t0_C1",
    "zeta_C1",
    "delta_dm_C",
    "t0_D1",
    "zeta_D1",
    "delta_dm_D",
]
_DRAWS = np.array([[1.0, 3.5, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0]])
_WEIGHTS = np.array([1.0])
_LOG_WEIGHT = np.array([0.0])
_LOG_EVIDENCE_HISTORY = np.array([0.0])
_LOG_EVIDENCE_ERROR_HISTORY = np.array([0.0])
_NCALL_HISTORY = np.array([1])
_BETA_BOUNDS = np.array([3.0, 4.0])
_ALPHA_BOUNDS = np.array([alpha_from_beta(_BETA_BOUNDS[1]), alpha_from_beta(_BETA_BOUNDS[0])])
_TIME = np.array([-1.0, 0.0, 1.0])
_FREQ = np.array([0.9, 1.1])
_DATA = np.ones((2, 3))
_NOISE = np.ones(2)
_VALID = np.ones(2, dtype=bool)


def _support_identity(array: np.ndarray) -> dict:
    import hashlib

    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(json.dumps(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes())
    return {
        "shape": list(canonical.shape),
        "dtype": canonical.dtype.str,
        "sha256": digest.hexdigest(),
    }


_RESOLVED_IDENTITY = {
    "likelihood_class": "_JointLogLikelihoodGainMulti",
    "parameter_names": _PARAMETER_NAMES,
    "fixed_parameters": {},
    "prior_spec": [
        {"name": "tau_1ghz", "lower": 0.01, "upper": 10.0, "log_uniform": True},
        {"name": "beta", "lower": 3.0, "upper": 4.0, "log_uniform": False},
    ],
    "sampler": {
        "nlive": 20,
        "nproc": 1,
        "dlogz": 0.5,
        "sample": "rwalk",
        "seed": 20260722,
    },
    "processed_support": {
        band: {
            "arrays": {
                "time": _support_identity(_TIME),
                "freq": _support_identity(_FREQ),
                "data": _support_identity(_DATA),
                "noise_std": _support_identity(_NOISE),
                "valid": _support_identity(_VALID),
            },
            "model_metadata": {
                "dm_init": 0.0,
                "df_MHz": 0.390625,
                "dispersion_beta": 2.0,
            },
        }
        for band in ("C", "D")
    },
}


def _complete_preparation(receipt: dict) -> dict:
    receipt["resolved_fit_identity"] = _RESOLVED_IDENTITY
    receipt["resolved_fit_identity_sha256"] = identity_sha256(_RESOLVED_IDENTITY)
    receipt["post_preparation_reverification_passed"] = True
    return receipt


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict[str, Path]]:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    lock = repo / "uv.lock"
    lock.write_text("locked\n", encoding="utf-8")
    source_files = {
        name: repo / f"{name}.py"
        for name in (
            "controlled_entrypoint",
            "fit_driver",
            "joint_tf_prep_source",
            "burstfit_joint_source",
            "controlled_run_source",
            "model_grid_source",
            "diagnostic_source",
        )
    }
    for name, source in source_files.items():
        source.write_text(f"# {name}\n", encoding="utf-8")
    _git(repo, "add", "uv.lock", *[path.name for path in source_files.values()])
    _git(repo, "commit", "-qm", "fixture")

    external = tmp_path / "external"
    external.mkdir()
    files = {
        "chime_input": external / "chime.npy",
        "dsa_input": external / "dsa.npy",
        "chime_config": external / "chime.yaml",
        "dsa_config": external / "dsa.yaml",
        "environment_lock": lock,
        **source_files,
    }
    np.save(files["chime_input"], np.arange(4))
    np.save(files["dsa_input"], np.arange(5))
    files["chime_config"].write_text("band: chime\n", encoding="utf-8")
    files["dsa_config"].write_text("band: dsa\n", encoding="utf-8")

    invocation = {
        "burst": "test",
        "seed": 20260722,
        "nlive": 20,
        "nproc": 1,
        "dlogz": 0.5,
        "sample": "rwalk",
        "components_C": 1,
        "components_D": 1,
        "gain_s2": 100,
        "shared_zeta": False,
        "marginalize_gain": False,
        "marginalize_gain_gp": False,
        "force_multi": False,
        "mu_degree": 1,
    }
    contract = {
        "schema": "flits-controlled-joint-fit-contract/v1",
        "burst": "test",
        "source_revision": _git(repo, "rev-parse", "HEAD"),
        "fit_configuration": invocation,
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in files.items()
        },
        "environment_variables": _PROCESSING_ENVIRONMENT,
        "environment_identity_sha256": environment_identity(lock)["identity_sha256"],
        "resolved_fit_identity_sha256": identity_sha256(_RESOLVED_IDENTITY),
        "command": {
            "argv": _COMMAND_ARGV,
            "working_directory": str(repo.resolve()),
        },
        "executed_source_files": list(source_files),
    }
    contract_path = external / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return repo, contract_path, invocation, files


def _write_valid_output_packet(root: Path, receipt: dict) -> dict[str, Path]:
    paths = {
        "fit_summary": root / "fit.json",
        "weighted_samples": root / "samples.npz",
        "model_grid": root / "model.npz",
        "diagnostics": root / "diagnostics.json",
        "panel": root / "panel.svg",
    }
    percentiles = _append_derived_alpha_percentiles(
        _weighted_percentiles(_DRAWS, _WEIGHTS, tuple(_PARAMETER_NAMES)),
        _DRAWS,
        _WEIGHTS,
        tuple(_PARAMETER_NAMES),
    )
    summary = {
        "burst": receipt["burst"],
        "seed": receipt["fit_configuration"]["seed"],
        "components_C": 1,
        "components_D": 1,
        "gain_model": "proper_gaussian",
        "gain_s2": 100,
        "shared_zeta": False,
        "marginalize_gain": False,
        "marginalize_gain_gp": False,
        "force_multi": False,
        "mu_degree": 1,
        "nlive": 20,
        "nproc": 1,
        "dlogz": 0.5,
        "sample": "rwalk",
        "fixed_parameters": {},
        "beta": {key: percentiles["beta"][key] for key in ("median", "err_minus", "err_plus")},
        "beta_bounds": [3.0, 4.0],
        "alpha_bounds": _ALPHA_BOUNDS.tolist(),
        "alpha": {key: percentiles["alpha"][key] for key in ("median", "err_minus", "err_plus")},
        "tau_1ghz": {
            key: percentiles["tau_1ghz"][key] for key in ("median", "err_minus", "err_plus")
        },
        "log_evidence": 0.0,
        "log_evidence_err": 0.0,
        "ncall": 1,
        "percentiles": percentiles,
        "controlled_contract_sha256": receipt["contract"]["sha256"],
        "resolved_fit_identity_sha256": receipt["resolved_fit_identity_sha256"],
        "source_revision": receipt["source"]["revision"],
    }
    paths["fit_summary"].write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    np.savez(
        paths["weighted_samples"],
        samples=_DRAWS,
        weights=_WEIGHTS,
        log_weight=_LOG_WEIGHT,
        log_evidence_history=_LOG_EVIDENCE_HISTORY,
        log_evidence_error_history=_LOG_EVIDENCE_ERROR_HISTORY,
        ncall_history=_NCALL_HISTORY,
        param_names=np.array(_PARAMETER_NAMES),
        beta_bounds=_BETA_BOUNDS,
        alpha_bounds=_ALPHA_BOUNDS,
    )
    model_c = FRBModel(time=_TIME, freq=_FREQ, data=_DATA, noise_std=_NOISE)
    model_d = FRBModel(time=_TIME, freq=_FREQ, data=_DATA, noise_std=_NOISE)
    grid = build_model_grid_arrays(model_c, model_d, summary)
    np.savez(paths["model_grid"], **grid)
    diagnostics = build_diagnostics(
        summary,
        grid,
        samples=_DRAWS,
        weights=_WEIGHTS,
        param_names=_PARAMETER_NAMES,
    )
    write_diagnostics(paths["diagnostics"], diagnostics)
    render_fit_panel(grid, paths["panel"])
    return paths


def test_preflight_binds_clean_source_contract_command_and_environment(
    tmp_path: Path,
) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)

    receipt = preflight(
        contract_path=contract,
        repo=repo,
        invocation=invocation,
        resolved_files=files,
        argv=_COMMAND_ARGV,
        cwd=repo,
        environment_variables=_PROCESSING_ENVIRONMENT,
    )

    assert receipt["preflight_passed"] is True
    assert receipt["source"]["clean_worktree"] is True
    assert receipt["source"]["revision"] == _git(repo, "rev-parse", "HEAD")
    assert receipt["fit_configuration"]["seed"] == 20260722
    assert receipt["command"]["argv"][-1] == "20260722"
    assert receipt["command"]["working_directory"] == str(repo.resolve())
    assert receipt["files"]["chime_input"]["sha256"] == sha256(files["chime_input"])
    assert len(receipt["environment"]["identity_sha256"]) == 64
    assert receipt["required_post_fit_guards"]["broad_width_to_window_ratio"] == 5.0
    assert receipt["required_post_fit_guards"]["low_fluence_fraction"] == 0.05


def test_distribution_snapshot_rejects_internal_file_mutation(tmp_path: Path) -> None:
    runtime_file = tmp_path / "sampler.py"
    runtime_file.write_bytes(b"original")

    def snapshot() -> tuple:
        stat = runtime_file.stat()
        expected = base64.urlsafe_b64encode(hashlib.sha256(b"original").digest()).rstrip(b"=")
        return (
            (
                "dynesty/sampler.py",
                str(runtime_file),
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_ino,
                "sha256",
                expected.decode("ascii"),
            ),
        )

    identity = _hash_distribution_snapshot(snapshot())
    assert identity["file_count"] == 1
    runtime_file.write_bytes(b"mutated!")
    _hash_distribution_snapshot.cache_clear()

    with pytest.raises(ControlledRunError, match="differs from installed RECORD"):
        _hash_distribution_snapshot(snapshot())


def test_preflight_rejects_missing_seed(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    invocation["seed"] = None

    with pytest.raises(ControlledRunError, match="explicit seed"):
        preflight(contract, repo, invocation, files, ["runner"], repo, {})


def test_preflight_rejects_invalid_seed(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    invocation["seed"] = -1

    with pytest.raises(ControlledRunError, match="unsigned 64-bit"):
        preflight(contract, repo, invocation, files, ["runner"], repo, {})


def test_preflight_rejects_dirty_source(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    (repo / "dirty.py").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ControlledRunError, match="source worktree is dirty"):
        preflight(contract, repo, invocation, files, ["runner"], repo, {})


def test_preflight_rejects_missing_input(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    files["chime_input"].unlink()

    with pytest.raises(ControlledRunError, match="missing file chime_input"):
        preflight(contract, repo, invocation, files, ["runner"], repo, {})


def test_preflight_rejects_hash_mismatch(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    files["chime_config"].write_text("changed: true\n", encoding="utf-8")

    with pytest.raises(ControlledRunError, match="hash mismatch for chime_config"):
        preflight(contract, repo, invocation, files, ["runner"], repo, {})


def test_preflight_rejects_command_or_cwd_mismatch(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)

    with pytest.raises(ControlledRunError, match="command or working directory"):
        preflight(
            contract,
            repo,
            invocation,
            files,
            [*_COMMAND_ARGV, "--changed"],
            repo,
            _PROCESSING_ENVIRONMENT,
        )


def test_preflight_rejects_runtime_environment_mismatch(tmp_path: Path) -> None:
    repo, contract_path, invocation, files = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["environment_identity_sha256"] = "0" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ControlledRunError, match="runtime environment identity"):
        preflight(
            contract_path,
            repo,
            invocation,
            files,
            _COMMAND_ARGV,
            repo,
            _PROCESSING_ENVIRONMENT,
        )


def test_preflight_rejects_off_repo_executed_source(tmp_path: Path) -> None:
    repo, contract_path, invocation, files = _fixture(tmp_path)
    copied_driver = tmp_path / "copied-runner.py"
    copied_driver.write_text("# controlled entrypoint\n", encoding="utf-8")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["files"]["controlled_entrypoint"] = {
        "path": str(copied_driver.resolve()),
        "sha256": sha256(copied_driver),
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    files["controlled_entrypoint"] = copied_driver

    with pytest.raises(ControlledRunError, match="outside repository"):
        preflight(
            contract_path,
            repo,
            invocation,
            files,
            _COMMAND_ARGV,
            repo,
            _PROCESSING_ENVIRONMENT,
        )


def test_reverification_rejects_input_changed_during_preprocessing(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = preflight(
        contract,
        repo,
        invocation,
        files,
        _COMMAND_ARGV,
        repo,
        _PROCESSING_ENVIRONMENT,
    )
    files["dsa_input"].write_bytes(b"changed")

    with pytest.raises(ControlledRunError, match="file changed after preflight: dsa_input"):
        reverify_preflight(receipt)


def test_npz_scientific_hash_ignores_container_order(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(first, a=np.arange(3), b=np.array(["x", "y"]))
    np.savez(second, b=np.array(["x", "y"]), a=np.arange(3))

    assert sha256(first) != sha256(second)
    assert canonical_npz_sha256(first) == canonical_npz_sha256(second)


def test_finalize_receipt_hashes_all_outputs(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = preflight(
        contract,
        repo,
        invocation,
        files,
        _COMMAND_ARGV,
        repo,
        _PROCESSING_ENVIRONMENT,
    )
    _complete_preparation(receipt)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    outputs = _write_valid_output_packet(tmp_path, receipt)

    final = finalize_receipt(receipt_path, outputs)

    assert final["outputs_complete"] is True
    assert final["outputs"]["fit_summary"]["sha256"] == sha256(outputs["fit_summary"])
    assert final["outputs"]["weighted_samples"]["scientific_sha256"] == (
        canonical_npz_sha256(outputs["weighted_samples"])
    )
    assert final["outputs"]["panel"]["sha256"] == sha256(outputs["panel"])


def test_finalize_receipt_rejects_empty_complete_packet(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = _complete_preparation(
        preflight(
            contract,
            repo,
            invocation,
            files,
            _COMMAND_ARGV,
            repo,
            _PROCESSING_ENVIRONMENT,
        )
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    outputs = _write_valid_output_packet(tmp_path, receipt)
    outputs["fit_summary"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ControlledRunError, match="lacks controlled-fit fields"):
        finalize_receipt(receipt_path, outputs)


@pytest.mark.parametrize(
    "substitution",
    [
        "negative_weights",
        "wrong_model",
        "blank_panel",
        "wrong_gain_prior",
        "wrong_evidence",
    ],
)
def test_finalize_receipt_rejects_scientific_output_substitution(
    tmp_path: Path, substitution: str
) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = _complete_preparation(
        preflight(
            contract,
            repo,
            invocation,
            files,
            _COMMAND_ARGV,
            repo,
            _PROCESSING_ENVIRONMENT,
        )
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    outputs = _write_valid_output_packet(tmp_path, receipt)
    if substitution == "negative_weights":
        np.savez(
            outputs["weighted_samples"],
            samples=np.vstack((_DRAWS, _DRAWS)),
            weights=np.array([-1.0, 2.0]),
            log_weight=np.array([0.0, 0.0]),
            log_evidence_history=np.array([0.0, 0.0]),
            log_evidence_error_history=np.array([0.0, 0.0]),
            ncall_history=np.array([1, 1]),
            param_names=np.array(_PARAMETER_NAMES),
            beta_bounds=_BETA_BOUNDS,
            alpha_bounds=_ALPHA_BOUNDS,
        )
        match = "invalid dimensions or values"
    elif substitution == "wrong_model":
        with np.load(outputs["model_grid"], allow_pickle=False) as archive:
            model = {name: archive[name] for name in archive.files}
        model["tau_1ghz"] = np.array(-999.0)
        np.savez(outputs["model_grid"], **model)
        match = "field differs from regeneration: tau_1ghz"
    elif substitution == "blank_panel":
        outputs["panel"].write_text('<svg xmlns="http://www.w3.org/2000/svg"/>\n', encoding="utf-8")
        match = "panel differs from regeneration"
    elif substitution == "wrong_gain_prior":
        summary = json.loads(outputs["fit_summary"].read_text(encoding="utf-8"))
        summary["gain_s2"] = 10
        outputs["fit_summary"].write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        model_c = FRBModel(time=_TIME, freq=_FREQ, data=_DATA, noise_std=_NOISE)
        model_d = FRBModel(time=_TIME, freq=_FREQ, data=_DATA, noise_std=_NOISE)
        grid = build_model_grid_arrays(model_c, model_d, summary)
        np.savez(outputs["model_grid"], **grid)
        diagnostics = build_diagnostics(
            summary,
            grid,
            samples=_DRAWS,
            weights=_WEIGHTS,
            param_names=_PARAMETER_NAMES,
        )
        write_diagnostics(outputs["diagnostics"], diagnostics)
        render_fit_panel(grid, outputs["panel"])
        match = "gain_s2 differs from frozen fit configuration"
    else:
        summary = json.loads(outputs["fit_summary"].read_text(encoding="utf-8"))
        summary["log_evidence"] = 999.0
        outputs["fit_summary"].write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        match = "sampler summary differs from sampler history"

    with pytest.raises(ControlledRunError, match=match):
        finalize_receipt(receipt_path, outputs)


def test_finalize_receipt_cannot_weaken_required_outputs(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = preflight(
        contract,
        repo,
        invocation,
        files,
        _COMMAND_ARGV,
        repo,
        _PROCESSING_ENVIRONMENT,
    )
    _complete_preparation(receipt)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    summary = tmp_path / "fit.json"
    summary.write_text("{}\n", encoding="utf-8")

    final = finalize_receipt(receipt_path, {"fit_summary": summary})
    assert final["outputs_complete"] is False


def test_finalize_receipt_requires_resolved_pre_sampler_identity(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = preflight(
        contract,
        repo,
        invocation,
        files,
        _COMMAND_ARGV,
        repo,
        _PROCESSING_ENVIRONMENT,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    summary = tmp_path / "fit.json"
    summary.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ControlledRunError, match="resolved fit identity was not verified"):
        finalize_receipt(receipt_path, {"fit_summary": summary})


def test_finalize_receipt_rejects_forged_preparation_flag_without_identity(
    tmp_path: Path,
) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = preflight(
        contract,
        repo,
        invocation,
        files,
        _COMMAND_ARGV,
        repo,
        _PROCESSING_ENVIRONMENT,
    )
    receipt["post_preparation_reverification_passed"] = True
    receipt["resolved_fit_identity_sha256"] = identity_sha256(_RESOLVED_IDENTITY)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    outputs = _write_valid_output_packet(tmp_path, receipt)

    with pytest.raises(ControlledRunError, match="identity object is missing or invalid"):
        finalize_receipt(receipt_path, outputs)


def test_finalize_receipt_rejects_same_file_for_canonical_roles(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = _complete_preparation(
        preflight(
            contract,
            repo,
            invocation,
            files,
            _COMMAND_ARGV,
            repo,
            _PROCESSING_ENVIRONMENT,
        )
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ControlledRunError, match="wrong file type"):
        finalize_receipt(
            receipt_path,
            {
                "fit_summary": artifact,
                "weighted_samples": artifact,
                "model_grid": artifact,
                "diagnostics": artifact,
                "panel": artifact,
            },
        )


def test_finalize_receipt_rejects_forged_input_rebinding(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = _complete_preparation(
        preflight(
            contract,
            repo,
            invocation,
            files,
            _COMMAND_ARGV,
            repo,
            _PROCESSING_ENVIRONMENT,
        )
    )
    files["chime_input"].write_bytes(b"forged")
    receipt["files"]["chime_input"]["sha256"] = sha256(files["chime_input"])
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    summary = tmp_path / "fit.json"
    summary.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ControlledRunError, match="receipt file identities differ"):
        finalize_receipt(receipt_path, {"fit_summary": summary})


def test_finalize_receipt_rejects_post_preflight_tamper(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = preflight(
        contract,
        repo,
        invocation,
        files,
        _COMMAND_ARGV,
        repo,
        _PROCESSING_ENVIRONMENT,
    )
    _complete_preparation(receipt)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    summary = tmp_path / "fit.json"
    summary.write_text("{}\n", encoding="utf-8")
    files["chime_input"].write_bytes(b"tampered")

    with pytest.raises(ControlledRunError, match="file changed after preflight: chime_input"):
        finalize_receipt(receipt_path, {"fit_summary": summary})


def test_finalize_receipt_rejects_changed_previous_output(tmp_path: Path) -> None:
    repo, contract, invocation, files = _fixture(tmp_path)
    receipt = preflight(
        contract,
        repo,
        invocation,
        files,
        _COMMAND_ARGV,
        repo,
        _PROCESSING_ENVIRONMENT,
    )
    _complete_preparation(receipt)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    summary = tmp_path / "fit.json"
    summary.write_text("{}\n", encoding="utf-8")
    finalize_receipt(receipt_path, {"fit_summary": summary})
    summary.write_text('{"changed": true}\n', encoding="utf-8")
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ControlledRunError, match="previously recorded output changed"):
        finalize_receipt(receipt_path, {"diagnostics": diagnostics})
