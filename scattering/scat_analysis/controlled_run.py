"""Fail-closed provenance receipts for controlled joint-scattering fits."""

from __future__ import annotations

import base64
import functools
import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

DEPRECATED_ZACH_GUARDS = {
    "component_arrival_16_84_intervals_inside_fitted_window": True,
    "broad_width_to_window_ratio": 5.0,
    "low_fluence_fraction": 0.05,
    "require_component_identity_across_outputs": True,
    "require_residual_map_and_profile_diagnostics": True,
    "evidence_comparison_requires_identical_likelihood_prior_support_and_mode": True,
}
CANONICAL_OUTPUTS = frozenset(
    {"fit_summary", "weighted_samples", "model_grid", "diagnostics", "panel"}
)
CANONICAL_OUTPUT_SUFFIXES = {
    "fit_summary": ".json",
    "weighted_samples": ".npz",
    "model_grid": ".npz",
    "diagnostics": ".json",
    "panel": ".svg",
}
CONTROLLED_SOURCE_NAMES = frozenset(
    {
        "controlled_entrypoint",
        "fit_driver",
        "joint_tf_prep_source",
        "burstfit_joint_source",
        "controlled_run_source",
        "model_grid_source",
        "diagnostic_source",
    }
)


class ControlledRunError(RuntimeError):
    """A controlled run cannot prove its required identity."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def identity_sha256(value: Any) -> str:
    """Hash a JSON-compatible scientific/provenance identity canonically."""
    return _canonical_json_sha256(value)


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    identity: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    if array.dtype.hasobject:
        identity["values"] = array.tolist()
    else:
        canonical = np.ascontiguousarray(array)
        if canonical.dtype.byteorder == ">" or (
            canonical.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
        identity["data_sha256"] = hashlib.sha256(canonical.tobytes()).hexdigest()
    return identity


def canonical_npz_sha256(path: Path) -> str:
    """Hash named scientific arrays, independent of ZIP member order/metadata."""
    with np.load(path, allow_pickle=True) as archive:
        identity = {name: _array_identity(archive[name]) for name in sorted(archive.files)}
    return _canonical_json_sha256(identity)


def _load_json_object(path: Path, role: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ControlledRunError(f"canonical {role} must be a JSON object")
    return value


def _scientific_array_identity(array: np.ndarray) -> dict[str, Any]:
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


def _arrays_equal(observed: Any, expected: Any) -> bool:
    first = np.asarray(observed)
    second = np.asarray(expected)
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    if np.issubdtype(first.dtype, np.inexact):
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _validate_complete_output_packet(
    receipt: Mapping[str, Any], identities: Mapping[str, Mapping[str, str]]
) -> None:
    """Rebuild every derived output and reject any internally substituted packet."""
    paths = {name: Path(identities[name]["path"]) for name in CANONICAL_OUTPUTS}
    summary = _load_json_object(paths["fit_summary"], "fit summary")
    required_summary = {
        "burst",
        "seed",
        "components_C",
        "components_D",
        "gain_model",
        "gain_s2",
        "shared_zeta",
        "marginalize_gain",
        "marginalize_gain_gp",
        "force_multi",
        "mu_degree",
        "nlive",
        "nproc",
        "dlogz",
        "sample",
        "fixed_parameters",
        "beta",
        "beta_bounds",
        "alpha_bounds",
        "alpha",
        "tau_1ghz",
        "log_evidence",
        "log_evidence_err",
        "ncall",
        "percentiles",
        "controlled_contract_sha256",
        "resolved_fit_identity_sha256",
        "source_revision",
    }
    if not required_summary.issubset(summary):
        raise ControlledRunError("canonical fit summary lacks controlled-fit fields")
    if (
        summary["burst"] != receipt["burst"]
        or summary["seed"] != receipt["fit_configuration"]["seed"]
        or summary["controlled_contract_sha256"] != receipt["contract"]["sha256"]
        or summary["resolved_fit_identity_sha256"] != receipt["resolved_fit_identity_sha256"]
        or summary["source_revision"] != receipt["source"]["revision"]
    ):
        raise ControlledRunError("canonical fit summary identity differs from receipt")
    resolved = receipt["resolved_fit_identity"]
    if (
        resolved.get("likelihood_class") != "_JointLogLikelihoodGainMulti"
        or summary["gain_model"] != "proper_gaussian"
    ):
        raise ControlledRunError("canonical gain model differs from resolved likelihood")
    configuration = receipt["fit_configuration"]
    for name in (
        "gain_s2",
        "components_C",
        "components_D",
        "shared_zeta",
        "marginalize_gain",
        "marginalize_gain_gp",
        "force_multi",
        "mu_degree",
        "nlive",
        "nproc",
        "dlogz",
        "sample",
        "seed",
    ):
        if summary[name] != configuration[name]:
            raise ControlledRunError(f"canonical {name} differs from frozen fit configuration")
    sampler = resolved.get("sampler", {})
    for name in ("nlive", "nproc", "dlogz", "sample", "seed"):
        if summary[name] != sampler.get(name):
            raise ControlledRunError(f"canonical {name} differs from resolved sampler")
    beta_prior = next(
        (prior for prior in resolved.get("prior_spec", []) if prior.get("name") == "beta"),
        None,
    )
    if beta_prior is None or summary["beta_bounds"] != [
        beta_prior["lower"],
        beta_prior["upper"],
    ]:
        raise ControlledRunError("canonical beta bounds differ from resolved prior")

    with np.load(paths["weighted_samples"], allow_pickle=True) as samples:
        required_samples = {
            "samples",
            "weights",
            "param_names",
            "log_weight",
            "log_evidence_history",
            "log_evidence_error_history",
            "ncall_history",
            "beta_bounds",
            "alpha_bounds",
        }
        if not required_samples.issubset(samples.files):
            raise ControlledRunError("canonical weighted samples lack required arrays")
        draws = np.asarray(samples["samples"])
        weights = np.asarray(samples["weights"])
        names = np.asarray(samples["param_names"])
        log_weight = np.asarray(samples["log_weight"], dtype=float)
        log_evidence_history = np.asarray(samples["log_evidence_history"], dtype=float)
        log_evidence_error_history = np.asarray(samples["log_evidence_error_history"], dtype=float)
        ncall_history = np.asarray(samples["ncall_history"])
        recorded_beta_bounds = np.asarray(samples["beta_bounds"], dtype=float)
        recorded_alpha_bounds = np.asarray(samples["alpha_bounds"], dtype=float)
        if (
            draws.ndim != 2
            or weights.shape != (draws.shape[0],)
            or names.shape != (draws.shape[1],)
            or log_weight.shape != (draws.shape[0],)
            or log_evidence_history.shape != (draws.shape[0],)
            or log_evidence_error_history.shape != (draws.shape[0],)
            or ncall_history.shape != (draws.shape[0],)
            or not np.isfinite(draws).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(log_weight).all()
            or not np.isfinite(log_evidence_history).all()
            or not np.isfinite(log_evidence_error_history).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0)
            or np.any(ncall_history < 0)
            or recorded_beta_bounds.shape != (2,)
            or recorded_alpha_bounds.shape != (2,)
        ):
            raise ControlledRunError("canonical weighted samples have invalid dimensions or values")
        derived_weights = np.exp(log_weight - log_evidence_history[-1])
        derived_weights /= derived_weights.sum()
        if not np.array_equal(weights, derived_weights):
            raise ControlledRunError("canonical weights differ from sampler history")
        parameter_names = [str(name) for name in names.tolist()]
        if parameter_names != resolved.get("parameter_names"):
            raise ControlledRunError("canonical sample parameters differ from resolved fit")

    if not np.array_equal(recorded_beta_bounds, np.asarray(summary["beta_bounds"], dtype=float)):
        raise ControlledRunError("canonical beta bounds differ from sampler artifact")
    from .turbulence import alpha_from_beta

    derived_alpha_bounds = np.array(
        [alpha_from_beta(recorded_beta_bounds[1]), alpha_from_beta(recorded_beta_bounds[0])]
    )
    if not np.array_equal(recorded_alpha_bounds, derived_alpha_bounds) or not np.array_equal(
        recorded_alpha_bounds, np.asarray(summary["alpha_bounds"], dtype=float)
    ):
        raise ControlledRunError("canonical alpha bounds differ from sampled beta bounds")
    if (
        summary["log_evidence"] != float(log_evidence_history[-1])
        or summary["log_evidence_err"] != float(log_evidence_error_history[-1])
        or summary["ncall"] != int(np.sum(ncall_history))
    ):
        raise ControlledRunError("canonical sampler summary differs from sampler history")

    from .burstfit_joint import _append_derived_alpha_percentiles, _weighted_percentiles

    expected_percentiles = _append_derived_alpha_percentiles(
        _weighted_percentiles(draws, weights, tuple(parameter_names)),
        draws,
        weights,
        tuple(parameter_names),
    )
    if summary["fixed_parameters"] != resolved.get("fixed_parameters", {}):
        raise ControlledRunError("canonical fixed parameters differ from resolved fit")
    for name, value in summary["fixed_parameters"].items():
        expected_percentiles[name] = {
            "median": value,
            "lower": value,
            "upper": value,
            "err_minus": 0.0,
            "err_plus": 0.0,
        }
    if summary["percentiles"] != expected_percentiles:
        raise ControlledRunError("canonical fit percentiles do not match weighted samples")
    for name in ("beta", "alpha", "tau_1ghz"):
        expected = {
            key: expected_percentiles[name][key] for key in ("median", "err_minus", "err_plus")
        }
        if summary[name] != expected:
            raise ControlledRunError(f"canonical {name} summary differs from weighted samples")

    model_keys = {
        f"{stem}{band}"
        for band in ("C", "D")
        for stem in ("data", "model", "freq", "time", "noise", "valid", "fluence")
    } | {
        "nC",
        "nD",
        "burst",
        "alpha",
        "beta",
        "tau_1ghz",
        "residual_mean_squareC",
        "residual_mean_squareD",
        "gain_model",
        "gain_s2_C",
        "gain_s2_D",
        "dm_initC",
        "dm_initD",
        "df_MHzC",
        "df_MHzD",
        "dispersion_betaC",
        "dispersion_betaD",
    }
    with np.load(paths["model_grid"], allow_pickle=False) as model:
        if not model_keys.issubset(model.files):
            raise ControlledRunError("canonical model grid lacks required arrays")
        if str(model["gain_model"]) != summary["gain_model"]:
            raise ControlledRunError("canonical model-grid gain model differs from fit summary")
        if str(model["burst"]) != summary["burst"]:
            raise ControlledRunError("canonical model-grid burst differs from fit summary")
        model_grid = {name: np.asarray(model[name]) for name in model.files}
        for band in ("C", "D"):
            data = np.asarray(model[f"data{band}"])
            prediction = np.asarray(model[f"model{band}"])
            frequency = np.asarray(model[f"freq{band}"])
            time = np.asarray(model[f"time{band}"])
            noise = np.asarray(model[f"noise{band}"])
            valid = np.asarray(model[f"valid{band}"])
            count = int(model[f"n{band}"])
            fluence = np.asarray(model[f"fluence{band}"])
            if (
                data.ndim != 2
                or prediction.shape != data.shape
                or frequency.shape != (data.shape[0],)
                or time.shape != (data.shape[1],)
                or noise.shape != (data.shape[0],)
                or valid.shape != (data.shape[0],)
                or fluence.shape != (count,)
            ):
                raise ControlledRunError(f"canonical {band} model grid has inconsistent dimensions")
            if count != int(summary[f"components_{band}"]):
                raise ControlledRunError(
                    f"canonical {band} component count differs from fit summary"
                )

    support_names = {
        "data": "data",
        "freq": "freq",
        "time": "time",
        "noise": "noise_std",
        "valid": "valid",
    }
    for band in ("C", "D"):
        support = resolved["processed_support"][band]
        for grid_name, support_name in support_names.items():
            if (
                _scientific_array_identity(model_grid[f"{grid_name}{band}"])
                != support["arrays"][support_name]
            ):
                raise ControlledRunError(
                    f"canonical {band} model support differs from resolved fit"
                )
        for grid_name, metadata_name in (
            ("dm_init", "dm_init"),
            ("df_MHz", "df_MHz"),
            ("dispersion_beta", "dispersion_beta"),
        ):
            if float(model_grid[f"{grid_name}{band}"]) != float(
                support["model_metadata"][metadata_name]
            ):
                raise ControlledRunError(
                    f"canonical {band} model metadata differs from resolved fit"
                )

    from .burstfit import FRBModel
    from .joint_model_grid import build_model_grid_arrays

    reconstructed = {}
    for band in ("C", "D"):
        reconstructed[band] = FRBModel(
            time=model_grid[f"time{band}"],
            freq=model_grid[f"freq{band}"],
            data=model_grid[f"data{band}"],
            noise_std=model_grid[f"noise{band}"],
            dm_init=float(model_grid[f"dm_init{band}"]),
            df_MHz=float(model_grid[f"df_MHz{band}"]),
            beta=float(model_grid[f"dispersion_beta{band}"]),
        )
    regenerated_grid = build_model_grid_arrays(reconstructed["C"], reconstructed["D"], summary)
    if set(regenerated_grid) != set(model_grid):
        raise ControlledRunError("canonical model-grid fields differ from regeneration")
    for name, expected in regenerated_grid.items():
        if not _arrays_equal(model_grid[name], expected):
            raise ControlledRunError(
                f"canonical model-grid field differs from regeneration: {name}"
            )

    diagnostics = _load_json_object(paths["diagnostics"], "diagnostics")
    if (
        diagnostics.get("schema") != "flits-controlled-joint-fit-diagnostics/v1"
        or diagnostics.get("burst") != summary["burst"]
        or diagnostics.get("source_revision") != summary["source_revision"]
        or diagnostics.get("controlled_contract_sha256") != summary["controlled_contract_sha256"]
        or diagnostics.get("resolved_fit_identity_sha256")
        != summary["resolved_fit_identity_sha256"]
    ):
        raise ControlledRunError("canonical diagnostics identity is invalid")

    from .joint_fit_diagnostics import build_diagnostics, render_fit_panel

    regenerated_diagnostics = build_diagnostics(
        summary,
        model_grid,
        samples=draws,
        weights=weights,
        param_names=parameter_names,
    )
    if diagnostics != regenerated_diagnostics:
        raise ControlledRunError("canonical diagnostics differ from regeneration")

    try:
        root = ET.parse(paths["panel"]).getroot()
    except (ET.ParseError, OSError) as error:
        raise ControlledRunError("canonical panel is not valid SVG") from error
    if root.tag.split("}")[-1] != "svg":
        raise ControlledRunError("canonical panel root is not SVG")
    with tempfile.TemporaryDirectory(prefix="flits-panel-check-") as temporary:
        regenerated_panel = Path(temporary) / "panel.svg"
        render_fit_panel(model_grid, regenerated_panel)
        if sha256(regenerated_panel) != sha256(paths["panel"]):
            raise ControlledRunError("canonical panel differs from regeneration")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@functools.lru_cache(maxsize=32)
def _hash_distribution_snapshot(
    snapshot: tuple[tuple[str, str, int, int, int, int, str | None, str | None], ...],
) -> dict[str, Any]:
    manifest = []
    for entry, raw_path, size, _mtime_ns, _ctime_ns, _inode, mode, expected in snapshot:
        path = Path(raw_path)
        actual_sha256 = sha256(path)
        if mode is not None:
            digest = hashlib.new(mode)
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            actual_recorded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
            if actual_recorded != expected:
                raise ControlledRunError(
                    f"runtime distribution file differs from installed RECORD: {path}"
                )
        manifest.append(
            {
                "path": entry,
                "size": size,
                "sha256": actual_sha256,
            }
        )
    return {
        "file_count": len(manifest),
        "content_sha256": _canonical_json_sha256(manifest),
    }


def _distribution_content_identity(distribution: metadata.Distribution) -> dict[str, Any]:
    """Hash and wheel-verify every installed file, including compiled libraries."""
    files = distribution.files
    if files is None:
        raise ControlledRunError(
            f"runtime distribution has no installed-file inventory: {distribution.metadata['Name']}"
        )
    snapshot = []
    for entry in sorted(files, key=str):
        path = Path(distribution.locate_file(entry)).resolve()
        if not path.is_file():
            raise ControlledRunError(f"runtime distribution file is missing: {path}")
        stat = path.stat()
        snapshot.append(
            (
                str(entry),
                str(path),
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_ino,
                entry.hash.mode if entry.hash is not None else None,
                entry.hash.value if entry.hash is not None else None,
            )
        )
    return _hash_distribution_snapshot(tuple(snapshot))


def environment_identity(environment_lock: Path) -> dict[str, Any]:
    packages = {}
    for distribution, module_name in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("dynesty", "dynesty"),
        ("pyyaml", "yaml"),
        ("matplotlib", "matplotlib"),
    ):
        try:
            installed = metadata.distribution(distribution)
        except metadata.PackageNotFoundError as error:
            raise ControlledRunError(
                f"required runtime distribution is missing: {distribution}"
            ) from error
        record = installed.read_text("RECORD")
        if record is None:
            raise ControlledRunError(
                f"runtime distribution has no installed-file record: {distribution}"
            )
        module_path = Path(importlib.import_module(module_name).__file__).resolve()
        packages[distribution] = {
            "version": installed.version,
            "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            "imported_module_path": str(module_path),
            "imported_module_sha256": sha256(module_path),
            **_distribution_content_identity(installed),
        }
    observed = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "packages": packages,
        "numerical_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "environment_lock": {
            "path": str(environment_lock.resolve()),
            "sha256": sha256(environment_lock),
        },
    }
    observed["identity_sha256"] = _canonical_json_sha256(observed)
    return observed


def _verify_source_files(
    repo: Path, source_names: Sequence[str], files: Mapping[str, dict[str, str]]
) -> None:
    if not source_names:
        raise ControlledRunError("contract does not identify executed source files")
    for name in source_names:
        if name not in files:
            raise ControlledRunError(f"contract source file is unresolved: {name}")
        path = Path(files[name]["path"])
        try:
            relative = path.relative_to(repo)
        except ValueError as error:
            raise ControlledRunError(
                f"executed source file is outside repository: {name}"
            ) from error
        try:
            _git(repo, "ls-files", "--error-unmatch", str(relative))
        except subprocess.CalledProcessError as error:
            raise ControlledRunError(f"executed source file is untracked: {name}") from error


def preflight(
    contract_path: Path,
    repo: Path,
    invocation: Mapping[str, Any],
    resolved_files: Mapping[str, Path],
    argv: Sequence[str],
    cwd: Path,
    environment_variables: Mapping[str, str],
) -> dict[str, Any]:
    """Validate a frozen contract and return a pre-sampling receipt."""
    if invocation.get("seed") is None:
        raise ControlledRunError("controlled fit requires an explicit seed")
    seed = invocation["seed"]
    if not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ControlledRunError("controlled fit seed must be an unsigned 64-bit integer")

    repo = Path(repo).resolve()
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "flits-controlled-joint-fit-contract/v1":
        raise ControlledRunError("unsupported controlled-fit contract schema")
    if contract.get("burst") != invocation.get("burst"):
        raise ControlledRunError("burst identity does not match contract")
    resolved_identity_digest = contract.get("resolved_fit_identity_sha256")
    if not isinstance(resolved_identity_digest, str) or len(resolved_identity_digest) != 64:
        raise ControlledRunError("contract lacks a resolved fit identity hash")

    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ControlledRunError(f"source worktree is dirty: {status.splitlines()[0]}")
    revision = _git(repo, "rev-parse", "HEAD")
    if revision != contract.get("source_revision"):
        raise ControlledRunError("source revision does not match contract")

    observed_configuration = dict(invocation)
    if observed_configuration != contract.get("fit_configuration"):
        raise ControlledRunError("fit configuration does not match contract")

    expected_files = contract.get("files", {})
    if set(resolved_files) != set(expected_files):
        raise ControlledRunError("resolved file set does not match contract")
    observed_files = {}
    for name, raw_path in resolved_files.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ControlledRunError(f"missing file {name}: {path}")
        expected = expected_files[name]
        if str(path) != expected.get("path"):
            raise ControlledRunError(f"path mismatch for {name}")
        digest = sha256(path)
        if digest != expected.get("sha256"):
            raise ControlledRunError(f"hash mismatch for {name}")
        observed_files[name] = {"path": str(path), "sha256": digest}

    source_names = contract.get("executed_source_files", [])
    if set(source_names) != CONTROLLED_SOURCE_NAMES:
        raise ControlledRunError("contract must identify the canonical executed source set")
    _verify_source_files(repo, source_names, observed_files)

    observed_command = {
        "argv": list(argv),
        "working_directory": str(Path(cwd).resolve()),
    }
    if observed_command != contract.get("command"):
        raise ControlledRunError("command or working directory does not match contract")

    observed_environment_variables = dict(environment_variables)
    if observed_environment_variables != contract.get("environment_variables"):
        raise ControlledRunError("processing environment does not match contract")
    environment_lock = Path(resolved_files["environment_lock"])
    environment = environment_identity(environment_lock)
    if environment["identity_sha256"] != contract.get("environment_identity_sha256"):
        raise ControlledRunError("runtime environment identity does not match contract")

    return {
        "schema": "flits-controlled-joint-fit-receipt/v1",
        "contract": {
            "path": str(contract_path),
            "sha256": sha256(contract_path),
        },
        "burst": contract.get("burst"),
        "preflight_passed": True,
        "outputs_complete": False,
        "source": {
            "repository": str(repo),
            "revision": revision,
            "clean_worktree": True,
        },
        "fit_configuration": observed_configuration,
        "files": observed_files,
        "executed_source_files": list(source_names),
        "command": observed_command,
        "processing_environment_variables": observed_environment_variables,
        "environment": environment,
        "required_post_fit_guards": DEPRECATED_ZACH_GUARDS,
        "outputs": {},
    }


def reverify_preflight(receipt: Mapping[str, Any]) -> None:
    """Recheck source, environment, and every input after preprocessing."""
    repo = Path(receipt["source"]["repository"])
    contract = Path(receipt["contract"]["path"])
    if not contract.is_file() or sha256(contract) != receipt["contract"]["sha256"]:
        raise ControlledRunError("contract changed after preflight")
    frozen = json.loads(contract.read_text(encoding="utf-8"))
    if receipt.get("fit_configuration") != frozen.get("fit_configuration"):
        raise ControlledRunError("receipt fit configuration differs from contract")
    if receipt.get("command") != frozen.get("command"):
        raise ControlledRunError("receipt command differs from contract")
    if receipt.get("processing_environment_variables") != frozen.get("environment_variables"):
        raise ControlledRunError("receipt processing environment differs from contract")
    if receipt.get("executed_source_files") != frozen.get("executed_source_files"):
        raise ControlledRunError("receipt executed-source list differs from contract")
    if receipt.get("files") != frozen.get("files"):
        raise ControlledRunError("receipt file identities differ from contract")
    if receipt.get("source", {}).get("revision") != frozen.get("source_revision"):
        raise ControlledRunError("receipt source revision differs from contract")
    if receipt.get("environment", {}).get("identity_sha256") != frozen.get(
        "environment_identity_sha256"
    ):
        raise ControlledRunError("receipt runtime environment differs from contract")
    resolved = receipt.get("resolved_fit_identity")
    resolved_digest = receipt.get("resolved_fit_identity_sha256")
    if resolved is not None and (
        identity_sha256(resolved) != resolved_digest
        or resolved_digest != frozen.get("resolved_fit_identity_sha256")
    ):
        raise ControlledRunError("receipt resolved fit identity differs from contract")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ControlledRunError("source changed after preflight")
    if _git(repo, "rev-parse", "HEAD") != receipt["source"]["revision"]:
        raise ControlledRunError("source revision changed after preflight")
    for name, identity in receipt["files"].items():
        path = Path(identity["path"])
        if not path.is_file() or sha256(path) != identity["sha256"]:
            raise ControlledRunError(f"file changed after preflight: {name}")
    lock = Path(receipt["files"]["environment_lock"]["path"])
    if environment_identity(lock)["identity_sha256"] != receipt["environment"]["identity_sha256"]:
        raise ControlledRunError("runtime environment changed after preflight")


def finalize_receipt(
    receipt_path: Path,
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Append exact output identities to a passed receipt and rewrite it atomically."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not receipt.get("preflight_passed"):
        raise ControlledRunError("cannot finalize a receipt that failed preflight")
    if not receipt.get("post_preparation_reverification_passed"):
        raise ControlledRunError("resolved fit identity was not verified before sampling")
    resolved = receipt.get("resolved_fit_identity")
    resolved_digest = receipt.get("resolved_fit_identity_sha256")
    contract = json.loads(Path(receipt["contract"]["path"]).read_text(encoding="utf-8"))
    if (
        not isinstance(resolved, dict)
        or identity_sha256(resolved) != resolved_digest
        or resolved_digest != contract.get("resolved_fit_identity_sha256")
    ):
        raise ControlledRunError("resolved fit identity object is missing or invalid")
    reverify_preflight(receipt)
    if not outputs:
        raise ControlledRunError("no outputs supplied for receipt finalization")

    identities = dict(receipt.get("outputs", {}))
    for name, identity in identities.items():
        path = Path(identity["path"])
        if not path.is_file() or sha256(path) != identity["sha256"]:
            raise ControlledRunError(f"previously recorded output changed: {name}")
        if (
            "scientific_sha256" in identity
            and canonical_npz_sha256(path) != identity["scientific_sha256"]
        ):
            raise ControlledRunError(f"scientific output content changed: {name}")
    for name, raw_path in outputs.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ControlledRunError(f"missing output {name}: {path}")
        if (
            name in CANONICAL_OUTPUT_SUFFIXES
            and path.suffix.lower() != (CANONICAL_OUTPUT_SUFFIXES[name])
        ):
            raise ControlledRunError(f"wrong file type for canonical output {name}")
        identity = {"path": str(path), "sha256": sha256(path)}
        if path.suffix.lower() == ".npz":
            identity["scientific_sha256"] = canonical_npz_sha256(path)
        identities[name] = identity
    receipt["outputs"] = identities
    receipt["outputs_hashed"] = True
    canonical_paths = [identities[name]["path"] for name in CANONICAL_OUTPUTS if name in identities]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ControlledRunError("canonical output roles must use distinct files")
    receipt["outputs_complete"] = CANONICAL_OUTPUTS.issubset(identities)
    if receipt["outputs_complete"]:
        _validate_complete_output_packet(receipt, identities)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    return receipt
