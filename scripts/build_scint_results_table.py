#!/usr/bin/env python3
"""Build publication-facing tables from CHIME scintillation campaign JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ESTIMATORS = ("odr_logspace", "loglog_unweighted", "joint_2d")


def finite_float(value: Any) -> float | None:
    """Return a finite float, excluding booleans, or None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers with JSON null."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def component_payloads(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    components = data.get("components")
    if not isinstance(components, dict):
        return {}
    return {name: value for name, value in components.items() if isinstance(value, dict)}


def select_component(data: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Select the component carrying the scaling fit, then the richest sub-band set."""
    components = component_payloads(data)
    for name, component in components.items():
        if isinstance(component.get("gamma_scaling"), dict):
            return name, component

    def measurement_count(item: tuple[str, dict[str, Any]]) -> int:
        measurements = item[1].get("subband_measurements")
        return len(measurements) if isinstance(measurements, list) else 0

    if components:
        name, component = max(components.items(), key=measurement_count)
        return name, component
    return None, {}


def valid_subbands(component: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    measurements = component.get("subband_measurements")
    if not isinstance(measurements, list):
        return 0, []
    cleaned = [json_safe(item) for item in measurements if isinstance(item, dict)]
    count = sum(
        1
        for item in measurements
        if isinstance(item, dict)
        and (bandwidth := finite_float(item.get("bw"))) is not None
        and bandwidth > 0
    )
    return count, cleaned


def modulation_frequency(data: dict[str, Any], component_name: str | None) -> list[dict[str, Any]]:
    root = data.get("modulation_index_frequency")
    if not isinstance(root, dict):
        return []
    acf = root.get("acf_amplitude")
    components = acf.get("components") if isinstance(acf, dict) else None
    if not isinstance(components, dict):
        return []

    selected = components.get(component_name) if component_name else None
    if not isinstance(selected, list):
        candidates = [value for value in components.values() if isinstance(value, list)]
        selected = max(candidates, key=len, default=[])
    return [
        {
            "freq_mhz": finite_float(item.get("freq_mhz")),
            "mod": finite_float(item.get("m", item.get("mod"))),
            "mod_err": finite_float(item.get("m_err", item.get("mod_err"))),
        }
        for item in selected
        if isinstance(item, dict)
    ]


def find_time_modulation(data: dict[str, Any]) -> dict[str, Any] | None:
    """Find the producer's time-modulation block or a legacy intra-pulse variant."""
    direct = data.get("modulation_index_time")
    if isinstance(direct, dict):
        return direct

    matches: dict[str, Any] = {}

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = key.lower()
                next_path = (*path, key)
                if isinstance(item, (dict, list)) and (
                    "intra" in lowered or lowered in {"m_time", "mod_time"}
                ):
                    matches[".".join(next_path)] = item
                walk(item, next_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(data)
    return matches or None


def has_finite_time_modulation(value: Any, parent_key: str = "") -> bool:
    if isinstance(value, dict):
        return any(
            has_finite_time_modulation(item, str(key).lower()) for key, item in value.items()
        )
    if isinstance(value, list):
        if parent_key in {"m", "mod", "m_time", "mod_time"}:
            return any(finite_float(item) is not None for item in value)
        return any(has_finite_time_modulation(item, parent_key) for item in value)
    return parent_key in {"m", "mod", "m_time", "mod_time"} and finite_float(value) is not None


def estimators_agree(gamma: dict[str, Any]) -> bool:
    odr = gamma.get("odr_logspace")
    loglog = gamma.get("loglog_unweighted")
    if not isinstance(odr, dict) or not isinstance(loglog, dict):
        return False
    alpha_odr = finite_float(odr.get("alpha"))
    error_odr = finite_float(odr.get("alpha_err"))
    alpha_loglog = finite_float(loglog.get("alpha"))
    return (
        alpha_odr is not None
        and error_odr is not None
        and alpha_loglog is not None
        and abs(alpha_odr - alpha_loglog) <= 2.0 * error_odr
    )


def gamma_verdict(
    name: str, estimator: dict[str, Any], n_valid_subbands: int, agreement: bool
) -> tuple[str, list[str]]:
    alpha = finite_float(estimator.get("alpha"))
    alpha_err = finite_float(estimator.get("alpha_err"))
    reasons: list[str] = []

    if n_valid_subbands < 3:
        reasons.append("fewer_than_3_valid_subbands")
    if alpha is None:
        reasons.append("missing_or_nonfinite_alpha")
    if name in {"odr_logspace", "joint_2d"} and alpha_err is None:
        reasons.append("missing_or_nonfinite_alpha_err")
    if alpha is not None and abs(alpha) > 20:
        reasons.append("unphysical_alpha")
    if alpha_err is not None and alpha_err < 1e-9:
        reasons.append("degenerate_alpha_err")
    relative_error = None
    if alpha is not None and alpha_err is not None:
        relative_error = math.inf if alpha == 0 else alpha_err / abs(alpha)
        if relative_error > 1.0:
            reasons.append("relative_error_gt_1")
    if (
        name == "joint_2d"
        and alpha is not None
        and (abs(alpha - 1.0) <= 1e-3 or abs(alpha - 8.0) <= 1e-3)
    ):
        reasons.append("optimizer_bound_saturated")

    if reasons:
        return "FAIL", reasons
    if (
        relative_error is not None
        and relative_error <= 0.3
        and alpha is not None
        and 2.0 <= alpha <= 8.0
        and agreement
    ):
        return "PASS", []
    marginal_reasons = []
    if relative_error is None:
        marginal_reasons.append("relative_error_unavailable")
    elif relative_error > 0.3:
        marginal_reasons.append("relative_error_gt_0.3")
    if alpha is not None and not 2.0 <= alpha <= 8.0:
        marginal_reasons.append("alpha_outside_2_to_8")
    if not agreement:
        marginal_reasons.append("odr_loglog_disagree")
    return "MARGINAL", marginal_reasons


def extract_gamma(component: dict[str, Any], n_valid_subbands: int) -> dict[str, Any]:
    raw = component.get("gamma_scaling")
    gamma = raw if isinstance(raw, dict) else {}
    agreement = estimators_agree(gamma)
    extracted: dict[str, Any] = {}
    for name in ESTIMATORS:
        source = gamma.get(name)
        estimator = source if isinstance(source, dict) else {}
        quality_flag, failed_checks = gamma_verdict(name, estimator, n_valid_subbands, agreement)
        extracted[name] = {
            "alpha": finite_float(estimator.get("alpha")),
            "alpha_err": finite_float(estimator.get("alpha_err")),
            "bw_at_ref_mhz": finite_float(estimator.get("bw_at_ref_mhz")),
            "bw_at_ref_mhz_err": finite_float(estimator.get("bw_at_ref_mhz_err")),
            "method": estimator.get("method"),
            "status": estimator.get("status"),
            "quality_flag": quality_flag,
            "failed_checks": failed_checks,
        }
    extracted["odr_loglog_agree"] = agreement
    return extracted


def combined_verdict(measurement_status: str | None, odr_verdict: str) -> str:
    if measurement_status == "diagnostic_only":
        return "DIAGNOSTIC"
    if measurement_status == "measurement" and odr_verdict == "PASS":
        return "PASS"
    return "FAIL" if odr_verdict == "FAIL" else "MARGINAL"


def extract_record(path: Path, campaign_root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(campaign_root)
    job = relative_path.parent.as_posix()
    record: dict[str, Any] = {
        "burst": relative_path.parent.name or path.name.removesuffix("_analysis_results.json"),
        "source_burst": path.name.removesuffix("_analysis_results.json"),
        "job": job,
        "source_file": relative_path.as_posix(),
        "parse_error": None,
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("top-level JSON value is not an object")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        record.update(
            {
                "parse_error": f"{type(error).__name__}: {error}",
                "measurement_status": {"status": None, "failed_checks": None},
                "selected_component": None,
                "n_valid_subbands": 0,
                "subband_measurements": [],
                "gamma_scaling": extract_gamma({}, 0),
                "modulation_index_frequency": [],
                "modulation_index_time": {"available": False, "results": None},
                "verdict": "FAIL",
            }
        )
        return record

    measurement = data.get("measurement_status")
    measurement = measurement if isinstance(measurement, dict) else {}
    status = measurement.get("status") if isinstance(measurement.get("status"), str) else None
    component_name, component = select_component(data)
    n_valid, subbands = valid_subbands(component)
    gamma = extract_gamma(component, n_valid)
    frequency_modulation = modulation_frequency(data, component_name)
    time_modulation = find_time_modulation(data)
    record.update(
        {
            "measurement_status": {
                "status": status,
                "failed_checks": json_safe(measurement.get("failed_checks")),
            },
            "selected_component": component_name,
            "n_valid_subbands": n_valid,
            "subband_measurements": subbands,
            "gamma_scaling": gamma,
            "modulation_index_frequency": frequency_modulation,
            "modulation_index_time": {
                "available": has_finite_time_modulation(time_modulation),
                "results": json_safe(time_modulation),
            },
            "verdict": combined_verdict(status, gamma["odr_logspace"]["quality_flag"]),
        }
    )
    return record


def significant(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    return "--" if number is None else f"{number:.{digits}g}"


def plus_minus(value: Any, error: Any, digits: int = 3) -> str:
    value_text = significant(value, digits)
    error_text = significant(error, digits)
    return value_text if error_text == "--" else f"{value_text} ± {error_text}"


def failed_checks_text(value: Any) -> str:
    if value in (None, [], {}):
        return "--"
    if isinstance(value, dict):
        failed = [str(key) for key, passed in value.items() if not passed]
        return ", ".join(failed or map(str, value))
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "--"
    return str(value)


def modulation_range(record: dict[str, Any]) -> str:
    values = [
        value
        for item in record.get("modulation_index_frequency", [])
        if (value := finite_float(item.get("mod"))) is not None
    ]
    if not values:
        return "--"
    if math.isclose(min(values), max(values), rel_tol=1e-12, abs_tol=0.0):
        return significant(values[0])
    return f"{significant(min(values))}–{significant(max(values))}"


def markdown_table(records: list[dict[str, Any]]) -> str:
    header = (
        "| burst | status | gates failed | n_subbands | alpha_odr ± err | alpha_loglog | "
        "alpha_2d | Dnu_d(600 MHz) kHz | m(nu) range | m(t) available | verdict |"
    )
    separator = "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---|"
    rows = [header, separator]
    for record in records:
        status = record["measurement_status"]["status"] or "--"
        failed = failed_checks_text(record["measurement_status"]["failed_checks"])
        gamma = record["gamma_scaling"]
        odr = gamma["odr_logspace"]
        bandwidth_khz = None if odr["bw_at_ref_mhz"] is None else 1000.0 * odr["bw_at_ref_mhz"]
        bandwidth_error_khz = (
            None if odr["bw_at_ref_mhz_err"] is None else 1000.0 * odr["bw_at_ref_mhz_err"]
        )
        cells = [
            record["burst"],
            status,
            failed,
            str(record["n_valid_subbands"]),
            plus_minus(odr["alpha"], odr["alpha_err"]),
            significant(gamma["loglog_unweighted"]["alpha"]),
            significant(gamma["joint_2d"]["alpha"]),
            plus_minus(bandwidth_khz, bandwidth_error_khz),
            modulation_range(record),
            "yes" if record["modulation_index_time"]["available"] else "no",
            record["verdict"],
        ]
        rows.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |")
    return "\n".join(rows) + "\n"


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def latex_value(value: Any, error: Any = None) -> str:
    value_text = significant(value)
    if value_text == "--":
        return r"\nodata"
    error_text = significant(error)
    return value_text if error_text == "--" else rf"${value_text} \pm {error_text}$"


def latex_table(records: list[dict[str, Any]]) -> str:
    rows = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Nickname & $\Delta\nu_d(600\,\mathrm{MHz})$ (kHz) & $\alpha_{\rm ODR}$ & $m(\nu)$ range & Verdict \\",
        r"\midrule",
    ]
    for record in records:
        odr = record["gamma_scaling"]["odr_logspace"]
        bandwidth = None if odr["bw_at_ref_mhz"] is None else 1000.0 * odr["bw_at_ref_mhz"]
        error = None if odr["bw_at_ref_mhz_err"] is None else 1000.0 * odr["bw_at_ref_mhz_err"]
        m_range = modulation_range(record)
        if m_range == "--":
            m_range = r"\nodata"
        rows.append(
            f"{latex_escape(record['burst'])} & {latex_value(bandwidth, error)} & "
            f"{latex_value(odr['alpha'], odr['alpha_err'])} & {m_range} & "
            f"{latex_escape(record['verdict'])} " + r"\\"
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(rows) + "\n"


def find_result_files(campaign_root: Path, include_sweep: bool) -> list[Path]:
    files = []
    for path in campaign_root.rglob("*_analysis_results.json"):
        relative_parts = path.relative_to(campaign_root).parts
        if not include_sweep and "freya_sweep" in relative_parts:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(campaign_root).as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="output directory (default: campaign root)")
    parser.add_argument("--include-sweep", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    output_dir = (args.out or campaign_root).expanduser().resolve()
    if not campaign_root.is_dir():
        raise SystemExit(f"campaign root is not a directory: {campaign_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        extract_record(path, campaign_root)
        for path in find_result_files(campaign_root, args.include_sweep)
    ]
    table = markdown_table(records)
    (output_dir / "results_table.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "results_table.md").write_text(table, encoding="utf-8")
    (output_dir / "results_table.tex").write_text(latex_table(records), encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
