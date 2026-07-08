#!/usr/bin/env python3
"""Fresh DSA ACF Lorentzian scintillation-bandwidth fits.

This driver intentionally bypasses legacy YAML ``stored_fits`` and any rescued
``acf_results.pkl`` products. It recomputes ACFs from the staged DSA `.npz`
dynamic spectra, then applies the existing 1/2/3-Lorentzian BIC + nested-F
selector to each sub-band ACF.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Use the checked-out pipeline source for this analysis, even if another FLITS
# checkout is installed editable in the active Python environment. Disable numba
# JIT before importing scintillation modules; old cross-checkout numba caches can
# try to resurrect modules by the stale top-level name ``scint_analysis``.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from scintillation.scint_analysis import analysis  # noqa: E402
from scintillation.scint_analysis import config as config_mod  # noqa: E402
from scintillation.scint_analysis.pipeline import ScintillationAnalysis  # noqa: E402
from scintillation.scint_analysis.revalidation import (  # noqa: E402
    compare_lorentzian_components,
)

BURSTS = [
    "casey",
    "chromatica",
    "freya",
    "hamilton",
    "isha",
    "johndoeII",
    "mahi",
    "oran",
    "phineas",
    "whitney",
    "wilhelm",
    "zach",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _config_for_fresh_acf(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)

    # Keep the checked-in science choices, but remove fit/result reuse knobs from
    # this generated run configuration.
    cfg.pop("stored_fits", None)

    pipe_opts = cfg.setdefault("pipeline_options", {})
    pipe_opts["force_recalc"] = True
    pipe_opts["save_intermediate_steps"] = False
    pipe_opts["halt_after_acf"] = True
    pipe_opts["cache_directory"] = str(output_dir / "cache" / cfg.get("burst_id", "unknown"))
    pipe_opts.setdefault("log_level", "INFO")
    pipe_opts["diagnostic_plots"] = {"enable": False}

    analysis_cfg = cfg.setdefault("analysis", {})
    noise_cfg = analysis_cfg.setdefault("noise", {})
    noise_cfg.setdefault("disable", False)
    # The Lorentzian-only selector does not consume the MC template. Disabling it
    # keeps this pass deterministic and much faster without changing the ACF.
    noise_cfg["disable_template"] = True

    analysis_cfg.setdefault("fit_2d", {})["enable"] = False
    return cfg


def _slice_fit_window(
    lags: np.ndarray, acf: np.ndarray, err: np.ndarray | None, fit_range_mhz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mask = np.isfinite(lags) & np.isfinite(acf) & (np.abs(lags) <= fit_range_mhz)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)
    sliced_err = err[mask] if err is not None else None
    return lags[mask], acf[mask], sliced_err


def _plurality_n(per_subband: list[dict[str, Any]]) -> int:
    counts = Counter(int(v.get("n_preferred", 1)) for v in per_subband)
    if not counts:
        return 1
    top = max(counts.values())
    return min(n for n, count in counts.items() if count == top)


def _selected_fit(verdict: dict[str, Any]) -> dict[str, Any]:
    n_pref = int(verdict.get("n_preferred", 1))
    for fit in verdict.get("fits", []):
        if int(fit.get("n", -1)) == n_pref:
            return fit
    return {"n": n_pref, "success": False, "components": []}


def _component_quality_flags(component: dict[str, Any], *, fit_range_mhz: float) -> list[str]:
    flags = []
    dnu = float(component.get("dnu_mhz", np.nan))
    dnu_err = float(component.get("dnu_err", np.nan))
    mod = float(component.get("m", np.nan))
    mod_err = float(component.get("m_err", np.nan))

    if not np.isfinite(dnu) or dnu <= 0:
        flags.append("invalid_dnu")
    elif dnu > fit_range_mhz:
        flags.append("dnu_exceeds_fit_window")

    if np.isfinite(dnu) and dnu > 0 and np.isfinite(dnu_err) and dnu_err / dnu > 1.0:
        flags.append("fractional_dnu_err_gt_1")
    if np.isfinite(mod) and mod > 3.0:
        flags.append("modulation_gt_3")
    if np.isfinite(mod) and mod > 0 and np.isfinite(mod_err) and mod_err / mod > 1.0:
        flags.append("fractional_mod_err_gt_1")

    return flags


def _fit_one_burst(
    config_path: Path,
    *,
    output_dir: Path,
    max_components: int,
) -> dict[str, Any]:
    loaded = config_mod.load_config(config_path)
    cfg = _config_for_fresh_acf(loaded, output_dir=output_dir)
    burst = str(cfg.get("burst_id", config_path.stem.removesuffix("_dsa")))

    analysis.clear_noise_acf_cache()
    pipe = ScintillationAnalysis(cfg)
    pipe.run()
    acf_results = pipe.acf_results
    if not acf_results or not acf_results.get("subband_acfs"):
        raise RuntimeError(f"{burst}: no ACF results produced")

    fit_cfg = cfg.get("analysis", {}).get("fitting", {})
    configured_fit_range = float(fit_cfg.get("fit_lagrange_mhz", 45.0))

    subbands = []
    for i, acf in enumerate(acf_results["subband_acfs"]):
        lags = np.asarray(acf_results["subband_lags_mhz"][i], dtype=float)
        acf_arr = np.asarray(acf, dtype=float)
        err_values = acf_results.get("subband_acfs_err")
        err = np.asarray(err_values[i], dtype=float) if err_values else None

        center_freq = float(acf_results["subband_center_freqs_mhz"][i])
        chan_width = float(acf_results["subband_channel_widths_mhz"][i])
        n_chan = int(acf_results["subband_num_channels"][i])
        subband_bw = n_chan * chan_width
        fit_range = min(configured_fit_range, subband_bw / 2.0)
        fit_lags, fit_acf, fit_err = _slice_fit_window(lags, acf_arr, err, fit_range)
        verdict = compare_lorentzian_components(
            fit_lags,
            fit_acf,
            max_components=max_components,
            acf_err=fit_err,
        )
        fit = _selected_fit(verdict)
        components = sorted(
            fit.get("components", []),
            key=lambda c: float(c.get("dnu_mhz", np.inf)),
        )
        for component in components:
            component["quality_flags"] = _component_quality_flags(
                component,
                fit_range_mhz=fit_range,
            )

        subbands.append(
            {
                "index": i,
                "center_freq_mhz": center_freq,
                "channel_width_mhz": chan_width,
                "num_channels": n_chan,
                "fit_range_mhz": fit_range,
                "n_fit_points": int(np.sum(fit_lags > 0)),
                "n_preferred": int(verdict.get("n_preferred", 1)),
                "criterion": verdict.get("criterion"),
                "delta_bic": verdict.get("delta_bic", {}),
                "f_test_p": verdict.get("f_test", {}),
                "selected_bic": fit.get("bic"),
                "selected_redchi": fit.get("redchi"),
                "selected_components": components,
                "all_fit_summaries": [
                    {
                        "n": int(f.get("n", 0)),
                        "success": bool(f.get("success", False)),
                        "bic": f.get("bic"),
                        "aic": f.get("aic"),
                        "chi2": f.get("chi2"),
                        "redchi": f.get("redchi"),
                        "n_params": f.get("n_params"),
                        "ndata": f.get("ndata"),
                        "components": sorted(
                            f.get("components", []),
                            key=lambda c: float(c.get("dnu_mhz", np.inf)),
                        ),
                    }
                    for f in verdict.get("fits", [])
                ],
            }
        )

    component_bands: dict[int, list[float]] = defaultdict(list)
    usable_component_bands: dict[int, list[float]] = defaultdict(list)
    for subband in subbands:
        for comp_idx, comp in enumerate(subband["selected_components"], start=1):
            dnu = comp.get("dnu_mhz")
            if dnu is not None and np.isfinite(float(dnu)):
                component_bands[comp_idx].append(float(dnu))
                if not comp.get("quality_flags"):
                    usable_component_bands[comp_idx].append(float(dnu))

    return {
        "burst": burst,
        "config_path": str(config_path),
        "input_data_path": cfg.get("input_data_path"),
        "fit_lagrange_mhz": configured_fit_range,
        "max_components": max_components,
        "num_subbands": len(subbands),
        "burst_preferred_n": _plurality_n(subbands),
        "n_per_subband": [s["n_preferred"] for s in subbands],
        "component_median_dnu_mhz": {
            str(k): float(np.nanmedian(v)) for k, v in sorted(component_bands.items())
        },
        "component_usable_median_dnu_mhz": {
            str(k): float(np.nanmedian(v))
            for k, v in sorted(usable_component_bands.items())
            if v
        },
        "subbands": subbands,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "burst",
        "subband",
        "center_freq_mhz",
        "n_preferred",
        "component",
        "dnu_mhz",
        "dnu_err_mhz",
        "modulation_m",
        "modulation_err",
        "fit_range_mhz",
        "selected_bic",
        "selected_redchi",
        "quality_flags",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(results: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# DSA Lorentzian ACF Fit Summary",
        "",
        "Fresh DSA ACFs were computed from the staged `.npz` dynamic spectra. Each sub-band",
        "was fit with 1, 2, and 3 Lorentzian components; adding a component required both",
        "strong BIC improvement and the nested-F test threshold in the existing",
        "`compare_lorentzian_components` selector.",
        "",
        "## Burst Overview",
        "",
        "| burst | subbands | preferred n by subband | plurality n | median dnu by component (MHz) |",
        "|---|---:|---|---:|---|",
    ]
    for result in results:
        usable = result.get("component_usable_median_dnu_mhz", {})
        if usable:
            med = ", ".join(f"c{k}={v:.4g}" for k, v in usable.items())
        else:
            med = "no unflagged components"
        lines.append(
            "| {burst} | {num_subbands} | {n_per_subband} | {burst_preferred_n} | {med} |".format(
                med=med or "-",
                **result,
            )
        )

    lines.extend(
        [
            "",
            "## Component Rows",
            "",
            "| burst | subband | freq MHz | n | component | dnu MHz | dnu err | m | redchi | flags |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {burst} | {subband} | {center_freq_mhz:.3f} | {n_preferred} | {component} | "
            "{dnu_mhz:.6g} | {dnu_err_mhz:.3g} | {modulation_m:.4g} | "
            "{selected_redchi:.4g} | {quality_flags} |".format(**row)
        )

    path.write_text("\n".join(lines) + "\n")


def _flatten_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for subband in result["subbands"]:
            for comp_idx, comp in enumerate(subband["selected_components"], start=1):
                rows.append(
                    {
                        "burst": result["burst"],
                        "subband": subband["index"],
                        "center_freq_mhz": float(subband["center_freq_mhz"]),
                        "n_preferred": int(subband["n_preferred"]),
                        "component": comp_idx,
                        "dnu_mhz": float(comp.get("dnu_mhz", np.nan)),
                        "dnu_err_mhz": float(comp.get("dnu_err", np.nan)),
                        "modulation_m": float(comp.get("m", np.nan)),
                        "modulation_err": float(comp.get("m_err", np.nan)),
                        "fit_range_mhz": float(subband["fit_range_mhz"]),
                        "selected_bic": float(subband.get("selected_bic", np.nan)),
                        "selected_redchi": float(subband.get("selected_redchi", np.nan)),
                        "quality_flags": ";".join(comp.get("quality_flags", [])),
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for JSON/CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--flits-root",
        type=Path,
        default=Path(os.environ.get("FLITS_ROOT", Path.home() / "Data/Faber2026/dsa110")),
        help="Root containing scintillation/data/{burst}.npz.",
    )
    parser.add_argument("--max-components", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--bursts", nargs="*", default=BURSTS, help="Burst nicknames to run.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record failed bursts and continue instead of raising immediately.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ["FLITS_ROOT"] = str(args.flits_root.expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    for burst in args.bursts:
        config_path = Path("scintillation/configs/bursts") / f"{burst}_dsa.yaml"
        logging.info("Running %s from %s", burst, config_path)
        try:
            result = _fit_one_burst(
                config_path,
                output_dir=args.output_dir,
                max_components=args.max_components,
            )
        except Exception as exc:
            logging.exception("%s failed", burst)
            failures.append({"burst": burst, "error": str(exc)})
            if not args.keep_going:
                raise
        else:
            results.append(result)
            burst_path = args.output_dir / f"{burst}_lorentzian_fits.json"
            burst_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True))

    rows = _flatten_rows(results)
    all_results = {
        "run": {
            "flits_root": os.environ["FLITS_ROOT"],
            "max_components": args.max_components,
            "bursts_requested": args.bursts,
            "n_success": len(results),
            "n_failure": len(failures),
            "failures": failures,
            "notes": (
                "Fresh DSA ACFs from npz; YAML stored_fits and pkl ACF products are not read. "
                "Pipeline caches, diagnostic plots, MC noise templates, and 2D fits are disabled."
            ),
        },
        "results": results,
    }
    (args.output_dir / "dsa_lorentzian_fits.json").write_text(
        json.dumps(_jsonable(all_results), indent=2, sort_keys=True)
    )
    _write_csv(rows, args.output_dir / "dsa_lorentzian_components.csv")
    _write_markdown(results, rows, args.output_dir / "DSA_LORENTZIAN_FITS.md")

    if failures:
        logging.error("Completed with %d failures", len(failures))
        return 1
    logging.info("Completed %d bursts; wrote %s", len(results), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
