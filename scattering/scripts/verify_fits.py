#!/usr/bin/env python3
"""Verify and label a directory of scattering fit_results.json against the gate.

Applies the recalibrated acceptance criteria (see burstfit.classify_fit_quality)
plus evidence-based model selection to label each burst's fit:

  DETECTION    quality PASS and the scattering model (M3) is decisively preferred
               (dlogZ over the best non-scattering model > LOGZ_DECISIVE)
  UPPER-LIMIT  quality PASS but scattering not decisively preferred -> tau is an
               upper limit, not a measurement
  MARGINAL     chi2_red elevated/low (fit usable with caution)
  UNFITTABLE   quality FAIL (chi2_red catastrophic) -- not locked in

Usage:
    python scattering/scripts/verify_fits.py <dir_of_fit_results> [--csv out.csv]

The pure helpers (label_fit, _quality_from_chi2) are unit-tested in
test_verify_fits.py; this script needs only the standard library + the JSON
files, so it runs on the remote machine without importing the heavy pipeline.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

# Mirror of burstfit.classify_fit_quality thresholds (kept in sync deliberately
# so this verifier is dependency-free and runnable on the compute host).
CHI2_PASS_LO = 0.3
CHI2_PASS_HI = 1.5
CHI2_FAIL_HI = 10.0

# A log-evidence margin of 5 is the conventional "decisive" Bayes-factor cut
# (Jeffreys / Kass & Raftery 1995: 2 ln B > 10).
LOGZ_DECISIVE = 5.0

# Non-scattering models in the scan; M3 (and M3_multi) carry the scattering tau.
_SCATTERING_MODELS = {"M3", "M3_multi"}


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _quality_from_chi2(chi2_reduced: float) -> str:
    """PASS / MARGINAL / FAIL from reduced chi-squared (gate primary)."""
    c = _f(chi2_reduced)
    if not math.isfinite(c):
        return "FAIL"
    if c > CHI2_FAIL_HI:
        return "FAIL"
    if c > CHI2_PASS_HI or c < CHI2_PASS_LO:
        return "MARGINAL"
    return "PASS"


def _delta_logz_scattering(all_results: dict) -> float:
    """log Z(best scattering model) - log Z(best non-scattering model)."""
    if not isinstance(all_results, dict):
        return math.nan
    logz = {}
    for k, v in all_results.items():
        if isinstance(v, dict) and v.get("log_evidence") is not None:
            logz[k] = _f(v.get("log_evidence"))
    if not logz:
        return math.nan
    scat = [logz[k] for k in logz if k in _SCATTERING_MODELS]
    nonscat = [logz[k] for k in logz if k not in _SCATTERING_MODELS]
    if not scat or not nonscat:
        return math.nan
    return max(scat) - max(nonscat)


def label_fit(data: dict) -> dict:
    """Return a labeling summary for one parsed fit_results.json dict."""
    gof = data.get("goodness_of_fit", {}) or {}
    chi2 = _f(gof.get("chi2_reduced"))
    # Prefer the stored flag (written by the fixed pipeline); fall back to chi2.
    quality = gof.get("quality_flag") or _quality_from_chi2(chi2)

    best_model = data.get("best_model")
    pct = data.get("best_params_percentiles") or {}
    tau = err_minus = err_plus = math.nan
    if isinstance(pct, dict) and isinstance(pct.get("tau_1ghz"), dict):
        tau = _f(pct["tau_1ghz"].get("median"))
        err_minus = _f(pct["tau_1ghz"].get("err_minus"))
        err_plus = _f(pct["tau_1ghz"].get("err_plus"))
    if not math.isfinite(tau):
        tau = _f((data.get("best_params") or {}).get("tau_1ghz"))

    dlogz = _delta_logz_scattering(data.get("all_results", {}))

    if quality == "FAIL":
        label = "UNFITTABLE"
    elif quality == "MARGINAL":
        label = "MARGINAL"
    elif best_model in _SCATTERING_MODELS and math.isfinite(dlogz) and dlogz > LOGZ_DECISIVE:
        label = "DETECTION"
    else:
        label = "UPPER-LIMIT"

    has_unc = math.isfinite(err_minus) and math.isfinite(err_plus)
    return {
        "best_model": best_model,
        "tau_ms": tau,
        "tau_err_minus": err_minus,
        "tau_err_plus": err_plus,
        "chi2_reduced": chi2,
        "quality_flag": quality,
        "delta_logz_scat": dlogz,
        "label": label,
        "has_uncertainty": has_unc,
        "locked_in": label == "DETECTION" and has_unc,
    }


def summarize_dir(results_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*fit_results.json"))):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        burst = os.path.basename(path).split("_")[0]
        row = {"burst": burst, "file": os.path.basename(path)}
        row.update(label_fit(data))
        rows.append(row)
    return rows


def _fmt(x, spec):
    v = _f(x)
    return format(v, spec) if math.isfinite(v) else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="Directory containing *fit_results.json")
    ap.add_argument("--csv", help="Optional path to write a CSV summary")
    args = ap.parse_args()

    rows = summarize_dir(args.results_dir)
    if not rows:
        print(f"No fit_results.json found in {args.results_dir}")
        return

    hdr = f"{'burst':<12}{'model':<8}{'tau (ms)':<22}{'chi2_red':<10}{'quality':<10}{'dlogZ':<9}{'label':<12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tau_str = _fmt(r["tau_ms"], ".4g")
        if r["has_uncertainty"]:
            tau_str = f"{tau_str} (-{_fmt(r['tau_err_minus'], '.2g')}/+{_fmt(r['tau_err_plus'], '.2g')})"
        print(
            f"{r['burst']:<12}{str(r['best_model']):<8}{tau_str:<22}"
            f"{_fmt(r['chi2_reduced'], '.2f'):<10}{str(r['quality_flag']):<10}"
            f"{_fmt(r['delta_logz_scat'], '.1f'):<9}{r['label']:<12}"
        )

    n_lock = sum(1 for r in rows if r["locked_in"])
    print(f"\nLocked in (DETECTION + uncertainty): {n_lock}/{len(rows)}")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
