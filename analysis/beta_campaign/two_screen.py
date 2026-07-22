#!/usr/bin/env python
"""Reproduce the archived beta-campaign two-screen calculation.

SCIENCE STATUS: INVALID FOR CURRENT USE. This historical calculation pairs
free-alpha joint-fit tau with DSA bandwidths. Current screen consistency
requires fixed-index tau_consistency refits. Outputs are forced into the archive.

For each fleet output {burst}_joint_fit{suffix}.json, join the campaign
tau_1ghz / beta-derived alpha with the DSA decorrelation bandwidth recorded in
scintillation/configs/bursts/{burst}_dsa.yaml stored_fits, and evaluate the
single-screen statistic tau x delta_nu = C1/(2*pi) via the batch pipeline's
check_tau_deltanu_consistency (flits/batch/analysis_logic.py -- the one source
of truth for C_THIN_SCREEN/C_EXTENDED/C_RANGE and the screen verdicts).

Delta_nu_d per burst: for every stored subband, take the lowest-BIC model,
read the narrowest Lorentzian-family HWHM (l_*_gamma / lg_*_gamma; a pure
Gaussian best model records no decorrelation scale and the subband is
skipped), scale from the subband reference frequency to 1400 MHz with the
burst's own campaign alpha (delta_nu ~ nu^alpha), then inverse-variance
average across subbands. CHIME configs carry no stored_fits, so this table is
DSA-band only; CHIME ACFs (4 bursts) would need a fresh ACF pass.

Writes two_screen_consistency.{json,md} to the dated archive. Bursts without
a campaign fit are listed missing.

  FLITS_RUNS=... conda run -n flits python analysis/beta_campaign/two_screen.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = (
    REPO / ".archive" / "outdated-science" / "2026-07-17" / "regenerated" /
    "analysis" / "beta_campaign"
)
RUNS = Path(os.environ.get("FLITS_RUNS", "/Users/jakobfaber/Developer/scratch/flits-local-runs"))
sys.path.insert(0, str(REPO))

from flits.batch.analysis_logic import check_tau_deltanu_consistency  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "grade_beta_campaign", Path(__file__).parent / "grade_beta_campaign.py"
)
_grade = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grade)
SUFFIX = _grade.SUFFIX

DNU_REF_MHZ = 1400.0  # match FREQ_DSA in analysis_logic (tau is scaled to the same nu)
_GAMMA_RE = re.compile(r"^(l|lg)_\d+_gamma$")


def _find_stored_fits(node):
    if isinstance(node, dict):
        if "stored_fits" in node:
            return node["stored_fits"]
        for v in node.values():
            hit = _find_stored_fits(v)
            if hit is not None:
                return hit
    return None


def dsa_delta_nu(burst: str, alpha: float) -> dict | None:
    """Weighted-mean DSA decorrelation bandwidth at DNU_REF_MHZ, with provenance."""
    scint_name = burst.removesuffix("_fine")
    path = REPO / "scintillation" / "configs" / "bursts" / f"{scint_name}_dsa.yaml"
    if not path.exists():
        return None
    stored = _find_stored_fits(yaml.safe_load(path.read_text()))
    if not stored:
        return None
    per_subband = []
    for sb, models in sorted(stored.items()):
        fitted = {k: v for k, v in models.items() if v.get("bic") is not None}
        if not fitted:
            continue
        model_name, blk = min(fitted.items(), key=lambda kv: kv[1]["bic"])
        ref = float(blk["reference_frequency_mhz"])
        gammas = []
        for key, p in blk["best_fit_params"].items():
            if not _GAMMA_RE.match(key):
                continue
            v, e = p.get("value"), p.get("stderr")
            # v > e (S/N >= 1): a gamma consistent with zero is not a measured
            # decorrelation scale -- whitney subband_1's BIC winner rails
            # lg_1_gamma at its 0.06 MHz lower bound (0.06 +/- 0.12) and would
            # otherwise dominate the inverse-variance mean via its small
            # absolute error.
            if v and e and math.isfinite(v) and math.isfinite(e) and e > 0 and v > e:
                gammas.append((float(v), float(e), key))
        if not gammas:
            continue  # pure-Gaussian winner: no decorrelation Lorentzian recorded
        v, e, key = min(gammas)
        scale = (DNU_REF_MHZ / ref) ** alpha
        per_subband.append(
            {
                "subband": sb,
                "model": model_name,
                "component": key,
                "ref_mhz": ref,
                "gamma_mhz": v,
                "gamma_err": e,
                "scaled_mhz": v * scale,
                "scaled_err": e * scale,
            }
        )
    if not per_subband:
        return None
    wsum = sum(1.0 / s["scaled_err"] ** 2 for s in per_subband)
    mean = sum(s["scaled_mhz"] / s["scaled_err"] ** 2 for s in per_subband) / wsum
    return {
        "delta_nu_dc": mean,
        "delta_nu_dc_err": 1.0 / math.sqrt(wsum),
        "config": str(path.relative_to(REPO)),
        "subbands": per_subband,
    }


def main() -> int:
    jdir = RUNS / "data/joint"
    rows, provenance, missing_fit, missing_dnu = [], {}, [], []
    for burst, suffix in SUFFIX.items():
        fp = jdir / f"{burst}_joint_fit{suffix}.json"
        if not fp.exists():
            missing_fit.append(burst)
            continue
        fit = json.loads(fp.read_text())
        if "beta" not in fit:  # legacy alpha-era file sharing the campaign name
            missing_fit.append(burst)
            continue
        tau = fit["tau_1ghz"]
        alpha = fit["alpha"]["median"]
        dnu = dsa_delta_nu(burst, alpha)
        if dnu is None:
            missing_dnu.append(burst)
            continue
        rows.append(
            {
                "burst_name": burst,
                "telescope": "dsa",
                "tau_1ghz": tau["median"],
                "tau_1ghz_err": 0.5 * (tau["err_minus"] + tau["err_plus"]),
                "delta_nu_dc": dnu["delta_nu_dc"],
                "delta_nu_dc_err": dnu["delta_nu_dc_err"],
                "alpha": alpha,
                "beta": fit["beta"]["median"],
            }
        )
        provenance[burst] = dnu
    results = check_tau_deltanu_consistency(pd.DataFrame(rows))

    out_rows = []
    for row, res in zip(rows, results, strict=True):
        out_rows.append(
            {
                **row,
                "tau_at_1p4ghz_ms": res.tau_at_scint_freq_ms,
                "tau_delta_nu_product": res.tau_delta_nu_product,
                "tau_delta_nu_product_err": res.tau_delta_nu_product_err,
                "implied_tau_from_dnu_ms": res.implied_tau_from_dnu_ms,
                "is_consistent": res.is_consistent,
                "quality_flag": res.quality_flag,
                "screen_verdict": res.screen_verdict,
                "interpretation": res.interpretation,
            }
        )
        print(
            f"[two-screen] {row['burst_name']}: product={res.tau_delta_nu_product:.3g} "
            f"-> {res.screen_verdict or res.quality_flag}",
            flush=True,
        )
    if missing_fit:
        print(f"[two-screen] MISSING campaign fits: {missing_fit}", flush=True)
    if missing_dnu:
        print(f"[two-screen] no stored DSA delta_nu_d: {missing_dnu}", flush=True)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = ARCHIVE_DIR / "two_screen_consistency.json"
    out.write_text(
        json.dumps(
            {
                "runs_root": str(RUNS),
                "dnu_reference_mhz": DNU_REF_MHZ,
                "missing_campaign_fit": missing_fit,
                "missing_delta_nu": missing_dnu,
                "rows": out_rows,
                "delta_nu_provenance": provenance,
            },
            indent=2,
        )
    )

    md = [
        "# Two-screen consistency (beta campaign, DSA band)",
        "",
        "product = tau(1.4 GHz)[s] x dnu_d(1.4 GHz)[Hz] = C1/(2*pi): one screen "
        "gives 0.159 (thin) ... 1.0 (extended); accepted range [0.1, 2]; "
        "product >> 2 => the resolved delta_nu_d samples a NEARER screen than "
        "the scattering one (wilhelm pattern).",
        "",
        "| burst | beta | tau_1GHz [ms] | tau_1.4GHz [ms] | dnu_d(1.4 GHz) [MHz] | tau.dnu | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in out_rows:
        md.append(
            f"| {r['burst_name']} | {r['beta']:.3f} | {r['tau_1ghz']:.4g} | "
            f"{r['tau_at_1p4ghz_ms']:.4g} | "
            f"{r['delta_nu_dc']:.3g} +/- {r['delta_nu_dc_err']:.2g} | "
            f"{r['tau_delta_nu_product']:.3g} | {r['screen_verdict'] or r['quality_flag']} |"
        )
    if missing_fit:
        md += ["", f"Missing campaign fits: {', '.join(missing_fit)}"]
    if missing_dnu:
        md += ["", f"No stored DSA delta_nu_d: {', '.join(missing_dnu)}"]
    md += [
        "",
        "CHIME-band delta_nu_d is not in stored_fits (needs a fresh ACF pass; 4 bursts have CHIME ACF pkls).",
    ]
    (ARCHIVE_DIR / "two_screen_consistency.md").write_text("\n".join(md) + "\n")
    print(f"[two-screen] wrote {out} (+.md); {len(out_rows)} bursts", flush=True)
    return 0 if not missing_fit else 1


if __name__ == "__main__":
    sys.exit(main())
