"""Gate the committed joint CHIME-DSA scattering fits against the runtime 3-level
FLITS fit-quality contract, reusing the authoritative classify_fit_quality.

Reads joint_json/{burst}_joint_fit.json + the paired {burst}_joint_ppc.json,
applies Level-1 physical bounds + prior-rail detection, Level-2 reduced-chi2
(classify_fit_quality, worst of the two bands), and Level-3 alpha-physics, then
writes joint_gate_verdicts.{csv,md} and a per-burst {burst}_joint_gate.json
(mirroring the *_fit_results.json field shape). These are standalone verdict
artifacts -- the fit-verify workflow globs *_fit_results.json, so it does NOT
auto-discover these *_joint_gate.json files. tau x dnu (Level-3) is not evaluable
here -- no per-sightline scintillation bandwidth -- so it is reported N/A.

Level-1 here is the physical-bounds + prior-rail subset of the contract; the
optimizer-convergence / Jacobian-conditioning Level-1 gates do not map to these
nested-sampling fits (no Jacobian; convergence is proxied by log_evidence_err).
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from scattering.scat_analysis.burstfit import classify_fit_quality  # noqa: E402

from scattering.scat_analysis.turbulence import alpha_from_beta

ALPHA_MIN, ALPHA_MAX = 1.0, 6.0  # Level-1 physical gate (ADR-0004 floor)
SUB_KOLM_LO = 2.0  # 1.0 <= alpha < SUB_KOLM_LO => sub-Kolmogorov (L3 MARGINAL)
TAU_MIN, TAU_MAX = 1e-4, 100.0  # ms
RAIL_EDGE = 0.1  # alpha within this of a prior bound => prior-railed (unconstrained)
KOLM_LO, KOLM_HI = 3.5, 4.5  # Level-3 PASS-consistent alpha window (Kolmogorov ref 4.0)
_RANK = {"FAIL": 2, "MARGINAL": 1, "PASS": 0}


def _worst(*flags):
    return max(flags, key=lambda f: _RANK[f])


def gate_one(burst, fit, ppc):
    """Classify one joint fit. ppc may be None (chi2 unknown -> Level-2 MARGINAL)."""
    if "beta" in fit:
        alpha = alpha_from_beta(fit["beta"]["median"])
        beta_med = fit["beta"]["median"]
        lo_b, hi_b = fit.get("beta_bounds", fit.get("alpha_bounds", (2.0, 6.0)))
        if "beta_bounds" in fit:
            rail = min(beta_med - lo_b, hi_b - beta_med) < RAIL_EDGE
        else:
            lo, hi = fit["alpha_bounds"]
            rail = min(alpha - lo, hi - alpha) < RAIL_EDGE
    else:
        alpha = fit["alpha"]["median"]
        beta_med = None
        lo, hi = fit["alpha_bounds"]
        rail = min(alpha - lo, hi - alpha) < RAIL_EDGE
    tau = fit["tau_1ghz"]["median"]

    # Level 1 -- physical bounds (any failure => FAIL, regardless of chi2)
    l1_fail = []
    if not (ALPHA_MIN <= alpha < ALPHA_MAX):
        l1_fail.append(f"alpha={alpha:.3f} outside [{ALPHA_MIN},{ALPHA_MAX})")
    if not (TAU_MIN < tau < TAU_MAX):
        l1_fail.append(f"tau={tau:.4g}ms outside ({TAU_MIN},{TAU_MAX})")
    l1 = "FAIL" if l1_fail else "PASS"

    # Level 2 -- reduced chi2 per band via the runtime classifier, worst of two.
    # A missing PPC, or a present-but-incomplete one (a chi2_* key absent), is
    # treated the same: chi2 unknown -> MARGINAL, rather than crashing the run.
    cc = ppc.get("chi2_chime") if ppc else None
    cd = ppc.get("chi2_dsa") if ppc else None
    if cc is None or cd is None:
        note = "no PPC (chi2 unknown)" if ppc is None else "incomplete PPC (chi2 missing)"
        l2, l2_note = "MARGINAL", note
    else:
        fc = classify_fit_quality(cc)[0]
        fd = classify_fit_quality(cd)[0]
        l2 = _worst(fc, fd)
        l2_note = f"chi2_C={cc:.2f}({fc}) chi2_D={cd:.2f}({fd})"

    # Level 3 -- alpha physics consistency. Physical bounds are Level 1's job;
    # Level 3 flags Kolmogorov proximity and the sub-Kolmogorov band (ADR-0004).
    # tau x dnu not evaluable here (no per-sightline dnu_d).
    if KOLM_LO <= alpha <= KOLM_HI:
        l3 = "PASS"
        l3_tag = None
    elif ALPHA_MIN <= alpha < SUB_KOLM_LO:
        l3 = "MARGINAL"
        l3_tag = "sub-Kolmogorov"
    else:
        l3 = "MARGINAL"
        l3_tag = "off-Kolmogorov"

    # A prior-railed alpha is unconstrained -- the fit pinned the prior bound
    # rather than measuring alpha -- so it is never better than MARGINAL even if
    # every level otherwise passes (consistent with excluding the rail-pinned
    # chromatica/freya/hamilton joint fits as non-measurements). And because the
    # Level-3 tau x dnu check is not evaluable here (no per-sightline dnu_d), the
    # contract is only partially verified, so the ceiling is capped at MARGINAL --
    # a fit is never certified PASS on an incomplete contract check.
    final = _worst(l1, l2, l3, "MARGINAL" if rail else "PASS", "MARGINAL")
    if l1_fail:
        reason = "L1 " + "; ".join(l1_fail)
    elif l2 == "FAIL":
        reason = "L2 catastrophic " + l2_note
    else:  # final is MARGINAL (capped); list whatever drove it beyond the cap
        bits = []
        if rail:
            bits.append(f"alpha prior-railed (within {RAIL_EDGE} of bound) -> unconstrained")
        if l2 == "MARGINAL":
            bits.append("L2 " + l2_note)
        if l3 == "MARGINAL":
            if ALPHA_MIN <= alpha < SUB_KOLM_LO:
                bits.append(f"L3 alpha={alpha:.2f} sub-Kolmogorov")
            else:
                bits.append(f"L3 alpha={alpha:.2f} off Kolmogorov")
        if not bits:  # passed every evaluable level; MARGINAL only because of the cap
            bits.append("L3 tau x dnu not evaluable (no dnu_d) -> capped at MARGINAL")
        reason = "; ".join(bits)

    return {
        "burst": burst,
        "alpha": alpha,
        "beta": beta_med,
        "tau": tau,
        "rail": rail,
        "chi2_chime": cc,
        "chi2_dsa": cd,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "tau_dnu": "N/A (no dnu_d)",
        "final": final,
        "reason": reason,
    }


def main():
    jdir = Path(__file__).resolve().parent / "joint_json"
    rows = []
    for fp in sorted(jdir.glob("*_joint_fit.json")):
        burst = fp.name.replace("_joint_fit.json", "")
        fit = json.loads(fp.read_text())
        ppc_fp = jdir / f"{burst}_joint_ppc.json"
        ppc = json.loads(ppc_fp.read_text()) if ppc_fp.exists() else None
        v = gate_one(burst, fit, ppc)
        rows.append(v)
        worst_chi2 = None
        if v["chi2_chime"] is not None and v["chi2_dsa"] is not None:
            worst_chi2 = max(v["chi2_chime"], v["chi2_dsa"])
        (jdir / f"{burst}_joint_gate.json").write_text(
            json.dumps(
                {
                    "burst": burst,
                    "chi2_reduced": worst_chi2,
                    "chi2_chime": v["chi2_chime"],
                    "chi2_dsa": v["chi2_dsa"],
                    "tau": v["tau"],
                    "alpha": v["alpha"],
                    "alpha_railed": v["rail"],
                    "quality_flag": v["final"],
                    "notes": [v["reason"]],
                },
                indent=2,
            )
        )

    cols = [
        "burst",
        "alpha",
        "rail",
        "tau",
        "chi2_chime",
        "chi2_dsa",
        "l1",
        "l2",
        "l3",
        "tau_dnu",
        "final",
        "reason",
    ]
    out = Path(__file__).resolve().parent
    with (out / "joint_gate_verdicts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = [
        "# Joint-fit quality verdicts (committed " + jdir.name + ")",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        md.append(
            "| "
            + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols)
            + " |"
        )
    n = {f: sum(r["final"] == f for r in rows) for f in ("PASS", "MARGINAL", "FAIL")}
    md += [
        "",
        f"**{len(rows)} fits**: {n['PASS']} PASS / {n['MARGINAL']} MARGINAL / {n['FAIL']} FAIL.",
    ]
    (out / "joint_gate_verdicts.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
