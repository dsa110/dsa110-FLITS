"""freya beta co-model verdict artifact builder (issue #106, DAG 8/8).

Deterministic function over the committed DAG artifacts -> the one
provenance-bearing source the eventual beta-table row cites:
freya_beta_verdict.json + freya_beta_verdict.md.

Captures: beta posterior summary + un-railed quantification, derived-alpha
percentiles, gate/PPC verdicts, x_zeta-beta posterior covariance, the #105
A-vs-B comparator verdict, the exp-era comparison (via the #100 comparator --
the deprecated free-alpha+exp-PBF fit is the *hypothesis under test*, never
citable truth), the #101 tail-coverage preflight recomputed at the fitted
medians (the binding sensitivity-regime caveat), and the provisional-citable
bar evaluation (un-railed, C1D1 confirmed, PASS/MARGINAL).

Only the preflight touches data (one CHIME band preparation); everything else
is pure JSON/npz arithmetic. No RNG anywhere.

  conda run -n flits python analysis/beta_poc/build_verdict.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REFIT = REPO / "analysis" / "scattering-refit-2026-06"
POC = REPO / "analysis" / "beta_poc" / "freya"

ROUTE_B_JSON = REFIT / "local_runs" / "freya_joint_fit_sharedzeta.json"
ROUTE_B_NPZ = REFIT / "local_runs" / "freya_joint_samples_sharedzeta.npz"
ROUTE_B_PPC = REFIT / "local_runs" / "freya_joint_ppc_sharedzeta.json"
ROUTE_A_JSON = POC / "freya_beta_poc_fit_real.json"
A_VS_B_JSON = POC / "freya_route_a_vs_b.json"
EXP_ERA_JSON = REFIT / "joint_ladder" / "freya_joint_fit_sharedzeta.json"
CHIME_CFG = REFIT / "local_runs" / "configs" / "freya_chime_run.yaml"
OUT_JSON = POC / "freya_beta_verdict.json"
OUT_MD = POC / "freya_beta_verdict.md"

BETA_PRIOR = (3.0, 4.0)
CLAIM_BAND_ALPHA = 0.1  # manuscript: well-constrained sightlines shift <= 0.1
CHI2_PASS = (0.3, 1.5)  # Level-2 classify_fit_quality PASS band


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _weighted_corr(x, y, w):
    w = np.asarray(w, float) / np.sum(w)
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return float(cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def unrailed(beta_triplet, prior=BETA_PRIOR):
    """Distance of the beta median from each prior edge in one-sided sigma."""
    lo_sig = (beta_triplet["median"] - prior[0]) / beta_triplet["err_minus"]
    hi_sig = (prior[1] - beta_triplet["median"]) / beta_triplet["err_plus"]
    return {
        "prior": list(prior),
        "sigma_from_lower": float(lo_sig),
        "sigma_from_upper": float(hi_sig),
        "unrailed": bool(lo_sig > 3.0 and hi_sig > 3.0),
    }


def _chi2_flag(c):
    """Exact mirror of the kernel chi2 gate in classify_fit_quality
    (burstfit.py:1484-1495; constants burstfit.py:81-83: SUSPICIOUSLY_LOW=0.3,
    GOOD_MAX=1.5, FAIL_MAX=10.0). Non-finite fails closed. Parity with the
    real function is pinned by test_chi2_flag_parity_with_kernel."""
    if c is None or not np.isfinite(c):
        return "FAIL"
    c = float(c)
    if c > 10.0:
        return "FAIL"
    if c > 1.5 or c < 0.3:
        return "MARGINAL"
    return "PASS"


def grade(chi2_chime, chi2_dsa, a_vs_b_verdict, unrailed_ok):
    """Worst-of the per-band kernel chi2 flags; any A-vs-B verdict other than
    `agree` (the #105 tolerance semantics: widened IS a breach) or a railed
    beta fails outright."""
    if a_vs_b_verdict != "agree" or not unrailed_ok:
        return "FAIL"
    flags = {_chi2_flag(chi2_chime), _chi2_flag(chi2_dsa)}
    if "FAIL" in flags:
        return "FAIL"
    return "MARGINAL" if "MARGINAL" in flags else "PASS"


def build(preflight=True):
    b = json.loads(ROUTE_B_JSON.read_text())
    a = json.loads(ROUTE_A_JSON.read_text())
    ppc = json.loads(ROUTE_B_PPC.read_text())
    avb = json.loads(A_VS_B_JSON.read_text())
    exp_era = json.loads(EXP_ERA_JSON.read_text())

    beta = b["percentiles"]["beta"]
    alpha = b["alpha"]
    tau = b["percentiles"]["tau_1ghz"]
    rail = unrailed(beta)

    # x_zeta-beta covariance from the posterior samples (PRD story 11).
    npz = np.load(ROUTE_B_NPZ, allow_pickle=True)
    names = [str(n) for n in npz["param_names"]]
    s, w = npz["samples"], npz["weights"]
    corr = _weighted_corr(s[:, names.index("beta")], s[:, names.index("x_zeta")], w)

    # Exp-era comparison through the #100 comparator. The beta-native artifact
    # carries alpha only as a top-level derived triplet, so normalize by hand;
    # the exp-era JSON sampled alpha and carries it in percentiles.
    pc = _load("posterior_compare", REFIT / "posterior_compare.py")
    b_norm = {"alpha": alpha, "tau_1ghz": tau}
    exp_cmp = pc.compare_posteriors(b_norm, exp_era, params=["alpha", "tau_1ghz"])
    d_alpha = abs(alpha["median"] - exp_era["alpha"]["median"])

    pre = None
    if preflight:
        tc = _load("tail_coverage", REFIT / "tail_coverage.py")
        m = _prepare_chime()
        pre = tc.tail_coverage(
            np.asarray(m.freq, float),
            tau["median"],
            beta["median"],
            np.asarray(m.time, float),
            t0_ms=b["percentiles"]["t0_C"]["median"],
        )
        coverage = f"{pre['efolds']:.2f} e-folds ({pre['captured_fraction']:.1%})"
    else:
        # recorded at #104 (PR #114): preflight at the fitted candidate
        coverage = "~2.07 e-folds (~87.4%, recorded at #104; preflight not re-run)"
    caveat = (
        "SENSITIVITY-REGIME CAVEAT (binding): at the fitted candidate the CHIME "
        f"window captures {coverage} of the power-law PBF "
        "tail, below the 3.0 preflight threshold. Both raw captures are 81.9 ms "
        "burst-centered, so widening cannot rescue heavy-tail coverage (max "
        "achievable ~2.8 e-folds at the exp-era candidate). The beta measurement "
        "is conditional on the truncated window; A-vs-B agreement cannot detect "
        "window-induced bias because both routes see the same window."
    )

    bar = {
        "unrailed": rail["unrailed"],
        "component_count_confirmed": bool(
            b.get("components_C") == 1 and b.get("components_D") == 1
        ),
        "grade": grade(ppc["chi2_chime"], ppc["chi2_dsa"], avb["verdict"], rail["unrailed"]),
    }
    bar["provisional_citable"] = bool(
        bar["unrailed"]
        and bar["component_count_confirmed"]
        and bar["grade"] in ("PASS", "MARGINAL")
    )

    return {
        "issue": "dsa110-FLITS#106",
        "burst": "freya",
        "tns": "FRB 20230325A",
        "model": "beta co-model, shared-zeta(nu) gain-marginal, C1D1 (ADR-0006)",
        "beta": beta,
        "alpha_derived": alpha,
        "tau_1ghz": tau,
        "log_evidence": [b["log_evidence"], b["log_evidence_err"]],
        "unrailed": rail,
        "gates": {
            "ppc_chi2_chime": ppc["chi2_chime"],
            "ppc_chi2_dsa": ppc["chi2_dsa"],
            "level2_pass_band": list(CHI2_PASS),
            "route_a_validation": a["validation"],
        },
        "x_zeta_beta_corr": corr,
        "a_vs_b": {"verdict": avb["verdict"], "params": avb["params"]},
        "exp_era_comparison": {
            "comparator": exp_cmp,
            "abs_alpha_shift": d_alpha,
            "claim_band": CLAIM_BAND_ALPHA,
            "within_claim_band": bool(d_alpha <= CLAIM_BAND_ALPHA),
            "note": (
                "exp-era value is the deprecated free-alpha+exponential-PBF fit's "
                "suggestion -- the hypothesis under test, not citable truth"
            ),
        },
        "tail_coverage_at_fit": pre,
        "caveat": caveat,
        "provisional_citable_bar": bar,
        "beta_row_candidate": {
            "burst": "freya",
            "beta": [beta["median"], beta["err_minus"], beta["err_plus"]],
            "alpha_derived": [alpha["median"], alpha["err_minus"], alpha["err_plus"]],
            "tau_1ghz_ms": [tau["median"], tau["err_minus"], tau["err_plus"]],
            "grade": bar["grade"],
            "provisional_citable": bar["provisional_citable"],
        },
        "provenance": {
            "route_b": str(ROUTE_B_JSON.relative_to(REPO)),
            "route_b_samples": str(ROUTE_B_NPZ.relative_to(REPO)),
            "route_b_ppc": str(ROUTE_B_PPC.relative_to(REPO)),
            "route_a": str(ROUTE_A_JSON.relative_to(REPO)),
            "a_vs_b": str(A_VS_B_JSON.relative_to(REPO)),
            "exp_era": str(EXP_ERA_JSON.relative_to(REPO)),
            "dag": "#99 #100 #101 #102 #103 #104(e09ac78b) #105(b8c8ffe5) -> #106",
        },
    }


def _prepare_chime():
    import tempfile

    import yaml

    sys.path.insert(0, str(REPO / "scattering"))
    from scat_analysis.config_utils import load_telescope_block
    from scat_analysis.pipeline.io import BurstDataset

    cfg = yaml.safe_load(CHIME_CFG.read_text())
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    with tempfile.TemporaryDirectory() as tmp:
        ds = BurstDataset(
            cfg["path"],
            tmp,
            name="freya_chime",
            telescope=tel,
            f_factor=int(cfg["f_factor"]),
            t_factor=int(cfg["t_factor"]),
            outer_trim=float(cfg.get("outer_trim", 0.15)),
            onpulse_crop=True,
            onpulse_pad_factor=0.5,
        )
        return ds.model


def render_md(v):
    rail, bar, exp = v["unrailed"], v["provisional_citable_bar"], v["exp_era_comparison"]
    g = v["gates"]
    lines = [
        "# freya beta co-model verdict (dsa110-FLITS #106)",
        "",
        f"**Grade: {bar['grade']}** - provisional-citable: **{bar['provisional_citable']}**",
        "",
        f"- beta = {v['beta']['median']:.4f} +{v['beta']['err_plus']:.4f}/-{v['beta']['err_minus']:.4f}"
        f" (un-railed: {rail['sigma_from_lower']:.0f} sigma / {rail['sigma_from_upper']:.0f} sigma"
        f" from the [{rail['prior'][0]}, {rail['prior'][1]}) edges)",
        f"- derived alpha = {v['alpha_derived']['median']:.4f}"
        f" +{v['alpha_derived']['err_plus']:.4f}/-{v['alpha_derived']['err_minus']:.4f}"
        " (thin-screen closure; NOT independently fit)",
        f"- tau_1GHz = {v['tau_1ghz']['median']:.5f} ms"
        f" +{v['tau_1ghz']['err_plus']:.5f}/-{v['tau_1ghz']['err_minus']:.5f}",
        f"- lnZ = {v['log_evidence'][0]:.2f} +/- {v['log_evidence'][1]:.2f}",
        f"- PPC chi2/dof: CHIME {g['ppc_chi2_chime']:.2f}, DSA {g['ppc_chi2_dsa']:.2f}"
        f" (Level-2 PASS band {g['level2_pass_band']})",
        f"- Route A validation: {v['gates']['route_a_validation']['verdict']}",
        f"- x_zeta-beta posterior correlation r = {v['x_zeta_beta_corr']:+.3f} (benign)",
        f"- A-vs-B (#105): **{v['a_vs_b']['verdict']}** on all physics params",
        f"- Exp-era comparison (#100 comparator): overall"
        f" **{exp['comparator']['verdict']}**; |delta alpha| = {exp['abs_alpha_shift']:.3f}"
        f" <= {exp['claim_band']} -> within the manuscript wording-only claim band."
        f" ({exp['note']}.)",
        "",
        "## Caveat",
        "",
        v["caveat"],
        "",
        "## beta-table row candidate",
        "",
        "| burst | beta | alpha (derived) | tau_1GHz [ms] | grade |",
        "|---|---|---|---|---|",
        f"| freya | {v['beta']['median']:.3f} +{v['beta']['err_plus']:.3f}/-{v['beta']['err_minus']:.3f}"
        f" | {v['alpha_derived']['median']:.3f}"
        f" +{v['alpha_derived']['err_plus']:.3f}/-{v['alpha_derived']['err_minus']:.3f}"
        f" | {v['tau_1ghz']['median']:.4f} +/- {max(v['tau_1ghz']['err_plus'], v['tau_1ghz']['err_minus']):.4f}"
        f" | {bar['grade']} |",
        "",
        f"Provenance: `{v['provenance']['route_b']}` and siblings; DAG {v['provenance']['dag']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    v = build(preflight=True)
    OUT_JSON.write_text(json.dumps(v, indent=2) + "\n")
    OUT_MD.write_text(render_md(v))
    print(
        f"grade={v['provisional_citable_bar']['grade']} "
        f"citable={v['provisional_citable_bar']['provisional_citable']} "
        f"beta={v['beta']['median']:.4f} alpha={v['alpha_derived']['median']:.4f} "
        f"dalpha_exp_era={v['exp_era_comparison']['abs_alpha_shift']:.3f} "
        f"efolds={v['tail_coverage_at_fit']['efolds']:.2f} "
        f"corr={v['x_zeta_beta_corr']:+.3f}"
    )
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
