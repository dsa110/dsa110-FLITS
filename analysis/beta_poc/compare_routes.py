"""Route A vs Route B posterior comparison for the freya beta co-model (issue #105).

Thin, deterministic driver over the #100 comparator (pure function, no RNG):
Route A medians (run_beta_poc.py --real POC-style JSON) against Route B's
weighted posterior samples (run_joint_fit.py --shared-zeta npz). Physics
parameters only -- t0/ddm are per-band nuisance with matching prep/window here,
but excluded per the comparator docstring's cross-fit guidance so the overall
verdict is carried by the measurement, not alignment bookkeeping.

Exit is nonzero on any overall verdict other than `agree` (the #105 stop
condition: "agreement beyond tolerance" -- and the comparator emits `widened`
precisely when width_ratio breaches width_ratio_max, so it IS a tolerance
breach), mirroring the likelihood_equivalence.py CLI precedent.

  conda run -n flits python analysis/beta_poc/compare_routes.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROUTE_A_JSON = REPO / "analysis" / "beta_poc" / "freya" / "freya_beta_poc_fit_real.json"
ROUTE_B_NPZ = (
    REPO
    / "analysis"
    / "scattering-refit-2026-06"
    / "local_runs"
    / "freya_joint_samples_sharedzeta.npz"
)
OUT_JSON = REPO / "analysis" / "beta_poc" / "freya" / "freya_route_a_vs_b.json"
# Shared physics of the 8-vector theta; per-band t0/delta_dm are nuisance.
PHYSICS_PARAMS = ["beta", "tau_1ghz", "zeta_1ghz", "x_zeta"]
STOP_VERDICTS = ("widened", "shifted", "incompatible")


def _load_comparator():
    spec = importlib.util.spec_from_file_location(
        "posterior_compare",
        REPO / "analysis" / "scattering-refit-2026-06" / "posterior_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(route_a=ROUTE_A_JSON, route_b=ROUTE_B_NPZ, params=PHYSICS_PARAMS):
    """Comparator verdict dict plus #105 bookkeeping (pure given the artifacts)."""
    pc = _load_comparator()
    result = pc.compare_posteriors(route_a, route_b, params=list(params))
    result["issue"] = "dsa110-FLITS#105"
    result["routes"] = {
        "a": "BetaCoupledLogL POC harness (run_beta_poc.py --real, seed-pinned)",
        "b": "_JointLogLikelihoodGainSharedZeta production driver (run_joint_fit.py --shared-zeta)",
    }
    result["stop_condition_triggered"] = result["verdict"] in STOP_VERDICTS
    return result


def main() -> int:
    result = compare()
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    for name, p in result["params"].items():
        print(
            f"{name:>10}: A={p['median_a']:.6g} B={p['median_b']:.6g} "
            f"shift={p['shift_sigma']:.2f}sigma width_ratio={p['width_ratio']:.2f} "
            f"overlap={p['overlap_fraction']:.2f} -> {p['verdict']}"
        )
    print(f"overall verdict: {result['verdict']}  wrote {OUT_JSON}")
    if result["stop_condition_triggered"]:
        print("STOP: verdict beyond tolerance -- diagnose before any downstream motion (#105).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
