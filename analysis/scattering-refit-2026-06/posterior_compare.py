"""Posterior comparator: pure function over two posterior artifacts (issue #100).

Downstream consumers (freya beta co-model DAG): Route A medians vs Route B
posterior samples (#105), and Route B vs the deprecated exp-era summary at
alpha = 4.355 (#106). External behavior only -- no sampler internals, no RNG,
no closure math: alpha enters only because the joint driver already appends
derived-alpha percentiles to its artifacts (reporting-only, ADR-0006).

Accepted artifact forms, normalized by load_params():
- joint-fit summary JSON: {"percentiles": {param: {median, err_minus, err_plus}}}
  (run_joint_fit.py; the exp-era _a1_fits JSONs carry the same triplet shape
  as top-level keys, also handled);
- POC-style JSON: {"median": {param: {median, err_minus, err_plus}}};
- weighted-samples .npz: samples (n, ndim), weights (n), param_names (ndim)
  -> deterministic weighted 16/50/84 percentiles;
- an already-normalized dict {param: {median[, err_minus, err_plus]}} or
  bare medians {param: float} (point estimates, widths unknown).

Verdict enum per shared parameter and overall (worst-of):
agree < widened < shifted < incompatible. Thresholds are caller-tunable
defaults, not scientific adjudication (out of scope per the issue).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Worst-of ordering for the overall verdict.
_RANK = {"agree": 0, "widened": 1, "shifted": 2, "incompatible": 3}
_TRIPLET = ("median", "err_minus", "err_plus")


def _weighted_percentiles(values, weights, qs=(0.16, 0.5, 0.84)):
    """Deterministic weighted percentiles: sort, Hazen/midpoint plotting position
    cw = (cumsum(w) - w/2)/sum(w), interp. corner/dynesty use a left-cumulative
    convention instead; the two agree to O(1/N) for well-sampled posteriors
    (verified ~5e-9 on a 4001-point gaussian against corner.quantile)."""
    order = np.argsort(values)
    v = np.asarray(values, float)[order]
    w = np.asarray(weights, float)[order]
    cw = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(qs, cw, v)


def _from_triplets(container):
    out = {}
    for name, entry in container.items():
        if isinstance(entry, dict) and "median" in entry:
            out[name] = {k: (float(entry[k]) if k in entry else None) for k in _TRIPLET}
        elif isinstance(entry, (int, float)) and not isinstance(entry, bool):
            # bool is an int subclass: without the guard, metadata flags in real
            # exp-era artifacts (shared_zeta, marginalize_gain) would be
            # mis-extracted as median=1.0/0.0 point estimates.
            out[name] = {"median": float(entry), "err_minus": None, "err_plus": None}
    return out


def load_params(artifact):
    """Normalize an artifact (path or dict) to {param: {median, err_minus, err_plus}}.

    err_minus/err_plus are None for point estimates (widths unknown).
    """
    if isinstance(artifact, (str, Path)):
        p = Path(artifact)
        if p.suffix == ".npz":
            npz = np.load(p, allow_pickle=True)
            names = [str(n) for n in npz["param_names"]]
            samples, weights = npz["samples"], npz["weights"]
            out = {}
            for i, name in enumerate(names):
                p16, p50, p84 = _weighted_percentiles(samples[:, i], weights)
                out[name] = {
                    "median": float(p50),
                    "err_minus": float(p50 - p16),
                    "err_plus": float(p84 - p50),
                }
            return out
        artifact = json.loads(p.read_text())

    for key in ("percentiles", "median"):  # known artifact layouts, most specific first
        if isinstance(artifact.get(key), dict):
            found = _from_triplets(artifact[key])
            if found:
                return found
    # exp-era style: per-param triplets (or bare medians) at the top level
    found = _from_triplets(artifact)
    if not found:
        raise ValueError("no recognizable posterior parameters in artifact")
    return found


def _sigma(entry):
    """Symmetrized one-sided width; None when the artifact carries no widths."""
    if entry["err_minus"] is None or entry["err_plus"] is None:
        return None
    return 0.5 * (entry["err_minus"] + entry["err_plus"])


def compare_posteriors(
    artifact_a,
    artifact_b,
    *,
    params=None,
    shift_sigma_max=2.0,
    width_ratio_max=2.0,
    incompatible_sigma=5.0,
):
    """Compare two posterior artifacts on their shared parameters.

    Returns a JSON-serializable verdict dict; identical inputs give identical
    output (no randomness anywhere). Raises ValueError if a compared parameter
    has widths on neither side (shift would be meaningless) or if there is
    nothing to compare.

    params=None intersects ALL shared parameters, nuisance ones included --
    across two unrelated fits (different windows, different t0 conventions)
    that yields a meaningless overall verdict. Cross-fit callers (#105, #106)
    should pass the physics parameters explicitly, e.g. params=["beta",
    "tau_1ghz"] or ["alpha"].
    """
    a, b = load_params(artifact_a), load_params(artifact_b)
    names = list(params) if params else sorted(set(a) & set(b))
    if not names:
        raise ValueError("no shared parameters to compare")

    per_param = {}
    for name in names:
        if name not in a or name not in b:
            raise ValueError(f"parameter '{name}' missing from one artifact")
        ea, eb = a[name], b[name]
        sa, sb = _sigma(ea), _sigma(eb)
        if sa is None and sb is None:
            raise ValueError(f"parameter '{name}' has widths on neither side")
        # Shift in sigma units against the quadrature width of whatever widths
        # exist (one-sided point-vs-posterior comparisons use the posterior's).
        scale = float(np.hypot(sa or 0.0, sb or 0.0))
        shift = abs(ea["median"] - eb["median"]) / scale
        width_ratio = (sa / sb) if (sa is not None and sb is not None) else None

        overlap = None
        if sa is not None and sb is not None:
            lo_a, hi_a = ea["median"] - ea["err_minus"], ea["median"] + ea["err_plus"]
            lo_b, hi_b = eb["median"] - eb["err_minus"], eb["median"] + eb["err_plus"]
            common = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
            overlap = common / min(hi_a - lo_a, hi_b - lo_b)

        if shift > incompatible_sigma:
            verdict = "incompatible"
        elif shift > shift_sigma_max:
            verdict = "shifted"
        elif width_ratio is not None and not (
            1.0 / width_ratio_max <= width_ratio <= width_ratio_max
        ):
            verdict = "widened"
        else:
            verdict = "agree"

        per_param[name] = {
            "median_a": ea["median"],
            "median_b": eb["median"],
            "shift_sigma": shift,
            "width_ratio": width_ratio,
            "overlap_fraction": overlap,
            "verdict": verdict,
        }

    overall = max((p["verdict"] for p in per_param.values()), key=_RANK.__getitem__)
    return {
        "provenance": {
            "artifact_a": str(artifact_a) if isinstance(artifact_a, (str, Path)) else "<dict>",
            "artifact_b": str(artifact_b) if isinstance(artifact_b, (str, Path)) else "<dict>",
        },
        "thresholds": {
            "shift_sigma_max": shift_sigma_max,
            "width_ratio_max": width_ratio_max,
            "incompatible_sigma": incompatible_sigma,
        },
        "params": per_param,
        "verdict": overall,
    }
