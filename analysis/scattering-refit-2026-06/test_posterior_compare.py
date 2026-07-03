"""Posterior comparator (issue #100): normalization of the three artifact
shapes, deterministic verdicts on synthetic agreeing/shifted/widened cases,
and the two downstream comparison shapes (point-vs-samples, samples-vs-summary)."""

import json

import numpy as np
import pytest
from posterior_compare import compare_posteriors, load_params


def _triplet(median, sig):
    return {"median": median, "err_minus": sig, "err_plus": sig}


def _joint_json(tmp_path, name, **params):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"percentiles": params}))
    return p


def _poc_json(tmp_path, name, **params):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"median": params}))
    return p


def _samples_npz(tmp_path, name, **medians_sigmas):
    # Deterministic gaussian samples via inverse-CDF on an even quantile grid
    # (no RNG anywhere: the comparator promises identical output on identical
    # inputs, and the test inputs honor that too).
    from scipy.special import erfinv

    q = np.linspace(0.001, 0.999, 4001)
    z = np.sqrt(2.0) * erfinv(2.0 * q - 1.0)
    names = list(medians_sigmas)
    samples = np.column_stack([m + s * z for m, s in medians_sigmas.values()])
    p = tmp_path / f"{name}.npz"
    np.savez(
        p,
        samples=samples,
        weights=np.ones(z.size),
        param_names=np.array(names, dtype=object),
    )
    return p


def test_agreeing_posteriors_agree(tmp_path):
    a = _joint_json(tmp_path, "a", beta=_triplet(3.7, 0.05), tau_1ghz=_triplet(0.12, 0.01))
    b = _joint_json(tmp_path, "b", beta=_triplet(3.71, 0.05), tau_1ghz=_triplet(0.121, 0.01))
    v = compare_posteriors(a, b)
    assert v["verdict"] == "agree"
    assert set(v["params"]) == {"beta", "tau_1ghz"}
    assert v["params"]["beta"]["shift_sigma"] < 1.0
    assert v["params"]["beta"]["overlap_fraction"] > 0.5


def test_shifted_posteriors_flagged(tmp_path):
    a = _joint_json(tmp_path, "a", beta=_triplet(3.7, 0.05))
    b = _joint_json(tmp_path, "b", beta=_triplet(3.9, 0.05))  # ~2.8 sigma apart
    v = compare_posteriors(a, b)
    assert v["params"]["beta"]["verdict"] == "shifted"
    assert v["verdict"] == "shifted"


def test_widened_posterior_flagged(tmp_path):
    a = _joint_json(tmp_path, "a", beta=_triplet(3.7, 0.05))
    b = _joint_json(tmp_path, "b", beta=_triplet(3.7, 0.15))  # 3x wider, same median
    v = compare_posteriors(a, b)
    assert v["params"]["beta"]["verdict"] == "widened"
    assert v["params"]["beta"]["width_ratio"] == pytest.approx(1.0 / 3.0)


def test_incompatible_beats_shifted(tmp_path):
    a = _joint_json(tmp_path, "a", beta=_triplet(3.0, 0.05))
    b = _joint_json(tmp_path, "b", beta=_triplet(3.9, 0.05))  # ~12.7 sigma
    assert compare_posteriors(a, b)["verdict"] == "incompatible"


def test_point_estimates_vs_samples(tmp_path):
    # Downstream shape 1 (#105): Route A medians (no widths) vs Route B samples.
    a = _poc_json(tmp_path, "poc", beta={"median": 3.71}, tau_1ghz={"median": 0.12})
    b = _samples_npz(tmp_path, "joint", beta=(3.7, 0.05), tau_1ghz=(0.121, 0.01))
    v = compare_posteriors(a, b)
    assert v["verdict"] == "agree"
    assert v["params"]["beta"]["width_ratio"] is None  # point estimate: no width
    assert v["params"]["beta"]["overlap_fraction"] is None


def test_samples_vs_expera_summary(tmp_path):
    # Downstream shape 2 (#106): Route B samples vs the exp-era percentile
    # summary (top-level triplets, the _a1_fits layout).
    b = _samples_npz(tmp_path, "joint", alpha=(4.34, 0.04))
    exp = tmp_path / "expera.json"
    exp.write_text(json.dumps({"alpha": _triplet(4.355, 0.04), "alpha_bounds": [1.0, 6.0]}))
    v = compare_posteriors(b, exp, params=["alpha"])
    assert v["params"]["alpha"]["shift_sigma"] < 1.0
    assert v["verdict"] == "agree"


def test_npz_percentiles_match_analytic_gaussian(tmp_path):
    p = _samples_npz(tmp_path, "g", beta=(3.7, 0.05))
    got = load_params(p)["beta"]
    assert got["median"] == pytest.approx(3.7, abs=1e-3)
    assert got["err_minus"] == pytest.approx(0.05, rel=0.03)  # 16/84 of a gaussian ~ 1 sigma
    assert got["err_plus"] == pytest.approx(0.05, rel=0.03)


def test_deterministic_identical_inputs_identical_verdicts(tmp_path):
    a = _joint_json(tmp_path, "a", beta=_triplet(3.7, 0.05))
    b = _samples_npz(tmp_path, "b", beta=(3.72, 0.06))
    assert compare_posteriors(a, b) == compare_posteriors(a, b)


def test_bool_metadata_not_extracted_as_params(tmp_path):
    # Real exp-era artifacts carry bool flags (shared_zeta, marginalize_gain);
    # bool is an int subclass and must not become a median=1.0 point estimate.
    p = tmp_path / "expera_variant.json"
    p.write_text(
        json.dumps(
            {
                "alpha": _triplet(4.355, 0.04),
                "shared_zeta": True,
                "marginalize_gain": False,
                "components_C": 2,
            }
        )
    )
    got = load_params(p)
    assert "shared_zeta" not in got and "marginalize_gain" not in got
    assert set(got) == {"alpha", "components_C"}  # real ints still extract


def test_no_widths_on_either_side_raises(tmp_path):
    a = _poc_json(tmp_path, "a", beta={"median": 3.7})
    b = _poc_json(tmp_path, "b", beta={"median": 3.8})
    with pytest.raises(ValueError, match="widths on neither side"):
        compare_posteriors(a, b)


def test_no_shared_parameters_raises(tmp_path):
    a = _joint_json(tmp_path, "a", beta=_triplet(3.7, 0.05))
    b = _joint_json(tmp_path, "b", tau_1ghz=_triplet(0.12, 0.01))
    with pytest.raises(ValueError, match="no shared parameters"):
        compare_posteriors(a, b)


def test_provenance_names_both_artifacts(tmp_path):
    a = _joint_json(tmp_path, "a", beta=_triplet(3.7, 0.05))
    b = _joint_json(tmp_path, "b", beta=_triplet(3.7, 0.05))
    v = compare_posteriors(a, b)
    assert v["provenance"]["artifact_a"].endswith("a.json")
    assert v["provenance"]["artifact_b"].endswith("b.json")
