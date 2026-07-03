"""Likelihood equivalence check (issue #103): Route A (POC BetaCoupledLogL) and
Route B (driver _JointLogLikelihoodGainSharedZeta) agree at identical theta on
synthetic bands -- including the beta ~ 4 exponential-PBF dispatch points -- and
the adjudicator hard-fails on a fabricated disagreement. The real-window freya
application is slow-marked with the data-absent skip convention (#99)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from likelihood_equivalence import (
    RTOL,
    THETA_NAMES,
    adjudicate,
    evaluate,
    rel_diff,
    theta_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "analysis" / "scattering-refit-2026-06" / "local_runs" / "configs"


def _make_band(fmin, fmax, nch, seed):
    """Small deterministic synthetic band, as in burstfit_joint._demo_shared_zeta."""
    from scattering.scat_analysis.burstfit import FRBModel, FRBParams

    rng = np.random.default_rng(seed)
    freq = np.linspace(fmin, fmax, nch)
    time = np.arange(240) * 0.05
    truth = FRBParams(
        c0=1.0,
        t0=float(time.mean()),
        gamma=0.0,
        zeta=0.30 * freq**-0.6,
        tau_1ghz=0.8,
        beta=3.7,
        delta_dm=0.0,
    )
    m0 = FRBModel(time=time, freq=freq, data=np.zeros((nch, time.size)), dm_init=0.0)
    clean = m0(truth, "M3")
    noisy = clean + rng.normal(0.0, 0.04 * float(clean.max()), clean.shape)
    return FRBModel(time=time, freq=freq, data=noisy, dm_init=0.0)


@pytest.fixture(scope="module")
def bands():
    return _make_band(0.40, 0.80, 16, seed=2), _make_band(1.31, 1.50, 16, seed=3)


def test_theta_grid_contents():
    g = theta_grid(6.0, 6.0)
    assert g.shape == (260, 8)  # 2^8 corners + center + exp-era + two exp-branch betas
    # POC prior-box corners reached through its own ptform (tau/zeta log axes).
    assert g[:, 0].min() == pytest.approx(0.01) and g[:, 0].max() == pytest.approx(5.0)
    assert g[:, 1].min() == 3.0 and g[:256, 1].max() == 3.95
    assert g[:, 2].min() == pytest.approx(1e-3) and g[:, 2].max() == pytest.approx(1.0)
    # Special points: exp-era suggestion, then the exponential-branch dispatch pair.
    assert [round(b, 3) for b in g[-3:, 1]] == [3.7, 3.99, 4.0]
    assert np.all(g[-3:, 0] == 0.119)


def test_routes_agree_on_synthetic_grid(bands):
    m_C, m_D = bands
    records = evaluate(m_C, m_D, theta_grid(float(m_C.time.mean()), float(m_D.time.mean())))
    v = adjudicate(records)
    assert v["passes"] and v["n_theta"] == 260
    # RTOL = 1e-12 is the documented contract; the measured reality is bitwise
    # identity (identical float ops in identical order in both routes).
    assert v["max_rel_diff"] == 0.0
    # Totals came through each route's own __call__ and are the band sums.
    r = records[256]  # prior center: finite, un-clamped
    assert np.isfinite(r["route_A"]) and r["route_A"] > -1e100
    assert r["route_A"] == r["bands"]["C"]["route_A"] + r["bands"]["D"]["route_A"]


def test_exp_branch_dispatch_identical(bands):
    # beta = 3.97 (power-law side), 3.99 and 4.0 (exponential side of the
    # BETA_EXP_EPS = 0.02 branch point): both routes must dispatch identically.
    m_C, m_D = bands
    t0 = float(m_C.time.mean())
    thetas = [[0.119, b, 0.085, -0.79, t0, 0.0, t0, 0.0] for b in (3.97, 3.99, 4.0)]
    v = adjudicate(evaluate(m_C, m_D, thetas))
    assert v["passes"] and v["max_rel_diff"] == 0.0


def test_adjudicate_flags_disagreement():
    rec = {
        "theta": {n: float(i) for i, n in enumerate(THETA_NAMES)},
        "route_A": -1234.5,
        "route_B": -1234.5,
        "rel_diff": 0.0,
        "bands": {
            "C": {"route_A": -600.0, "route_B": -600.0, "rel_diff": 0.0},
            "D": {"route_A": -634.5, "route_B": -634.5, "rel_diff": 0.0},
        },
    }
    bad = json.loads(json.dumps(rec))
    bad["bands"]["D"]["route_B"] = -634.5 * (1.0 + 1e-9)
    # The stale rel_diff fields are left at 0.0 on purpose: adjudicate must
    # recompute from the route values, never trust a precomputed diff.
    v = adjudicate([rec, bad])
    assert not v["passes"] and len(v["failures"]) == 1
    f = v["failures"][0]  # per-band, per-theta breakdown rides with the failure
    assert f["index"] == 1
    assert rel_diff(f["bands"]["D"]["route_A"], f["bands"]["D"]["route_B"]) > RTOL
    assert f["theta"]["tau_1ghz"] == 0.0 and v["max_rel_diff"] > RTOL


def test_rel_diff_properties():
    assert rel_diff(-5.0, -5.0) == 0.0
    assert rel_diff(2.0, 1.0) == rel_diff(1.0, 2.0) == 0.5
    assert rel_diff(1e-300, 0.0) == 1e-300  # denominator floored at 1: absolute near zero
    assert np.isnan(rel_diff(float("nan"), 1.0))  # NaN propagates; adjudicate fails closed
    assert not adjudicate([{"route_A": float("nan"), "route_B": -5.0, "bands": {}}])["passes"]


@pytest.mark.slow
def test_freya_real_equivalence(tmp_path):
    # Real data resolves through the pinned manuscript checkout's symlinks
    # (external to this repo) -- skip where not staged, as in the #99 smoke test.
    for band in ("chime", "dsa"):
        cfg = yaml.safe_load((CONFIGS / f"freya_{band}_run.yaml").read_text())
        if not Path(cfg["path"]).exists():
            pytest.skip(f"{Path(cfg['path']).name} not staged (pinned-checkout data symlink)")
        if not Path(cfg["telcfg_path"]).exists():
            pytest.skip("pinned-checkout telescopes.yaml not staged")

    import likelihood_equivalence as le

    out = tmp_path / "freya_likelihood_equivalence.json"
    assert le.main(["--out", str(out)]) == 0
    art = json.loads(out.read_text())
    v = art["verdict"]
    assert v["passes"] and v["n_theta"] == 260 and v["max_rel_diff"] <= v["rtol"]
    assert art["theta_names"] == list(THETA_NAMES)
    assert len(art["records"]) == 260
    for band in ("chime", "dsa"):
        assert len(art["bands"][band]["shape"]) == 2
