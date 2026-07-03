"""Tail-coverage preflight (issue #101): analytic unit tests for the closed-form
captured fraction / e-folds, threshold boundary behavior, and the slow-marked
real-window application on freya's prepared CHIME+DSA bands (#99 configs)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from tail_coverage import (
    DEFAULT_MIN_EFOLDS,
    pbf_captured_fraction,
    pbf_crossover,
    tail_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "analysis" / "scattering-refit-2026-06" / "local_runs" / "configs"


def _numeric_fraction(s_window, beta):
    """Independent check: trapezoid integration of the piecewise PBF itself."""
    s_c = 2.0 * np.log(2.0 / (4.0 - beta))
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, min(s_window, 50.0), 200_001),
                np.geomspace(1e-3, 1e8, 200_001),  # far tail, so I(inf) converges
                [s_c, s_window],
            ]
        )
    )
    h = np.exp(-grid)
    tail = grid > s_c
    h[tail] = np.exp(-s_c) * (grid[tail] / s_c) ** (-0.5 * beta)
    total = np.trapezoid(h, grid)
    inside = grid <= s_window
    return np.trapezoid(h[inside], grid[inside]) / total


def test_crossover_known_values():
    # s_c = 2 ln(2/(4-beta)): Kolmogorov 11/3 -> 2 ln 6; beta -> 2 floors at 1e-3.
    assert pbf_crossover(11.0 / 3.0) == pytest.approx(2.0 * np.log(6.0))
    assert pbf_crossover(2.0) == 1e-3


def test_exponential_limit_efolds_equal_window_depth():
    # Within BETA_EXP_EPS of 4 the model uses the pure exponential PBF, where
    # "captured e-folds" is literal: E = window span in units of tau.
    r = tail_coverage(freq_ghz=1.0, tau_1ghz_ms=0.1, beta=3.99, time_ms=(0.0, 0.5))
    assert r["s_window"] == pytest.approx(5.0)
    assert r["efolds"] == pytest.approx(5.0, rel=1e-12)
    assert r["captured_fraction"] == pytest.approx(1.0 - np.exp(-5.0))


@pytest.mark.parametrize("beta", [3.0, 11.0 / 3.0, 3.7])
@pytest.mark.parametrize("s_window", [0.5, 2.0, 5.0, 20.0])
def test_captured_fraction_matches_numerical_integration(beta, s_window):
    got = float(pbf_captured_fraction(s_window, beta))
    want = _numeric_fraction(s_window, beta)
    assert got == pytest.approx(want, rel=1e-3)


def test_fraction_monotone_in_window_and_saturates():
    # The heavy tail saturates slowly by design: at beta = 3.7 the window must
    # reach s ~ 2e4 (not ~7 as for an exponential) before <0.1% of the PBF area
    # is outstanding — the truncation risk this preflight exists to expose.
    s = np.array([0.5, 1.0, 2.0, 5.0, 20.0, 200.0, 2e4])
    f = pbf_captured_fraction(s, 3.7)
    assert np.all(np.diff(f) > 0)
    assert f[-2] < 0.999  # s=200 still short of an exponential's ~1-1e-87
    assert f[-1] == pytest.approx(1.0, abs=1e-3)
    assert float(pbf_captured_fraction(0.0, 3.7)) == 0.0


def test_heavier_tail_captures_less_at_fixed_window():
    # Lower beta = heavier power-law tail = more area beyond any fixed window.
    for s in (1.0, 3.0, 10.0):
        assert float(pbf_captured_fraction(s, 3.1)) < float(pbf_captured_fraction(s, 3.7))
        assert float(pbf_captured_fraction(s, 3.7)) < float(pbf_captured_fraction(s, 3.99))


def test_threshold_boundary():
    # Exp limit makes the boundary exact: efolds == s_window.
    at = tail_coverage(1.0, 0.1, 3.99, (0.0, 0.3), min_efolds=3.0)
    below = tail_coverage(1.0, 0.1, 3.99, (0.0, 0.299), min_efolds=3.0)
    assert at["passes"] and at["efolds"] == pytest.approx(3.0)
    assert not below["passes"]


def test_worst_channel_is_lowest_frequency():
    band = np.linspace(0.4, 0.8, 64)
    r_band = tail_coverage(band, 0.119, 3.7, (0.0, 20.0))
    r_fmin = tail_coverage(0.4, 0.119, 3.7, (0.0, 20.0))
    assert r_band == r_fmin
    r_fmax = tail_coverage(0.8, 0.119, 3.7, (0.0, 20.0))
    assert r_band["efolds"] < r_fmax["efolds"]


def test_tau_frequency_scaling_and_t0():
    # At 1 GHz tau = tau_1ghz exactly; alpha from the thin-screen closure.
    r = tail_coverage(1.0, 0.2, 3.7, (0.0, 1.0), t0_ms=0.5)
    assert r["tau_at_fmin_ms"] == pytest.approx(0.2)
    assert r["alpha"] == pytest.approx(2.0 * 3.7 / 1.7)
    assert r["window_span_ms"] == pytest.approx(0.5)
    assert r["s_window"] == pytest.approx(2.5)


def test_burst_after_window_end_captures_nothing():
    r = tail_coverage(1.0, 0.1, 3.7, (0.0, 1.0), t0_ms=2.0)
    assert r["s_window"] == 0.0
    assert r["captured_fraction"] == 0.0
    assert r["efolds"] == 0.0
    assert not r["passes"]


def test_beta_edges_clip_like_the_kernel():
    # beta <= thin-screen min clips (kernel clips to 2.01); beta = 4 -> exp branch.
    r_lo = tail_coverage(1.0, 0.1, 2.0, (0.0, 1.0))
    assert np.isfinite(r_lo["efolds"]) and r_lo["alpha"] > 4.0
    r_hi = tail_coverage(1.0, 0.1, 4.0, (0.0, 1.0))
    assert r_hi["alpha"] == 4.0
    assert r_hi["s_crossover"] is None
    assert r_hi["efolds"] == pytest.approx(r_hi["s_window"])


def test_huge_window_stays_json_serializable():
    r = tail_coverage(1.0, 1e-4, 3.99, (0.0, 100.0))
    assert np.isfinite(r["efolds"])
    json.dumps(r)


@pytest.mark.slow
def test_freya_real_window_preflight(tmp_path):
    # Real data resolves through the pinned manuscript checkout's symlinks
    # (external to this repo) — skip where not staged, as in the #99 smoke test.
    for band in ("chime", "dsa"):
        cfg = yaml.safe_load((CONFIGS / f"freya_{band}_run.yaml").read_text())
        if not Path(cfg["path"]).exists():
            pytest.skip(f"{Path(cfg['path']).name} not staged (pinned-checkout data symlink)")
        if not Path(cfg["telcfg_path"]).exists():
            pytest.skip("pinned-checkout telescopes.yaml not staged")

    import tail_coverage as tc

    out = tmp_path / "freya_tail_coverage.json"
    assert tc.main(["--out", str(out)]) == 0
    art = json.loads(out.read_text())

    assert set(art["bands"]) == {"chime", "dsa"}
    for band in ("chime", "dsa"):
        b = art["bands"][band]
        # Grid verdicts stand on the raw data-driven t0; the MLE-refined t0 is
        # recorded but not trusted (post-#98 refine rails beta and drags t0).
        assert b["window_ms"][0] <= b["t0_raw_init_ms"] <= b["window_ms"][1]
        assert "t0_mle_init_ms" in b
        assert len(b["grid"]) == len(tc.TAU_GRID_MS) * len(tc.BETA_GRID)
        for g in b["grid"]:
            assert 0.0 <= g["captured_fraction"] < 1.0
            assert g["efolds"] >= 0.0
            assert isinstance(g["passes"], bool)

    # DSA at the exp-era-suggested scale: tau(1.31 GHz) ~ 0.037 ms against a
    # millisecond-scale prepared window — passes at any sane threshold. The CHIME
    # verdict is the science question the artifact records; not asserted here.
    dsa_expera = next(
        g
        for g in art["bands"]["dsa"]["grid"]
        if g["tau_1ghz_ms"] == 0.119 and abs(g["beta"] - 3.7) < 1e-9
    )
    assert dsa_expera["passes"]
    assert dsa_expera["efolds"] >= DEFAULT_MIN_EFOLDS
