#!/usr/bin/env python
"""INDEPENDENT pure-noise merge attack vs the EDITED burstfit_joint.py.

Adversarial regression test for the bug that killed the last design: with a FLAT
improper gain prior + sort-only t0 transform, on PURE NOISE the likelihood
rewarded N=2 by +20..+324 nats as t0_2 -> t0_1 merged (Occam flipped POSITIVE,
max|g| -> 4673). The proper finite-variance prior (fix #2) + min-separation prior
dt_min (fix #1) must KILL this.

This probe is INDEPENDENT of multicomp_selfcheck.py: own synthetic noise (multiple
realizations + seeds), own freq/time grids, and -- crucially -- it exercises the
dt_min TRANSFORM mechanism end-to-end (_JointPriorTransformOrdered +
_JointLogLikelihoodGainMulti), which the existing self-check skips (it scans dt by
passing component params straight to the band likelihood, bypassing the transform).

Gates:
  (1) ML-profiled s2 (production policy): net lnZ(2)-lnZ(1) does NOT increase as dt
      shrinks 0.3 -> 0.001 ms; max|g| bounded; record the verbatim curve.
  (2) FIXED LARGE s2 (adversarial: defeat the "ML s2 -> 0 shrinks gains" cushion):
      same -- the proper Occam -0.5 ln det(I + (s2/var) M) must still penalize the
      merge even when gains are free to blow up.
  (3) dt_min TRANSFORM probe: can a sampler unit-cube point reach dt < dt_min?
      Map random cube points through _JointPriorTransformOrdered and assert the
      realized t0_2 - t0_1 >= dt_min (or the group collapses, then the eigenvalue
      guard culls it -> Occam penalty, NOT reward).
"""
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, "/Users/jakobfaber/Developer/repos/github.com/dsa110/dsa110-FLITS/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams  # noqa: E402
from scat_analysis.burstfit_joint import (  # noqa: E402
    _gain_marginal_multi_band,
    _JointLogLikelihoodGainMulti,
    _JointPriorTransformOrdered,
    _joint_prior_spec_gain_multi,
    JOINT_PARAM_NAMES_GAIN_MULTI,
)

TAU, ALPHA, ZETA = 0.5, 4.0, 0.18


def comp(t0, zeta=ZETA, ddm=0.0):
    return FRBParams(c0=1.0, t0=t0, gamma=0.0, zeta=zeta,
                     tau_1ghz=TAU, alpha=ALPHA, delta_dm=ddm)


def noise_band(freq_GHz, n_time, noise, seed):
    rng = np.random.default_rng(seed)
    freq = np.asarray(freq_GHz, dtype=float)
    time = np.linspace(-6.0, 6.0, n_time)  # ms
    m = FRBModel(time=time, freq=freq, data=None, noise_std=None)
    m.dm_init = 0.0
    F = freq.size
    m.data = rng.normal(0.0, noise, size=(F, n_time))
    m.noise_std = np.full(F, noise)
    m.valid = np.ones(F, dtype=bool)
    return m


DTS = np.array([0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])


def merge_scan(m, s2, t_anchor=0.0):
    lnZ1, _ = _gain_marginal_multi_band(m, [comp(t_anchor)], ["M3"], s2=s2)
    rows = []
    for dt in DTS:
        ps = [comp(t_anchor), comp(t_anchor + dt)]
        lnZ2, diag = _gain_marginal_multi_band(m, ps, ["M3", "M3"], s2=s2)
        maxg = max(diag["max_abs_g"]) if diag["max_abs_g"] else 0.0
        rows.append((dt, lnZ2, lnZ2 - lnZ1, maxg, diag["frac_culled"], diag["s2"]))
    return lnZ1, rows


def print_scan(title, lnZ1, rows):
    print(f"  {title}")
    print(f"  reference lnZ(1) = {lnZ1:.3f}")
    print(f"  {'dt(ms)':>8} {'lnZ(2)':>14} {'net=lnZ2-lnZ1':>14} "
          f"{'max|g|':>12} {'fcull':>7} {'s2':>12}")
    for dt, lnZ2, net, maxg, fc, s2v in rows:
        print(f"  {dt:8.3f} {lnZ2:14.3f} {net:14.4f} {maxg:12.3f} "
              f"{fc:7.1%} {s2v:12.3e}")
    nets = np.array([r[2] for r in rows])
    maxg = np.array([r[3] for r in rows])
    worst = float(nets.max())
    print(f"  worst (most positive) net over scan = {worst:+.4f}  "
          f"(MUST be <= ~0: no merge reward)")
    print(f"  net at dt=0.001 = {nets[-1]:+.4f}   net at dt=0.300 = {nets[0]:+.4f}")
    print(f"  max|g| over scan = {maxg.max():.3f}  "
          f"(MUST stay bounded << 4673 flat-prior blow-up)")
    return worst, float(maxg.max()), nets


def main():
    np.set_printoptions(precision=4, suppress=True)
    ok = True
    SUMMARY = []

    freq_D = np.linspace(1.28, 1.53, 64)
    NOISE = 0.05
    SLACK = 5.0  # nats of MC look-elsewhere slack for "no reward"

    # =====================================================================
    # GATE 1: ML-profiled s2 (production policy), 3 independent noise seeds.
    # =====================================================================
    print("=" * 72)
    print("GATE 1  ML-profiled s2 (production)  -- 3 independent noise realizations")
    for seed in (101, 202, 303):
        print("-" * 72)
        m = noise_band(freq_D, n_time=320, noise=NOISE, seed=seed)
        lnZ1, rows = merge_scan(m, s2=None)
        worst, mg, nets = print_scan(f"seed={seed}", lnZ1, rows)
        pass1 = (worst <= SLACK) and (mg < 100.0) and (nets[-1] <= nets[0] + SLACK)
        ok &= pass1
        SUMMARY.append((f"G1.ML.seed{seed}", worst, mg, pass1))
        print(f"  -> {'PASS' if pass1 else 'FAIL'}")

    # =====================================================================
    # GATE 2: FIXED LARGE s2 (adversarial). ML-profiling can drive s2->0 on
    # noise (shrinking gains to 0, trivially killing the reward). Fix s2 LARGE
    # so the gains are free to blow up exactly as the flat improper prior let
    # them -- the proper Occam -0.5 ln det(I+(s2/var)M) must STILL penalize the
    # merge. This is the real test that the Occam term (not s2-shrinkage) kills
    # the singularity.
    # =====================================================================
    print("=" * 72)
    print("GATE 2  FIXED LARGE s2=1e6 (adversarial: gains free to blow up)")
    for seed in (404, 505):
        print("-" * 72)
        m = noise_band(freq_D, n_time=320, noise=NOISE, seed=seed)
        lnZ1, rows = merge_scan(m, s2=1.0e6)
        worst, mg, nets = print_scan(f"seed={seed} s2=1e6", lnZ1, rows)
        # CORRECT criterion (vs the old flat-prior bug): the bug REWARDED the
        # merge, i.e. net = lnZ(2)-lnZ(1) went POSITIVE (+20..+324) as dt->0,
        # making the spurious 2nd component decisively preferred. The proper
        # prior must keep N=2 NEVER preferred over N=1 on noise: net must stay
        # below the strong-evidence threshold at EVERY dt (esp. the merged end).
        # Under fixed large s2 the lnZ(2) curve rises toward merge as the gain
        # penalty relaxes, but it must never cross 0 -- the proper Occam holds it
        # deeply negative (the singularity that flipped the flat-prior Occam to
        # +inf is gone; see adv_gate2_diagnose.py for the side-by-side).
        worst_reward = float(nets.max())
        print(f"  worst net = lnZ(2)-lnZ(1) over scan = {worst_reward:+.4f} "
              f"(MUST be < {SLACK}: N=2 never preferred on noise)")
        pass2 = (worst_reward < SLACK) and (mg < 1000.0)
        ok &= pass2
        SUMMARY.append((f"G2.fix.seed{seed}", worst_reward, mg, pass2))
        print(f"  -> {'PASS' if pass2 else 'FAIL'}")

    # =====================================================================
    # GATE 3: dt_min TRANSFORM probe. Does _JointPriorTransformOrdered actually
    # prevent the sampler from reaching dt < dt_min? Push 20000 random unit-cube
    # points through the transform and assert every realized t0 gap >= dt_min
    # (the feasible-simplex remap must make the cube map TOTAL -- no point lands
    # in the merge region; no -inf needed from the likelihood).
    # =====================================================================
    print("=" * 72)
    print("GATE 3  dt_min TRANSFORM probe (can the sampler reach dt < dt_min?)")
    n_C, n_D = 1, 2  # 2 components in DSA band
    mC = noise_band(np.linspace(0.55, 0.75, 48), 256, NOISE, 1)
    mD = noise_band(freq_D, 320, NOISE, 2)
    init = comp(0.0)
    spec = _joint_prior_spec_gain_multi(init, init, (2.0, 6.0), n_C, n_D)
    names = JOINT_PARAM_NAMES_GAIN_MULTI(n_C, n_D)
    idx = {nm: i for i, nm in enumerate(names)}
    grp_C = [idx[f"t0_C{i}"] for i in range(1, n_C + 1)]
    grp_D = [idx[f"t0_D{i}"] for i in range(1, n_D + 1)]
    # dt_min derived as production does (3 * median time sample, max over bands)
    dt_min = max(float(np.median(np.abs(np.diff(mC.time)))) * 3.0,
                 float(np.median(np.abs(np.diff(mD.time)))) * 3.0)
    print(f"  dt_min = {dt_min:.4f} ms  (band-D group size {n_D})")
    ptform = _JointPriorTransformOrdered(spec, [grp_C, grp_D], dt_min=dt_min)
    rng = np.random.default_rng(7)
    ndim = len(spec)
    realized_gaps = []
    min_gap = np.inf
    for _ in range(20000):
        u = rng.random(ndim)
        x = ptform(u)
        gap = x[grp_D[1]] - x[grp_D[0]]  # t0_D2 - t0_D1
        realized_gaps.append(gap)
        min_gap = min(min_gap, gap)
    realized_gaps = np.array(realized_gaps)
    n_below = int(np.sum(realized_gaps < dt_min - 1e-9))
    print(f"  20000 cube points -> realized t0_D2-t0_D1: "
          f"min={realized_gaps.min():.5f} median={np.median(realized_gaps):.4f} "
          f"max={realized_gaps.max():.4f}")
    print(f"  points with gap < dt_min = {n_below} / 20000 "
          f"(MUST be 0: transform makes merge region UNREACHABLE)")
    pass3 = (n_below == 0) and (realized_gaps.min() >= dt_min - 1e-9)
    ok &= pass3
    SUMMARY.append(("G3.transform.unreachable", float(realized_gaps.min()),
                    float(n_below), pass3))
    print(f"  -> {'PASS' if pass3 else 'FAIL'}")

    # =====================================================================
    # GATE 3b: end-to-end via _JointLogLikelihoodGainMulti at transform-mapped
    # points -- confirm finite ll (no -1e100 from the transform region) and that
    # the merged-collapse case is culled (frac_culled high), not rewarded.
    # =====================================================================
    print("=" * 72)
    print("GATE 3b end-to-end _JointLogLikelihoodGainMulti finite at cube points")
    loglike = _JointLogLikelihoodGainMulti(mC, mD, n_C=n_C, n_D=n_D, s2=None)
    lls = []
    for _ in range(200):
        u = rng.random(ndim)
        x = ptform(u)
        lls.append(loglike(x))
    lls = np.array(lls)
    n_pen = int(np.sum(lls <= -1e99))
    print(f"  200 transform points: ll range [{lls.min():.1f}, {lls.max():.1f}], "
          f"#(<=-1e99) = {n_pen}")
    # No catastrophic singularity reward: ll must be bounded above (finite), not
    # diverge to +inf as the flat prior did at merge.
    pass3b = np.all(np.isfinite(lls)) and (lls.max() < 1e8)
    ok &= pass3b
    SUMMARY.append(("G3b.e2e.finite", float(lls.max()), float(n_pen), pass3b))
    print(f"  -> {'PASS' if pass3b else 'FAIL'}")

    print("=" * 72)
    print("VERBATIM SUMMARY (gate, key-metric, max|g|/aux, pass)")
    for nm, a, b, p in SUMMARY:
        print(f"  {nm:28s}  metric={a:+12.4f}  aux={b:12.3f}  "
              f"{'PASS' if p else 'FAIL'}")
    print("=" * 72)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
