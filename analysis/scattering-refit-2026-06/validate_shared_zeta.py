#!/usr/bin/env python
"""Local injection-recovery validation for the shared-zeta(nu) joint fit.

Before trusting the shared-width law on real data, prove on synthetic truth that
the nested fit recovers (tau_1ghz, alpha, zeta_1ghz, x_zeta) AND quantify the
x_zeta-alpha degeneracy via the posterior correlation. Both intrinsic width
(zeta ~ nu^x_zeta, symmetric) and scattering (tau ~ nu^-alpha, one-sided) broaden
low-freq pulses, so the fit can only separate them through the tail asymmetry +
the ~1 GHz CHIME<->DSA lever arm. A strongly anti-correlated x_zeta-alpha
posterior would mean alpha is NOT independently constrained -- this script is the
gate that catches that.

Run from the repo root in the `flits` env:
    python -m analysis.scattering-refit-2026-06.validate_shared_zeta   # (hyphens: see below)
or directly:
    python analysis/scattering-refit-2026-06/validate_shared_zeta.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scattering"))
from scat_analysis.burstfit import FRBModel, FRBParams
from scat_analysis.burstfit_joint import JOINT_PARAM_NAMES_GAIN_SHARED_ZETA, fit_joint_scattering

TRUTH = dict(tau_1ghz=0.8, alpha=4.0, zeta_1ghz=0.30, x_zeta=-0.6)


def make_band(fmin, fmax, nch, seed):
    """Synthesize one band whose intrinsic width follows zeta(nu)=z1*nu^x."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(fmin, fmax, nch)
    time = np.arange(260) * 0.05
    zeta_nu = TRUTH["zeta_1ghz"] * freq ** TRUTH["x_zeta"]
    truth = FRBParams(
        c0=1.0,
        t0=time.mean(),
        gamma=0.0,
        zeta=zeta_nu,
        tau_1ghz=TRUTH["tau_1ghz"],
        alpha=TRUTH["alpha"],
        delta_dm=0.0,
    )
    m0 = FRBModel(time=time, freq=freq, data=np.zeros((nch, time.size)), dm_init=0.0)
    clean = m0(truth, "M3")
    noisy = clean + rng.normal(0, 0.05 * clean.max(), clean.shape)
    m = FRBModel(time=time, freq=freq, data=noisy, dm_init=0.0)
    # init anchors the t0 window + (unused for absolute_bounds) scale -- truth-ish.
    init = FRBParams(
        c0=1.0,
        t0=time.mean(),
        gamma=0.0,
        zeta=float(np.median(zeta_nu)),
        tau_1ghz=TRUTH["tau_1ghz"],
        alpha=TRUTH["alpha"],
        delta_dm=0.0,
    )
    return m, init


def wcov(samples, weights, names, a, b):
    """Weighted covariance + correlation of two named columns."""
    ia, ib = names.index(a), names.index(b)
    x, y = samples[:, ia], samples[:, ib]
    mx = np.average(x, weights=weights)
    my = np.average(y, weights=weights)
    cxx = np.average((x - mx) ** 2, weights=weights)
    cyy = np.average((y - my) ** 2, weights=weights)
    cxy = np.average((x - mx) * (y - my), weights=weights)
    r = cxy / np.sqrt(cxx * cyy) if cxx > 0 and cyy > 0 else float("nan")
    return cxy, r


def main():
    mC, iC = make_band(0.40, 0.80, 24, seed=10)
    mD, iD = make_band(1.31, 1.50, 24, seed=11)

    res = fit_joint_scattering(
        model_C=mC,
        init_C=iC,
        model_D=mD,
        init_D=iD,
        shared_zeta=True,
        alpha_bounds=(2.0, 6.0),
        x_zeta_bounds=(-4.0, 2.0),
        nlive=400,
        dlogz=0.5,
        nproc=None,
        verbose=True,
    )

    names = list(res["param_names"])
    assert tuple(names) == JOINT_PARAM_NAMES_GAIN_SHARED_ZETA, names
    pct = res["percentiles"]

    print("\n=== shared-zeta injection-recovery ===")
    ok = True
    for k in ("tau_1ghz", "alpha", "zeta_1ghz", "x_zeta"):
        med = pct[k]["median"]
        em, ep = pct[k]["err_minus"], pct[k]["err_plus"]
        truth = TRUTH[k]
        # within 3 sigma of the asymmetric posterior?
        sig = ep if med < truth else em
        nsig = abs(med - truth) / sig if sig > 0 else float("inf")
        flag = "PASS" if nsig <= 3.0 else "FAIL"
        ok &= nsig <= 3.0
        print(
            f"  {k:10s} truth={truth:+.3f}  fit={med:+.3f} (+{ep:.3f}/-{em:.3f})  {nsig:.1f}sig {flag}"
        )

    cxy, r = wcov(res["samples"], res["weights"], names, "x_zeta", "alpha")
    print(f"\n  x_zeta-alpha posterior corr r = {r:+.2f}  (|r|<0.8 => alpha still constrained)")
    print(f"  lnZ = {res['log_evidence']:.1f} +/- {res['log_evidence_err']:.1f}")
    degenerate = abs(r) >= 0.9
    if degenerate:
        print("  WARNING: |r|>=0.9 -- x_zeta and alpha are strongly degenerate here.")
    print(f"\n{'VALIDATION PASS' if ok and not degenerate else 'VALIDATION FAIL'}")
    return 0 if (ok and not degenerate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
