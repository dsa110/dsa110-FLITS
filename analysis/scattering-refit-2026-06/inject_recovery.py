#!/usr/bin/env python
"""Sampler-level injection-recovery for the multi-component gain-marginal joint fit.

The self-check validated the LIKELIHOOD FORM. This runs the actual dynesty
sampler end-to-end to prove the SCIENCE claim: a hidden 2nd pulse biases the
single-component (alpha) fit, and the 2-component fit recovers the true alpha +
both pulses -- i.e. modeling the extra pulse un-biases/un-rails alpha.

Inject two temporal components per band sharing (tau_1ghz, alpha); fit with
components=1 (expect alpha biased) and components=2 (expect alpha recovered,
lnZ(2) >> lnZ(1)). Synthetic only; no HPCC.
"""
import os
import sys

import numpy as np

REPO = "/Users/jakobfaber/Developer/repos/github.com/dsa110/dsa110-FLITS"
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams
from scat_analysis.burstfit_init import data_driven_initial_guess
from scat_analysis.burstfit_joint import fit_joint_scattering

# ---- truth ----
TAU_TRUE, ALPHA_TRUE = 0.20, 3.5          # interior alpha (prior (1.5,6))
SEP_MS = 1.5                               # pulse separation (resolvable both bands)
rng = np.random.default_rng(7)


def make_band(fmin, fmax, nch, nu_label):
    """Two-pulse synthetic band: clean = K(p_a)+K(p_b), shared (tau,alpha)."""
    freq = np.linspace(fmin, fmax, nch)
    time = np.arange(260) * 0.05
    m0 = FRBModel(time=time, freq=freq, data=np.zeros((nch, time.size)), dm_init=0.0)
    t0a = time.mean() - SEP_MS / 2
    t0b = time.mean() + SEP_MS / 2
    pa = FRBParams(c0=24.0, t0=t0a, gamma=-1.4, zeta=0.30,
                   tau_1ghz=TAU_TRUE, alpha=ALPHA_TRUE, delta_dm=0.0)
    pb = FRBParams(c0=14.0, t0=t0b, gamma=+0.8, zeta=0.22,   # diff amp+spectrum
                   tau_1ghz=TAU_TRUE, alpha=ALPHA_TRUE, delta_dm=0.0)
    clean = m0(pa, "M3") + m0(pb, "M3")
    noisy = clean + rng.normal(0, 0.05 * clean.max(), clean.shape)
    m = FRBModel(time=time, freq=freq, data=noisy, dm_init=0.0)
    init = data_driven_initial_guess(data=m.data, freq=m.freq, time=m.time,
                                     dm=0.0, verbose=False).params
    print(f"  [{nu_label}] {nch}ch {fmin}-{fmax}GHz  true t0 = {t0a:.2f},{t0b:.2f} ms  init.t0={init.t0:.2f}")
    return m, init, (t0a, t0b)


def run(label, model_C, init_C, model_D, init_D, n_C, n_D):
    res = fit_joint_scattering(
        model_C=model_C, init_C=init_C, model_D=model_D, init_D=init_D,
        alpha_bounds=(1.5, 6.0), nlive=500, dlogz=0.5, nproc=4,
        marginalize_gain=True, components_C=n_C, components_D=n_D, verbose=False,
    )
    a = res["percentiles"]["alpha"]
    t = res["percentiles"]["tau_1ghz"]
    lnz = res["log_evidence"]
    print(f"\n[{label}] components C={n_C} D={n_D}  ndim={len(res['param_names'])}")
    print(f"  alpha = {a['median']:.3f} (+{a['err_plus']:.3f}/-{a['err_minus']:.3f})   "
          f"[truth {ALPHA_TRUE}]   bias = {a['median']-ALPHA_TRUE:+.3f}")
    print(f"  tau   = {t['median']:.3f} (+{t['err_plus']:.3f}/-{t['err_minus']:.3f})   [truth {TAU_TRUE}]")
    print(f"  lnZ   = {lnz:.2f} +/- {res['log_evidence_err']:.2f}")
    if n_D > 1:
        for s in ("C", "D"):
            ts = [res["percentiles"].get(f"t0_{s}_{i}", {}).get("median") for i in (1, 2)]
            print(f"  band {s} recovered t0 = {[round(x,2) for x in ts if x is not None]}")
    return a["median"], lnz


def main():
    print("=" * 70)
    print(f"INJECTION: 2 pulses/band, sep={SEP_MS}ms, shared tau={TAU_TRUE} alpha={ALPHA_TRUE}")
    mC, iC, tC = make_band(0.50, 0.80, 24, "CHIME")
    mD, iD, tD = make_band(1.31, 1.50, 24, "DSA")

    a1, z1 = run("1-COMPONENT (wrong model)", mC, iC, mD, iD, 1, 1)
    a2, z2 = run("2-COMPONENT (correct model)", mC, iC, mD, iD, 2, 2)

    print("\n" + "=" * 70)
    dlnz = z2 - z1
    bias1, bias2 = abs(a1 - ALPHA_TRUE), abs(a2 - ALPHA_TRUE)
    railed1 = (a1 < 1.7) or (a1 > 5.8)
    print(f"alpha bias:  1-comp {bias1:+.3f}  ->  2-comp {bias2:+.3f}   (2-comp must be smaller)")
    print(f"1-comp alpha railed (near 1.5/6.0 bound)? {railed1}  (alpha_1comp={a1:.3f})")
    print(f"delta-lnZ (2 vs 1) = {dlnz:+.2f}   (must be >> +5 to select 2 components)")
    ok = (bias2 < 0.6) and (bias2 < bias1) and (dlnz > 5)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} -- "
          f"2-comp recovers alpha (bias {bias2:.2f}<0.6), beats 1-comp bias ({bias2:.2f}<{bias1:.2f}), "
          f"and lnZ decisively prefers 2 (dlnZ={dlnz:.1f}>5)")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
