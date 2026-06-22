#!/usr/bin/env python
"""Diagnose GATE 2: is the lnZ(2) rise toward merge under FIXED large s2 the bug,
or an artifact? Compare to the OLD flat improper prior to ground the magnitude,
and check the eigenvalue guard + ML-profiled production behavior at the SAME points.
"""
import sys
from dataclasses import replace
import numpy as np

sys.path.insert(0, "/Users/jakobfaber/Developer/repos/github.com/dsa110/dsa110-FLITS/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams  # noqa: E402
from scat_analysis.burstfit_joint import _gain_marginal_multi_band  # noqa: E402

TAU, ALPHA, ZETA = 0.5, 4.0, 0.18


def comp(t0):
    return FRBParams(c0=1.0, t0=t0, gamma=0.0, zeta=ZETA,
                     tau_1ghz=TAU, alpha=ALPHA, delta_dm=0.0)


def noise_band(freq_GHz, n_time, noise, seed):
    rng = np.random.default_rng(seed)
    freq = np.asarray(freq_GHz, dtype=float)
    time = np.linspace(-6.0, 6.0, n_time)
    m = FRBModel(time=time, freq=freq, data=None, noise_std=None)
    m.dm_init = 0.0
    F = freq.size
    m.data = rng.normal(0.0, noise, size=(F, n_time))
    m.noise_std = np.full(F, noise)
    m.valid = np.ones(F, dtype=bool)
    return m


def flat_improper_multi(model, params_list, model_keys):
    """The OLD BUG: flat improper gain prior. Per channel:
       chi2min = (S_dd - b^T M^-1 b)/var ; occam = -0.5 ln det M.
    As two kernels merge M->singular, ln det M -> -inf so -0.5 ln det M -> +inf
    (the +N ln s2 divergence analogue) and |g|=M^-1 b blows up. Reproduces the
    +20..+324 nat reward the spec describes."""
    valid = model.valid
    Ks = np.stack([model(replace(p, c0=1.0, gamma=0.0), mk, freq_subset=valid)
                   for p, mk in zip(params_list, model_keys)])
    N, F, T = Ks.shape
    d = model.data[valid]
    var = np.clip(model.noise_std[valid], 1e-9, None) ** 2
    S_dd = np.einsum("ft,ft->f", d, d)
    b = np.einsum("nft,ft->fn", Ks, d)
    M = np.einsum("nft,mft->fnm", Ks, Ks)
    lnZ = 0.0
    maxg = 0.0
    for f in range(F):
        Mf = M[f]
        try:
            gf = np.linalg.solve(Mf, b[f])
        except np.linalg.LinAlgError:
            gf = np.linalg.lstsq(Mf, b[f], rcond=None)[0]
        quad = b[f] @ gf
        sign, logdet = np.linalg.slogdet(Mf)
        if sign <= 0:
            logdet = np.log(max(np.linalg.det(Mf), 1e-300))
        chi2min = (S_dd[f] - quad) / var[f]
        occam = -0.5 * logdet
        lnZ += -0.5 * chi2min + occam + 0.5 * np.log(2 * np.pi * var[f])
        maxg = max(maxg, float(np.max(np.abs(gf))))
    return lnZ, maxg


DTS = np.array([0.3, 0.1, 0.02, 0.005, 0.001])

m = noise_band(np.linspace(1.28, 1.53, 64), 320, 0.05, seed=404)
t0 = 0.0

print("Compare 2-component evidence vs dt on PURE NOISE under 3 priors.")
print("net = lnZ(2) - lnZ(1) under the SAME prior. The bug = net INCREASES")
print("(toward +) and max|g| explodes as dt -> 0.\n")

for label, kw in (("FLAT IMPROPER (old bug)", None),
                  ("PROPER fixed s2=1e6", 1e6),
                  ("PROPER ML-profiled (production)", "ML")):
    print(f"=== {label} ===")
    if kw is None:
        lnZ1, mg1 = flat_improper_multi(m, [comp(t0)], ["M3"])
    elif kw == "ML":
        lnZ1, d1 = _gain_marginal_multi_band(m, [comp(t0)], ["M3"], s2=None)
    else:
        lnZ1, d1 = _gain_marginal_multi_band(m, [comp(t0)], ["M3"], s2=kw)
    print(f"  lnZ(1) = {lnZ1:.2f}")
    print(f"  {'dt':>7} {'lnZ(2)':>13} {'net':>12} {'max|g|':>12}")
    prev_net = None
    for dt in DTS:
        ps = [comp(t0), comp(t0 + dt)]
        if kw is None:
            lnZ2, mg = flat_improper_multi(m, ps, ["M3", "M3"])
        elif kw == "ML":
            lnZ2, diag = _gain_marginal_multi_band(m, ps, ["M3", "M3"], s2=None)
            mg = max(diag["max_abs_g"])
        else:
            lnZ2, diag = _gain_marginal_multi_band(m, ps, ["M3", "M3"], s2=kw)
            mg = max(diag["max_abs_g"])
        net = lnZ2 - lnZ1
        print(f"  {dt:7.3f} {lnZ2:13.2f} {net:12.3f} {mg:12.3f}")
    print()
