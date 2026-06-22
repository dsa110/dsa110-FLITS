#!/usr/bin/env python
"""Self-check the gain-marginalized joint likelihood.

Injects per-channel scintillation (random gains) on top of a common-(tau,alpha)
scattered burst in two bands, then verifies:
  1. the gain-marginal joint logL prefers the TRUE alpha over a wrong alpha
     (scintillation must NOT fool it -- the whole point), and beats wrong tau;
  2. gain_spectrum recovers the injected per-channel gains (corr > 0.9);
  3. at truth, the gain-marginal residuals are ~white (std ~ 1), whereas the
     fixed-amplitude (smooth-spectrum) likelihood leaves scintillation in the
     residual (std >> 1) -- demonstrating the fix.
"""
import os, sys
import numpy as np
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams
from scat_analysis.burstfit_joint import _JointLogLikelihoodGain, _JointLogLikelihood

rng = np.random.default_rng(1)
tau_true, alpha_true = 1.0, 3.0

def make(fmin, fmax, nch, scint_scale):
    freq = np.linspace(fmin, fmax, nch)
    time = np.arange(240) * 0.05
    # truth with a smooth spectrum gamma, then MULTIPLY by per-channel scintillation
    p = FRBParams(c0=20.0, t0=time.mean(), gamma=-1.0, zeta=0.3,
                  tau_1ghz=tau_true, alpha=alpha_true, delta_dm=0.0)
    base = FRBModel(time=time, freq=freq, data=np.zeros((nch, time.size)), dm_init=0.0)
    clean = base(p, "M3")
    gains = np.clip(1.0 + scint_scale * rng.standard_normal(nch), 0.1, None)  # scintillation
    clean = clean * gains[:, None]
    noisy = clean + rng.normal(0, 0.05 * clean.max(), clean.shape)
    return FRBModel(time=time, freq=freq, data=noisy, dm_init=0.0), gains

mC, gC = make(0.40, 0.80, 16, 0.6)
mD, gD = make(1.31, 1.50, 16, 0.6)
llg = _JointLogLikelihoodGain(mC, mD)

# 8-vector: [tau, alpha, t0_C, zeta_C, ddm_C, t0_D, zeta_D, ddm_D]
t0 = mC.time.mean()
def vec(alpha, tau=tau_true):
    return np.array([tau, alpha, t0, 0.3, 0.0, mD.time.mean(), 0.3, 0.0])

ll_true = llg(vec(alpha_true))
ll_wrong_a = llg(vec(alpha_true + 1.5))
ll_wrong_t = llg(vec(alpha_true, tau=tau_true * 4))
print(f"1) gain-marginal logL: true_a={ll_true:.1f}  wrong_a={ll_wrong_a:.1f}  wrong_tau={ll_wrong_t:.1f}")
assert ll_true > ll_wrong_a, "scintillation fooled alpha!"
assert ll_true > ll_wrong_t, "tau not preferred"

# 2) gain recovery
pC = FRBParams(c0=1.0, t0=t0, gamma=0.0, zeta=0.3, tau_1ghz=tau_true, alpha=alpha_true, delta_dm=0.0)
g_rec = mC.gain_spectrum(pC, "M3")
# injected gains are modulated by the true smooth spectrum c0*(f/fref)^gamma; the
# recovered gain absorbs both, so compare to gC * that spectrum (monotone corr).
ref = np.median(mC.freq); spec = 20.0 * (mC.freq / ref) ** (-1.0)
corr = np.corrcoef(g_rec, gC * spec)[0, 1]
print(f"2) gain recovery corr(recovered, injected*spectrum) = {corr:.3f}")
assert corr > 0.9, f"gain not recovered: corr={corr}"

# 3) whiteness: gain-marginal residual std ~1 vs fixed-amplitude residual >> 1
def resid_std_gain(m, p):
    from dataclasses import replace
    K = m(replace(p, c0=1.0, gamma=0.0), "M3")
    g = m.gain_spectrum(p, "M3")
    r = (m.data - g[:, None] * K) / np.clip(m.noise_std, 1e-9, None)[:, None]
    return r.std()
def resid_std_fixed(m, p):  # smooth-spectrum model, no per-channel gain
    r = (m.data - m(p, "M3")) / np.clip(m.noise_std, 1e-9, None)[:, None]
    return r.std()
pCt = FRBParams(c0=20.0, t0=t0, gamma=-1.0, zeta=0.3, tau_1ghz=tau_true, alpha=alpha_true, delta_dm=0.0)
sg, sf = resid_std_gain(mC, pCt), resid_std_fixed(mC, pCt)
print(f"3) CHIME resid std: gain-marginal={sg:.2f}  fixed-amplitude={sf:.2f}")
assert sg < sf, "gain marginal did not whiten residual"
assert sg < 1.5, f"gain-marginal residual not white: {sg}"
print("ALL OK: gain-marginal likelihood is scintillation-robust, recovers gains, whitens residuals")
