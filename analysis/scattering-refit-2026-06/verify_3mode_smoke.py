"""Adversarial 3-mode smoke: run fit_joint_scattering at tiny nlive in
default (12-param), gain (8-param), scint-GP (10-param) modes on synthetic
two-band data. Confirms no import/signature breakage and all three paths run
end-to-end and return well-formed posteriors with the expected param_names.
Also re-runs the joint demo() and an explicit dead-channel / degenerate-cov
stress on the GP likelihood.
"""
import os
import sys
import numpy as np

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams, _gp_amplitude_logL
from scat_analysis.burstfit_joint import (
    fit_joint_scattering, demo,
    JOINT_PARAM_NAMES, JOINT_PARAM_NAMES_GAIN, JOINT_PARAM_NAMES_GAIN_GP,
)

rng = np.random.default_rng(11)


def make(fmin, fmax, nch, ntime=200, snr=25.0):
    freq = np.linspace(fmin, fmax, nch)
    time = np.arange(ntime) * 0.05
    base = FRBModel(time=time, freq=freq, data=np.zeros((nch, ntime)), dm_init=0.0)
    spec = 20.0 * (freq / np.median(freq)) ** (-1.0)
    p = FRBParams(c0=1.0, t0=time.mean(), gamma=0.0, zeta=0.3,
                  tau_1ghz=1.0, alpha=3.5, delta_dm=0.0)
    clean = spec[:, None] * base(p, "M3")
    noisy = clean + rng.normal(0, clean.max() / snr, clean.shape)
    m = FRBModel(time=time, freq=freq, data=noisy, dm_init=0.0)
    init = FRBParams(c0=float(spec.max()), t0=time.mean(), gamma=-1.0, zeta=0.3,
                     tau_1ghz=1.0, alpha=3.5, delta_dm=0.0)
    return m, init


mC, iC = make(0.40, 0.80, 48)
mD, iD = make(1.311, 1.499, 64)

common = dict(model_C=mC, init_C=iC, model_D=mD, init_D=iD,
              alpha_bounds=(2.0, 6.0), nlive=40, dlogz=2.0, nproc=1, verbose=False)

print("=== 3-MODE fit_joint_scattering smoke (nlive=40, nproc=1) ===")

r0 = fit_joint_scattering(**common)
assert r0["param_names"] == list(JOINT_PARAM_NAMES), "default names mismatch"
assert r0["samples"].shape[1] == 12, f"default ndim {r0['samples'].shape[1]} != 12"
assert np.isfinite(r0["log_evidence"]), "default logz non-finite"
print(f"default: ndim={r0['samples'].shape[1]} logZ={r0['log_evidence']:.1f} "
      f"alpha_med={r0['percentiles']['alpha']['median']:.2f} OK")

r1 = fit_joint_scattering(marginalize_gain=True, **common)
assert r1["param_names"] == list(JOINT_PARAM_NAMES_GAIN), "gain names mismatch"
assert r1["samples"].shape[1] == 8, f"gain ndim {r1['samples'].shape[1]} != 8"
assert np.isfinite(r1["log_evidence"]), "gain logz non-finite"
print(f"gain:    ndim={r1['samples'].shape[1]} logZ={r1['log_evidence']:.1f} "
      f"alpha_med={r1['percentiles']['alpha']['median']:.2f} OK")

r2 = fit_joint_scattering(marginalize_gain_gp=True, mu_degree=1, **common)
assert r2["param_names"] == list(JOINT_PARAM_NAMES_GAIN_GP), "gp names mismatch"
assert r2["samples"].shape[1] == 10, f"gp ndim {r2['samples'].shape[1]} != 10"
assert np.isfinite(r2["log_evidence"]), "gp logz non-finite"
assert "Delta_nu_d_C" in r2["param_names"] and "Delta_nu_d_D" in r2["param_names"]
print(f"scint:   ndim={r2['samples'].shape[1]} logZ={r2['log_evidence']:.1f} "
      f"alpha_med={r2['percentiles']['alpha']['median']:.2f} "
      f"dnuC_med={r2['percentiles']['Delta_nu_d_C']['median']:.2f} "
      f"dnuD_med={r2['percentiles']['Delta_nu_d_D']['median']:.2f} OK")

print("\n=== joint demo() regression ===")
demo()

print("\n=== GP numerical stress: extreme Delta_nu_d + dead channels ===")
p = FRBParams(c0=1.0, t0=mD.time.mean(), gamma=0.0, zeta=0.3, tau_1ghz=1.0,
              alpha=3.5, delta_dm=0.0)
chan_w = float(np.median(np.diff(mD.freq))) * 1e3
band = float(mD.freq.max() - mD.freq.min()) * 1e3
for label, dnu in [("tiny 1e-4*chan", 1e-4 * chan_w),
                   ("0.3*chan(bound)", 0.3 * chan_w),
                   ("huge 100*band", 100.0 * band),
                   ("1e9 MHz", 1e9)]:
    val = mD.log_likelihood_gain_marginal_gp(p, "M3", delta_nu_d_MHz=dnu, mu_degree=1)
    flag = "FINITE" if np.isfinite(val) else "NONFINITE!!"
    print(f"  dnu={label:18s} ({dnu:.4g} MHz): logL={val:.3f} [{flag}]")
    assert np.isfinite(val), f"non-finite at {label}"

# near-degenerate cov: large dnu -> C ~ all-ones rank-1, whitened M ill-conditioned
md = mD
md2_noise = md.noise_std.copy()
md2_noise[3] = 0.0; md2_noise[10] = 0.0; md2_noise[40] = 0.0
mD2 = FRBModel(time=md.time, freq=md.freq, data=md.data, dm_init=0.0,
               noise_std=md2_noise)
v3 = mD2.log_likelihood_gain_marginal_gp(p, "M3", delta_nu_d_MHz=1e6, mu_degree=1)
print(f"  3 dead ch + dnu=1e6 (rank-1 C): logL={v3:.3f} "
      f"[{'FINITE' if np.isfinite(v3) else 'NONFINITE!!'}]")
assert np.isfinite(v3), "dead+degenerate non-finite"

# direct _gp_amplitude_logL with NaN-prone input: zero variance channel guard
ah = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
vv = np.array([0.01, 0.01, 0.0, 0.01, 0.01])  # one zero -> clipped to 1e-30
nu = np.array([1400., 1401., 1402., 1403., 1404.])
ll, s2, mu, mod = _gp_amplitude_logL(ah, vv, nu, 2.0, mu_degree=1)
print(f"  _gp_amplitude_logL w/ zero-var ch: logL={ll:.3f} s2={s2:.3g} "
      f"mod={mod:.3f} [{'FINITE' if np.isfinite(ll) else 'NONFINITE!!'}]")
assert np.isfinite(ll), "zero-var direct non-finite"

print("\nSMOKE ALL OK: 3 modes run, demo passes, GP numerically robust under "
      "extreme Delta_nu_d / dead / degenerate-cov")
