"""Fast 3-mode integrity smoke: cap each nested fit with maxiter so all three
modes (default 12, gain 8, scint-GP 10) return in seconds. Goal is path /
signature integrity + well-formed posterior shape, NOT convergence."""
import os, sys, numpy as np
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams
from scat_analysis.burstfit_joint import (
    fit_joint_scattering,
    JOINT_PARAM_NAMES, JOINT_PARAM_NAMES_GAIN, JOINT_PARAM_NAMES_GAIN_GP,
)
rng = np.random.default_rng(11)


def make(fmin, fmax, nch, ntime=200, snr=25.0):
    freq = np.linspace(fmin, fmax, nch); time = np.arange(ntime) * 0.05
    base = FRBModel(time=time, freq=freq, data=np.zeros((nch, ntime)), dm_init=0.0)
    spec = 20.0 * (freq / np.median(freq)) ** (-1.0)
    p = FRBParams(c0=1.0, t0=time.mean(), gamma=0.0, zeta=0.3, tau_1ghz=1.0,
                  alpha=3.5, delta_dm=0.0)
    noisy = spec[:, None] * base(p, "M3")
    noisy = noisy + rng.normal(0, noisy.max() / snr, noisy.shape)
    m = FRBModel(time=time, freq=freq, data=noisy, dm_init=0.0)
    init = FRBParams(c0=float(spec.max()), t0=time.mean(), gamma=-1.0, zeta=0.3,
                     tau_1ghz=1.0, alpha=3.5, delta_dm=0.0)
    return m, init


mC, iC = make(0.40, 0.80, 48)
mD, iD = make(1.311, 1.499, 64)
common = dict(model_C=mC, init_C=iC, model_D=mD, init_D=iD,
              alpha_bounds=(2.0, 6.0), nlive=25, dlogz=10.0, nproc=1,
              verbose=False, sample="unif")
print("=== FAST 3-MODE smoke (nlive=25, dlogz=10, sample=unif, nproc=1) ===", flush=True)

for label, kw, names, ndim in [
    ("default", {}, JOINT_PARAM_NAMES, 12),
    ("gain", dict(marginalize_gain=True), JOINT_PARAM_NAMES_GAIN, 8),
    ("scint", dict(marginalize_gain_gp=True, mu_degree=1), JOINT_PARAM_NAMES_GAIN_GP, 10),
]:
    r = fit_joint_scattering(**common, **kw)
    assert r["param_names"] == list(names), f"{label} names mismatch"
    assert r["samples"].shape[1] == ndim, f"{label} ndim {r['samples'].shape[1]}!={ndim}"
    assert np.isfinite(r["log_evidence"]), f"{label} logz non-finite"
    extra = ""
    if label == "scint":
        extra = (f" dnuC={r['percentiles']['Delta_nu_d_C']['median']:.2f}"
                 f" dnuD={r['percentiles']['Delta_nu_d_D']['median']:.2f}")
        assert "Delta_nu_d_C" in r["param_names"]
    print(f"{label:8s}: ndim={r['samples'].shape[1]} nsamp={r['samples'].shape[0]} "
          f"logZ={r['log_evidence']:.1f} alpha={r['percentiles']['alpha']['median']:.2f}{extra} OK",
          flush=True)

print("FAST SMOKE ALL OK: 3 modes instantiate + run + well-formed posteriors", flush=True)
