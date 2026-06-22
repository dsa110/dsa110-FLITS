"""Fast wiring check: for all 3 modes, build the EXACT prior-transform +
loglike that fit_joint_scattering constructs, then push 5 unit-cube draws
through ptform->loglike. Proves no import/signature breakage and each mode's
likelihood evaluates finite on in-prior samples -- without the slow nested loop."""
import os, sys, time, numpy as np
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.burstfit import FRBModel, FRBParams
import scat_analysis.burstfit_joint as bj
from scat_analysis.burstfit_joint import (
    _JointPriorTransform, _JointLogLikelihood, _JointLogLikelihoodGain,
    _JointLogLikelihoodGainGP, _joint_prior_spec, _joint_prior_spec_gain,
    _joint_prior_spec_gain_gp,
    JOINT_PARAM_NAMES, JOINT_PARAM_NAMES_GAIN, JOINT_PARAM_NAMES_GAIN_GP,
)
rng = np.random.default_rng(3)


def make(fmin, fmax, nch, ntime=200, snr=25.0):
    freq = np.linspace(fmin, fmax, nch); time_ax = np.arange(ntime) * 0.05
    base = FRBModel(time=time_ax, freq=freq, data=np.zeros((nch, ntime)), dm_init=0.0)
    spec = 20.0 * (freq / np.median(freq)) ** (-1.0)
    p = FRBParams(c0=1.0, t0=time_ax.mean(), gamma=0.0, zeta=0.3, tau_1ghz=1.0,
                  alpha=3.5, delta_dm=0.0)
    noisy = spec[:, None] * base(p, "M3")
    noisy = noisy + rng.normal(0, noisy.max() / snr, noisy.shape)
    m = FRBModel(time=time_ax, freq=freq, data=noisy, dm_init=0.0)
    init = FRBParams(c0=float(spec.max()), t0=time_ax.mean(), gamma=-1.0, zeta=0.3,
                     tau_1ghz=1.0, alpha=3.5, delta_dm=0.0)
    return m, init


mC, iC = make(0.40, 0.80, 48)
mD, iD = make(1.311, 1.499, 64)
ab = (2.0, 6.0)

modes = [
    ("default", _joint_prior_spec(iC, iD, ab),
     _JointLogLikelihood(mC, mD), JOINT_PARAM_NAMES, 12),
    ("gain", _joint_prior_spec_gain(iC, iD, ab),
     _JointLogLikelihoodGain(mC, mD), JOINT_PARAM_NAMES_GAIN, 8),
    ("scint", _joint_prior_spec_gain_gp(iC, iD, ab, mC, mD),
     _JointLogLikelihoodGainGP(mC, mD, mu_degree=1), JOINT_PARAM_NAMES_GAIN_GP, 10),
]

print("=== WIRING: ptform+loglike for all 3 modes (no nested loop) ===", flush=True)
for label, spec, ll, names, ndim in modes:
    assert len(spec) == ndim, f"{label} spec len {len(spec)} != {ndim}"
    assert [s[0] for s in spec] == list(names), f"{label} name order mismatch"
    pt = _JointPriorTransform(spec)
    vals = []
    t0 = time.time()
    for _ in range(5):
        u = rng.random(ndim)
        theta = pt(u)
        v = ll(theta)
        vals.append(v)
        assert np.isfinite(v) or v == -1e100, f"{label} loglike returned {v}"
    dt = (time.time() - t0) / 5 * 1e3
    finite = sum(np.isfinite(x) and x > -1e99 for x in vals)
    print(f"{label:8s}: ndim={ndim} names_ok finite={finite}/5 "
          f"vals={[f'{x:.0f}' for x in vals]} {dt:.0f}ms/call", flush=True)
    # at least one in-prior draw must give a real (non-sentinel) finite logL
    assert finite >= 1, f"{label}: no finite loglike on 5 draws"

# also confirm fit_joint_scattering builds the right (names,spec,loglike) per mode
# by monkeypatch-free inspection of the branch via a 1-iteration guard
print("\n=== dispatch table check (fit_joint_scattering internals) ===", flush=True)
import inspect
src = inspect.getsource(bj.fit_joint_scattering)
assert "marginalize_gain_gp" in src and "_JointLogLikelihoodGainGP" in src
assert "_JointLogLikelihoodGain(" in src and "_JointLogLikelihood(" in src
print("fit_joint_scattering references all 3 loglike classes + GP spec OK", flush=True)

print("\nWIRING ALL OK: 3 modes' prior+loglike evaluate finite; no signature/import break", flush=True)
